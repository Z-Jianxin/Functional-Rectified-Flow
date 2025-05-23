import abc
from functools import partial
from typing import Any, Callable, TypeVar, Union

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jax.random import PRNGKeyArray
from omegaconf import DictConfig

from ..models import BaseMAML, BaseViT
from ..sdetools import SDE

# from ..utils.common import batch_mul

Params = FrozenDict[str, Any]
T = TypeVar("T")

from jax import vmap
from jax.scipy.stats import norm


class LogitNormalDistribution:
    def __init__(self, mean: float = 0.5, std: float = 1.0, eps: float = 1e-6):
        self.mean = mean
        self.std = std
        self.eps = eps

    def logit(self, x: jnp.ndarray) -> jnp.ndarray:
        """Numerically stable logit function."""
        x = jnp.clip(x, self.eps, 1 - self.eps)
        return jnp.log(x) - jnp.log1p(-x)

    def pdf(self, x: jnp.ndarray) -> jnp.ndarray:
        """Probability density function of the logit-normal distribution."""
        logit_x = self.logit(x)
        logit_pdf = norm.pdf(logit_x, loc=self.mean, scale=self.std)
        # Apply the change of variables formula: f(x) = g(h(x)) * |h'(x)|
        logit_derivative = 1 / (x * (1 - x) + self.eps)
        return logit_pdf * logit_derivative

    def batch_pdf(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the PDF for a batch of values."""
        return vmap(self.pdf)(x)


class ODEMSELoss(abc.ABC):
    """MSE loss for the ODE setting where the model output is noise-data.

    The loss is computed as the mean squared error between the model's predicted noise
    and the actual noise that was used to corrupt the clean data via the process:
        x_t = (1-t)*x + t*ε,
    where ε ~ N(0, I).
    """

    def __init__(self, sde: SDE, loss_config: DictConfig) -> None:
        """Initializes the MSELoss instance with SDE object and loss configuration.

        Args:
            sde (SDE): An object representing the stochastic differential equation.
            loss_config (DictConfig): A configuration object holding parameters for loss computation.
        """
        self.sde = sde
        self.loss_config = loss_config

    def construct_loss_fn(self, model: Union[BaseMAML, BaseViT]) -> Callable:
        """Constructs a loss function for ODE training.

        Args:
            model (Union[BaseMAML, BaseViT]): The model for which the loss function is being constructed.

        Returns:
            Callable: A function to compute the MSE loss given the necessary inputs.
        """
        update_params_fn = model.make_update_params_fn()

        reduce_op = jnp.mean if self.loss_config.reduce_mean else lambda *args, **kwargs: jnp.sum(*args, **kwargs)

        @jax.jit
        @partial(jax.grad, argnums=1, has_aux=True)
        def loss_fn(
            rng: PRNGKeyArray, params: Params, step: int, batch_input: jnp.ndarray, batch_real: jnp.ndarray
        ) -> T:
            """Computes the MSE loss for ODE between the predicted noise and the target noise.

            Args:
                rng (PRNGKeyArray): A random number generator.
                params (Params): Parameters of the model.
                step (int): Current training step.
                batch_input (jnp.ndarray): Input data batch.
                batch_real (jnp.ndarray): Clean data batch.

            Returns:
                T: A tuple containing the gradient of the loss and a tuple of auxiliary outputs.
            """
            b, g, c = batch_real.shape

            # Sample a time t for each instance.
            rng, step_rng = jax.random.split(rng)
            t = jax.random.uniform(step_rng, (b, 1), minval=0, maxval=self.sde.sde_config.T - self.sde.sde_config.eps)
            if self.loss_config.use_scheduler:
                t = 1 - t * (1 - jnp.exp(-step / self.loss_config.scheduler_steps))
            t_new = jnp.reshape(t, (b, 1, 1))
            t_new = jnp.broadcast_to(t_new, (b, g, 1))
            if self.loss_config.normalize_time:
                t_new = t_new * 2 - 1

            batch_input = batch_input.at[:, :, -1:].set(t_new)
            shape = self.sde.sde_config.shape
            rng, step_rng = jax.random.split(rng)
            # noise = jax.random.normal(rng, (b, g, c))
            # noise_freq = self.sde.fourier_transform(state=noise.reshape(b, *shape, c))

            rng, step_rng = jax.random.split(rng)
            mean, std = self.sde.marginal_prob(step_rng, batch_real, t)

            noise = self.sde.prior_sampling(step_rng, (b, g, c)).reshape(b, *shape, c)
            noise_std = std * noise
            # jnp.real(self.sde.inverse_fourier_transform(batch_mul(std, noise_freq)).reshape(b, g, c))
            batch_corrupted = mean + noise_std
            if self.loss_config.y_input:
                batch_input = batch_input.at[:, :, len(shape) : len(shape) + c].set(batch_corrupted.reshape(b, g, c))

            psm = self.sde.get_psm(t)
            new_rng, model_output, loss_inner = update_params_fn(rng, params, batch_input, batch_corrupted, psm)
            prediction = model_output.reshape(b, *shape, c)
            batch_corrupted = batch_corrupted.reshape(b, *shape, c)
            prediction = prediction.reshape(b, *shape, c)
            batch_real = batch_real.reshape(b, *shape, c)
            target = batch_real
            if self.sde.sde_config.predict_noise:
                target = noise

            if self.loss_config.frequency_space:
                prediction_freq = self.sde.fourier_transform(state=prediction)
                target_freq = self.sde.fourier_transform(state=target)
                if self.loss_config.outer_fftshift:
                    prediction_freq = jnp.fft.fftshift(prediction_freq)
                    target_freq = jnp.fft.fftshift(target_freq)
                if self.loss_config.outer_psm:
                    psm = self.sde.get_psm(t)
                    squared_loss = jnp.abs(prediction_freq * psm - target_freq * psm) ** 2
                else:
                    squared_loss = jnp.abs(prediction_freq - target_freq) ** 2
            else:
                if self.loss_config.outer_psm:
                    psm = self.sde.get_psm(t)
                    squared_loss = jnp.abs(prediction * psm - batch_real * psm) ** 2
                else:
                    squared_loss = jnp.abs(prediction - target) ** 2

            if self.loss_config.use_weights:
                logit_normal = LogitNormalDistribution(
                    mean=self.loss_config.lognormal_mean, std=self.loss_config.lognormal_std
                )
                t_flat = t.reshape(
                    b,
                )
                weights = logit_normal.batch_pdf(t_flat)
                if self.sde.sde_config.predict_noise:
                    weights = weights / (t_flat**2 + 1e-6)
                weights = weights.reshape((b, 1, 1))
                weights = jnp.broadcast_to(weights, (b, g, c)).reshape(b, *shape, c)
                squared_loss = weights * squared_loss

            losses = reduce_op(squared_loss.reshape(squared_loss.shape[0], -1), axis=-1)
            loss = jnp.mean(losses) / c
            reconstruct = prediction
            if self.sde.sde_config.predict_noise:
                reconstruct = (batch_corrupted - prediction * std) / (1 - std + 1e-6)
            return loss, (new_rng, loss, loss_inner, reconstruct, batch_corrupted, batch_real)

        return loss_fn
