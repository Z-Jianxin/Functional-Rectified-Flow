# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from functools import partial
from typing import Any, Callable, Union

import diffrax
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jax.random import PRNGKeyArray
from omegaconf import DictConfig

from ..sdetools import SDE
from . import Sampler
from .correctors.base_corrector import Corrector
from .predictors.base_predictor import Predictor

Params = FrozenDict[str, Any]


class ODESampler(Sampler, abc.ABC):
    """Predictor-Corrector (PC) Sampler.

    The PCInpainter class extends the generic Sampler class to provide a specialized
    implementation for handling image or audio generation task using a stochastic
    differential equation (SDE) based approach.

    Attributes:
        predictor (Predictor): An instance of the Predictor class for estimating the next state.
        corrector (Corrector): An instance of the Corrector class for refining the estimated state.
        sde (SDE): A stochastic differential equation model describing the process.
        sampler_config (DictConfig): A configuration object containing sampler settings.
    """

    def __init__(self, predictor: Predictor, corrector: Corrector, sde: SDE, sampler_config: DictConfig) -> None:
        """Initializes an instance of the PCSampler class.

        Args:
            predictor (Predictor): The predictor object for the sampler.
            corrector (Corrector): The corrector object for the sampler.
            sde (SDE): The SDE object for the sampler.
            sampler_config (DictConfig): The configuration for the sampler.
        """
        super().__init__(predictor=predictor, corrector=corrector, sde=sde, sampler_config=sampler_config)

    def make_sampler(self, predict_fn: Callable, super_resolution_fn: Union[Any, Callable]) -> Callable:
        """Creates a sampler function capable of performing sampling over a range of time.

        This method defines the process of how each sample is generated at each time step
        using the predictor-corrector scheme.

        Args:
            predict_fn (Callable): The model prediction function.
            super_resolution_fn (Callable): The super resolution function.

        Returns:
            Callable: A function that performs sampling over a range of time.
        """
        times = (
            jnp.linspace(0, self.sampler_config.T - self.sampler_config.eps, self.sampler_config.N + 1)
            ** self.sampler_config.k
        )

        @partial(jax.pmap, axis_name="device")
        def sample_fn(rng: PRNGKeyArray, batch_input: jnp.ndarray, params: Params) -> tuple[Any, Any, Any]:
            """Performs parallel sampling to generate a sequence of states over time.

            This function leverages parallel processing for the sampling
            process, and it outputs the final state, the reconstructed state,
            and the states at all time steps.

            Args:
                rng (PRNGKeyArray): Random number generator for stochastic processes.
                batch_input (jnp.ndarray): Input data batch.
                params (Params): Parameters for the model.

            Returns:
                Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: The final state,
                the reconstructed state, and the states at all time steps.
            """
            rng, step_rng = jax.random.split(rng)
            b, g, _ = batch_input.shape
            c = self.sampler_config.output_size
            batch_noise = self.sde.prior_sampling(step_rng, (b, g, c))

            def func(t, x, args):
                (eps,) = args
                vec_t = t * jnp.ones((x.shape[0], 1))
                shape = self.sde.sde_config.shape
                psm = self.sde.get_psm(vec_t)
                if self.sde.sde_config.predict_noise:
                    pred_eps = predict_fn(params, x, batch_input, vec_t, psm, shape)
                    pred_x1 = (x - pred_eps * (1 - t)) / (t + eps)
                else:
                    pred_x1 = predict_fn(params, x, batch_input, vec_t, psm, shape)
                if self.sampler_config.clip:
                    pred_x1 = jnp.clip(
                        pred_x1, a_min=self.sampler_config.clip_lower, a_max=self.sampler_config.clip_upper
                    )
                return (pred_x1 - x) / (1 - t + eps)

            term = diffrax.ODETerm(func)  # ODE term
            t0 = self.sampler_config.eps
            t1 = self.sampler_config.T - self.sampler_config.eps
            saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, self.sampler_config.N + 1))
            if self.sampler_config.solver == "Dopri5":
                solver = diffrax.Dopri5()
                stepsize_controller = diffrax.PIDController(
                    rtol=self.sampler_config.rtol, atol=self.sampler_config.atol
                )
                dt0 = None
            elif self.sampler_config.solver == "Euler":
                solver = diffrax.Euler()
                stepsize_controller = diffrax.ConstantStepSize()
                dt0 = self.sampler_config.dt0
            else:
                raise ValueError("Unimplemented solver type.")
            solution = diffrax.diffeqsolve(
                term,
                solver,
                t0=self.sampler_config.t0,
                t1=self.sampler_config.t1,
                dt0=dt0,
                y0=batch_noise,
                saveat=saveat,
                args=(1e-6,),
                stepsize_controller=stepsize_controller,
            )
            x_all_steps = solution.ys
            x = x_all_steps[-1]
            x_mean = x_all_steps[-1]

            t = times[-1]
            vec_t = t * jnp.ones((b, 1))
            psm = self.sde.get_psm(vec_t)
            shape = self.sde.sde_config.shape
            if self.sampler_config.do_super_resolution:
                if super_resolution_fn is None:
                    raise ValueError("Super resolution function is not provided for this model.")
                else:
                    y_reconstructed = super_resolution_fn(
                        params, x_mean, batch_input, vec_t, psm, shape, self.sampler_config.target_shape
                    )
            else:
                if self.sde.sde_config.predict_noise:
                    pred_eps = predict_fn(params, x_mean, batch_input, vec_t, psm, shape)
                    y_reconstructed = (x_mean - pred_eps * (1 - t)) / t
                else:
                    y_reconstructed = predict_fn(params, x_mean, batch_input, vec_t, psm, shape)
            return x, y_reconstructed, x_all_steps

        return sample_fn
