import abc
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKeyArray

from ..sdetools.base_sde import SDE
from ..sdetools.sde_utils import construct_b_and_r, rand_phase
from ..utils.common import batch_mul


def sobolev_weights_2d(shape, s, c):
    """Output Sobolev scaling weights for a 2D frequency grid.

    Parameters
    ----------
    shape : tuple of int
        Spatial dimensions (height, width) of the input.
    s : float
        Order of the Sobolev norm.
    c : float
        Scaling of the coordinate system (1 / pixel size).

    Returns
    -------
    scale : jnp.ndarray of shape (H, W)
        Sobolev scaling weights.
    """
    sy, sx = shape

    # Frequency coordinates
    fx = jnp.arange(sx)
    fx = jnp.minimum(fx, sx - fx) / (sx // 2)

    fy = jnp.arange(sy)
    fy = jnp.minimum(fy, sy - fy) / (sy // 2)

    X, Y = jnp.meshgrid(fx, fy)  # shape (H, W)

    freq_squared = X**2 + Y**2  # shape (H, W)
    base = 1 + c * freq_squared  # shape (H, W)

    base = base[None, :, :]  # shape (1, H, W)

    scale = base ** (s / 2)  # shape (B, H, W)

    return scale


class RectifiedODE(SDE, abc.ABC):
    """Heat RectifiedODE class representing a deterministic ODE for functional rectified flow.

    This subclass implements:
      - score_fn that returns 0,
      - a marginal probability corresponding to the process:
            x_t = (1-t) * X0 + t * X1,
        where X1 is a sample of noise.
      - an sde method that returns the drift computed from this interpolation and a zero diffusion.

    The get_psm and prior_sampling interfaces are kept.
    """

    def __init__(self, sde_config):
        """Initialize the HeatSubVPODE instance.

        Args:
            sde_config: A configuration object/dictionary containing at least:
                - shape: the shape of the data.
                - (optional) b: a parameter used for get_psm (default 1.0).
                - (optional) r: a parameter used for get_psm (default 1.0).
        """
        super().__init__(sde_config)
        self.shape = sde_config.shape
        self.is_unidimensional = len(self.shape) == 1
        self.x_norm = sde_config.x_norm
        self.energy_norm = sde_config.energy_norm
        self.b, self.r = construct_b_and_r(self.x_norm, self.energy_norm, shape=self.shape)

        self.sigma = sde_config.sigma  # parameter for Matern Kernel
        self.bandwidth = sde_config.l  # parameter for Matern Kernel
        self.prior_type = sde_config.prior_type
        if self.prior_type == "matern_onehalf":
            self.cached_cholesky = self._compute_and_cache_marten_cholesky()

    def _k(self, t: jnp.ndarray, t0: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        raise NotImplementedError("This method should not be called for functional rectified flow.")

    def get_psm(self, t: jnp.ndarray) -> jnp.ndarray:
        """Compute the PSM (Power-Special-Matrix) at time t.

        This implementation is kept unchanged from the original.

        Args:
            t: The current time as a jnp.ndarray.

        Returns:
            A jnp.ndarray weighting the loss function.
        """
        if not self.is_unidimensional:
            t = jnp.expand_dims(t, axis=(-1))
        if self.sde_config.psm_choice == "fdp" or self.sde_config.psm_choice is None:
            psm = jnp.expand_dims(jnp.ones_like(t) * jnp.sqrt(self.b / self.r).reshape(1, *self.shape), -1)
        elif self.sde_config.psm_choice == "all_ones":
            psm = jnp.ones((t.shape[0], *self.shape, 1))
        elif self.sde_config.psm_choice == "sobolev":
            psm = sobolev_weights_2d(self.shape, s=-t / 2.0, c=1.0)
            psm = jnp.expand_dims(psm, axis=(-1))
            assert psm.shape == (t.shape[0], *self.shape, 1)
        elif Path(self.sde_config.psm_choice).is_file():
            psm = jnp.array(np.load(self.sde_config.psm_choice))
            psm = jnp.expand_dims(psm, axis=(0, -1))
            assert psm.shape == (1, *self.shape, 1), f"Shape mismatch: expected {(1, *self.shape, 1)}, got {psm.shape}"
            psm = jnp.broadcast_to(psm, (t.shape[0], *self.shape, 1))
            assert psm.shape == (
                t.shape[0],
                *self.shape,
                1,
            ), f"Shape mismatch: expected {(t.shape[0], *self.shape, 1)}, got {psm.shape}"
        else:
            raise NotImplementedError("Unimplemented PSM type")
        return psm

    def score_fn(
        self,
        y_corrupted: jnp.ndarray,
        y_reconstructed: jnp.ndarray,
        t: jnp.ndarray,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        raise NotImplementedError("This method should not be called for functional rectified flow.")

    def _fdp_prior(
        self, rng: jax.random.PRNGKey, shape: Tuple[int, ...], t0: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Sample from the original FDP prior distribution.

        Args:
            rng: The random number generator key.
            shape: The shape of the sample to be generated.
            t0: Optional initial time.

        Returns:
            A sample from a standard normal distribution.
        """
        b, g, c = shape
        t = jnp.ones((b, 1))
        t = jnp.expand_dims(t, axis=(-1))
        if t0 is not None:
            t0 = jnp.expand_dims(t0, axis=(-1))
        # Generate the noise as a Gaussian random variable.
        z = jax.random.normal(key=rng, shape=shape)
        z_freq = self.fourier_transform(state=z.reshape(b, *self.shape, c))

        rng, step_rng = jax.random.split(rng)
        phase = rand_phase(step_rng, z_freq.shape)
        fact = (self.r / self.b).reshape(1, *self.b.shape, 1)
        fact = jnp.broadcast_to(fact, (b, *self.shape, 1))  # * (1 - self._k(t, t0)) ** 2
        sigma = jnp.sqrt(jnp.abs(fact)).reshape(b, *self.shape, 1)
        z = jnp.real(self.inverse_fourier_transform(state=batch_mul(batch_mul(sigma, z_freq), phase))).reshape(b, g, c)
        return z

    def _compute_and_cache_marten_cholesky(self):
        H, W = self.shape
        g = H * W
        xs = jnp.linspace(0, 1, H)
        ys = jnp.linspace(0, 1, W)
        grid_x, grid_y = jnp.meshgrid(xs, ys, indexing="ij")
        grid_coords = jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)  # shape (g, 2)

        # Compute pairwise distances
        diff = grid_coords[:, None, :] - grid_coords[None, :, :]  # (g, g, 2)
        dists = jnp.sqrt(jnp.sum(diff**2, axis=-1))  # (g, g)

        # Matérn (ν=0.5) kernel: exponential covariance
        K = self.sigma**2 * jnp.exp(-dists / self.bandwidth)
        jitter = 1e-6 * jnp.eye(g)
        K = K + jitter

        # Compute Cholesky factor (lower-triangular)
        L = jax.scipy.linalg.cholesky(K, lower=True)
        # Stop gradients so that L is treated as a concrete value.
        return jax.lax.stop_gradient(L)

    def _marten_kerel_prior(
        self, rng: jax.random.PRNGKey, shape: Tuple[int, ...], t0: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        # Unpack shape: b = batch size, g = number of grid points, c = channels.
        b, g, c = shape
        H, W = self.shape

        L = self.cached_cholesky

        # Sample from the standard normal: shape (batch, g, channels).
        z = jax.random.normal(rng, shape=(b, g, c))

        # Multiply by the Cholesky factor: for each sample, L @ z.
        samples = jnp.einsum("ij,bjc->bic", L, z)

        # Reshape to (batch, H, W, channels)
        # samples = samples_flat.reshape(b, H, W, c)
        return samples

    def prior_sampling(
        self, rng: jax.random.PRNGKey, shape: Tuple[int, ...], t0: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Sample from the prior distribution.

        Args:
            rng: The random number generator key.
            shape: The shape of the sample to be generated.
            t0: Optional initial time.

        Returns:
            A sample from a standard normal distribution.
        """
        if self.prior_type == "fdp":
            return self._fdp_prior(rng, shape, t0)
        elif self.prior_type == "matern_onehalf":
            return self._marten_kerel_prior(rng, shape, t0)
        elif self.prior_type == "iid":
            return jax.random.normal(key=rng, shape=shape)
        raise NotImplementedError(f"Does not recognize prior type {self.prior_type}")

    def marginal_prob(
        self,
        rng: jax.random.PRNGKey,
        x: jnp.ndarray,
        t: jnp.float32,
        t0: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the marginal probability for the ODE.

        This implements the process:
            x_t = (1-t) * x + t * noise,
        where `x` is the original state (X0) and `noise` is sampled from a standard normal.

        Args:
            rng: The random number generator key.
            x: The original state X0.
            t: The interpolation time (should be between 0 and 1).
            t0: Unused in this ODE; kept for interface consistency.

        Returns:
            A tuple (x_t, noise_std) where:
              - x_t is the interpolated state.
              - noise_std is set to t.
        """
        b, g, c = x.shape
        t = t.reshape(b, *[1] * (len(self.shape) + 1))
        t = jnp.broadcast_to(t, (b, *self.shape, 1))
        x0t = x.reshape(b, *self.shape, c) * t
        return x0t, 1.0 - t

    def sde(
        self,
        y_corrupted: jnp.ndarray,
        t: jnp.float32,
        rng: Optional[jax.random.PRNGKey] = None,
        y_reconstructed: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError("This method should not be called for functional rectified flow.")

    def get_reverse_noise(self, rng: PRNGKeyArray, shape: Tuple[int, ...]) -> jnp.ndarray:
        raise NotImplementedError("This method should not be called for functional rectified flow.")

    def diffuse(
        self, rng: PRNGKeyArray, x: jnp.ndarray, t: jnp.ndarray, t0: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError("This method should not be called for functional rectified flow.")
