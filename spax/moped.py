import numpy as np
import jax
import jax.numpy as jnp
from functools import partial


class SimpleMOPED:
    """Calculating MOPED compression of the data, for simple Gaussian likelihood.

    Here we assume non-changing, diagonal covariance matrix. To compute the compression,
    we use a general summary statistics form Alsing J., Wandelt B., MNRAS 476, L60-L64 (2018).
    """

    def __init__(self, devices=None):
        self.devices = devices

    def fit(self, data, batch_size=None):
        """Calculating mean and variance from the data.

        Args:
            data: array of simulations at the fiducial parameter value, of shape `(N_dim, N_samples)`.
            batch_size: split computation in `N_dim // batch_size` steps.
                `N_dim % (batch_size * N_devices) == 0`
        """
        N_dim, N_samples = data.shape
        if self.devices is None:
            self.μ = jnp.mean(data, axis=-1)
            self.var = jnp.var(data, axis=-1, ddof=1)
        else:
            N_devices = len(self.devices)
            batch_size = N_dim // N_devices if batch_size is None else batch_size
            data = data.reshape(
                N_dim // (N_devices * batch_size), N_devices, batch_size, N_samples
            )
            self.μ = jnp.array(
                [
                    jax.pmap(
                        partial(jnp.mean, axis=-1), devices=self.devices, backend="gpu"
                    )(d)
                    for d in data
                ],
                axis=0,
            ).flatten()

            self.var = jnp.array(
                [
                    jax.pmap(
                        partial(jnp.var, axis=-1, ddof=1),
                        devices=self.devices,
                        backend="gpu",
                    )(d)
                    for d in data
                ],
                axis=0,
            ).flatten()

    def compute(self, derivatives, δθ, batch_size=None):
        """Computing the summary from likelihood gradients.

        Args:
            derivatives: array of simulations around fiducial,
                from which derivatives of the likelihood will be computed.
                It is important that all 2 * len(δθ) simulations for one sample
                are calculated with the same "seed". Of shape `(N_dim, 2, len(δθ), N_samples)`
            δθ: diferential value of each parameter used to build `derivatives`.
            batch_size: split computation in `N_samples // batch_size` steps.
                `N_dim % (batch_size * N_devices) == 0`
        Retuns:
            t: summaries, of shape `(len(δθ), N_samples)
        """
        N_dim, _, N_t, N_samples = derivatives.shape
        if self.devices is None:

            @partial(jax.vmap, in_axes=(1,), out_axes=1)
            @partial(jax.vmap, in_axes=(1,), out_axes=1)
            @partial(jax.vmap, in_axes=(1,), out_axes=1)
            def logL(x):
                return self.logL(x, self.μ, self.var)

            L = logL(derivatives)

        else:
            N_devices = len(self.devices)
            batch_size = N_dim // N_devices if batch_size is None else batch_size
            derivatives = jnp.swapaxes(derivatives, 0, -1)
            derivatives = derivatives.reshape(
                N_samples // (batch_size * N_devices),
                N_devices,
                batch_size,
                2,
                N_t,
                N_dim,
            )

            @partial(jax.pmap, devices=self.gpus, backend="gpu")
            @jax.vmap
            @jax.vmap
            @jax.vmap
            def logL(x):
                return self.logL(x, self.μ, self.var)

            L = jnp.array([logL(d) for d in derivatives]).reshape(N_samples, 2, N_t)
            L = jnp.moveaxis(L, 0, -1)

        t = (L[1] - L[0]) / δθ[:, jnp.newaxis]
        return t

    @staticmethod
    @jax.jit
    def logL(x, μ, var):
        return -0.5 * jnp.sum((x - μ) ** 2 / var)
