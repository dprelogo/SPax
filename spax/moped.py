import numpy as np
import jax
import jax.numpy as jnp
from functools import partial


class ChisqLikelihoodMOPED:
    """Calculating MOPED compression of the data, for simple Gaussian likelihood.

    Here we assume non-changing, diagonal covariance matrix. To compute the compression,
    we use a general summary statistics form Alsing J., Wandelt B., MNRAS 476, L60-L64 (2018).
    """

    def __init__(self, devices=None):
        self.devices = devices

    def fit(self, data, batch_size=None):
        """Calculating mean and variance from the data.

        Args:
            data: array of simulations at the fiducial parameter value, of shape `(N_samples, N_dim)`.
            batch_size: split computation in `N_dim // batch_size` steps.
                `N_dim % (batch_size * N_devices) == 0`
        """
        N_samples, N_dim = data.shape
        if self.devices is None:
            self.μ = jnp.mean(data, axis=0)
            self.var = jnp.var(data, axis=0, ddof=1)
        else:
            N_devices = len(self.devices)
            data = data.swapaxes(0, -1)
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
                ]
            ).flatten()

            self.var = jnp.array(
                [
                    jax.pmap(
                        partial(jnp.var, axis=-1, ddof=1),
                        devices=self.devices,
                        backend="gpu",
                    )(d)
                    for d in data
                ]
            ).flatten()

    def compute(self, derivatives, δθ, batch_size=None):
        """Computing the summary from likelihood gradients.

        Args:
            derivatives: array of simulations around fiducial,
                from which derivatives of the likelihood will be computed.
                It is important that all 2 * len(δθ) simulations for one sample
                are calculated with the same "seed". Of shape `(N_samples, 2, len(δθ), N_dim)`
            δθ: diferential value of each parameter used to build `derivatives`.
            batch_size: split computation in `N_samples // batch_size` steps.
                `N_dim % (batch_size * N_devices) == 0`
        Retuns:
            t: summaries, of shape `(N_samples, len(δθ))
        """
        N_samples, _, N_t, N_dim = derivatives.shape
        if self.devices is None:

            @jax.vmap
            @jax.vmap
            @jax.vmap
            def logL(x):
                return self.logL(x, self.μ, self.var)

            L = logL(derivatives)

        else:
            N_devices = len(self.devices)
            batch_size = N_samples // N_devices if batch_size is None else batch_size
            derivatives = derivatives.reshape(
                N_samples // (batch_size * N_devices),
                N_devices,
                batch_size,
                2,
                N_t,
                N_dim,
            )

            @partial(jax.pmap, devices=self.devices, backend="gpu")
            @jax.vmap
            @jax.vmap
            @jax.vmap
            def logL(x):
                return self.logL(x, self.μ, self.var)

            L = jnp.array([logL(d) for d in derivatives]).reshape(N_samples, 2, N_t)

        t = (L[:, 1, :] - L[:, 0, :]) / δθ[jnp.newaxis, :]
        return t

    @partial(jax.jit, static_argnums=0)
    def logL(self, x, μ, var):
        return -0.5 * jnp.sum((x - μ) ** 2 / var)


class SimpleMOPED:
    """Calculating MOPED compression of the data, for Gaussian likelihood, from analytical expression.

    Here we assume non-changing, diagonal covariance matrix.
    """

    def __init__(self, devices=None):
        self.devices = devices

    def fit(self, data, derivatives, δθ, batch_size=None):
        """Calculating MOPED "eigenvectors" used for compression.

        Args:
            data: array of simulations at the fiducial parameter value, of shape `(N_samples, N_dim)`.
            derivatives: array of simulations around fiducial.
                It is important that all 2 * len(δθ) simulations for one sample
                are calculated with the same "seed". Of shape `(N_samples, 2, len(δθ), N_dim)`
            δθ: diferential value of each parameter used to build `derivatives`.
            batch_size: split computation in `N_dim // batch_size` steps.
                `N_dim % (batch_size * N_devices) == 0`
        """
        N_samples_data, N_dim_data = data.shape
        N_samples_der, _, N_t, N_dim_der = derivatives.shape

        if self.devices is None:
            self.μ = jnp.mean(data, axis=0)
            self.δμ = jnp.mean(derivatives, axis=0)
            self.var = jnp.var(data, axis=0, ddof=1)
        else:
            N_devices = len(self.devices)
            data = data.swapaxes(0, -1)
            derivatives = derivatives.swapaxes(0, -1)
            batch_size_data = (
                N_dim_data // N_devices if batch_size is None else batch_size
            )
            batch_size_der = (
                N_dim_der // N_devices if batch_size is None else batch_size
            )
            data = data.reshape(
                N_dim_data // (N_devices * batch_size),
                N_devices,
                batch_size,
                N_samples_data,
            )
            derivatives = derivatives.reshape(
                N_dim_der // (N_devices * batch_size),
                N_devices,
                batch_size,
                2,
                N_t,
                N_samples_der,
            )

            calculate_mean = jax.pmap(
                partial(jnp.mean, axis=-1), devices=self.devices, backend="gpu"
            )

            self.μ = jnp.array([calculate_mean(d) for d in data]).flatten()
            self.δμ = jnp.array([calculate_mean(d) for d in derivatives]).reshape(
                N_dim_der, 2, N_t
            )
            self.δμ = jnp.moveaxis(self.δμ, 0, -1)
            self.var = jnp.array(
                [
                    jax.pmap(
                        partial(jnp.var, axis=-1, ddof=1),
                        devices=self.devices,
                        backend="gpu",
                    )(d)
                    for d in data
                ]
            ).flatten()

        self.δμ = (self.δμ[1] - self.δμ[0]) / δθ[:, jnp.newaxis]

        B = []
        var_inv = 1 / self.var
        B.append(
            var_inv
            * self.δμ[0]
            / jnp.sqrt(jnp.einsum("i,i,i", self.δμ[0], var_inv, self.δμ[0]))
        )
        for i in range(1, N_t):
            projection = jnp.sum(
                jnp.stack([jnp.einsum("i,i", self.δμ[i], b) * b for b in B], axis=0),
                axis=0,
            )
            projection_norm = sum([jnp.einsum("i,i", self.δμ[i], b) ** 2 for b in B])
            vec = var_inv * self.δμ[i] - projection
            vec_norm = jnp.einsum("i,i,i", self.δμ[i], var_inv, self.δμ[i])
            B.append((vec - projection) / jnp.sqrt(vec_norm - projection_norm))

        self.B = jnp.array(B)

    def compute(self, data):
        """Computing the summary from likelihood gradients.

        Args:
            data: array of simulations for which compression should be computed.
                Of shape `(..., N_dim)`
        Retuns:
            t: summaries, of shape `(..., len(δθ))
        """
        return jnp.einsum("...i,...ji->...j", data, self.B)
