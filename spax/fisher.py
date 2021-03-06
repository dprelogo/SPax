import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from .pca import PCA, PCA_m


class Fisher:
    """Computing Fisher information in Jax.

    Attributes:
        devices: list of `jax.devices`. Set only for usage with GPU(s).
            If not given, computation is done on a default device.
            Probably desired behaviour is to put the following at the beginning of your script:
            ```
            import jax
            jax.config.update('jax_platform_name', 'cpu')
            ```
            and pass `jax.devices("gpu")` as devices.
        full_covariance: To use full covariance model for Fisher or just diagonal.

    Methods:
        fit: fitting the data and obtaining covariance matrix and `dμ_dθ`.
        compute: computing the Fisher information matrix, assuming multivariate Gaussian.
    """

    def __init__(self, devices=None, full_covariance=True):
        self.devices = devices
        self.full_covariance = full_covariance

    def fit(self, data, derivatives, δθ, batch_size=None, use_SVD=True):
        """Fitting the data.

        Args:
            data: used to compute covariance matrix, of shape `(N_dim, N_samples)`.
            derivatives: used to compute dμ_dθ, of shape `(N_dim, 2, len(δθ), N_samples)`.
            δθ: diferential value of each parameter used to build `derivatives`.
            batch_size: split computation in `N_dim // batch_size`.
                `N_dim % batch_size == 0`
            use_SVD: Etiher to use SVD decomposition while computing covariance
                inverse or not. See `spax.pca.PCA` for details.

        Returns:
            An instance of itself.
        """
        N_dim, N_samples = data.shape
        if self.devices is None:
            dμ_dθ = jnp.mean(
                (derivatives[:, 1, ...] - derivatives[:, 0, ...])
                / jnp.reshape(δθ, (1, -1, 1)),
                axis=-1,
                dtype=jnp.float64,
            ).astype(jnp.float32)

            if self.full_covariance:
                self.pca = PCA()
                self.pca.fit(data, use_SVD=use_SVD)
                self.Σ_inv = 1 / self.pca.eigenvalues
                self.UtJ = jnp.einsum(
                    "ki,kj->ij", self.pca.U, dμ_dθ, precision=jax.lax.Precision.HIGH
                ).astype(jnp.float32)
            else:
                var = jnp.var(data, ddof=1, axis=-1)
                self.Σ_inv = 1 / var
                self.UtJ = dμ_dθ
        else:
            n_d = len(self.devices)
            batch_size = N_dim // n_d if batch_size is None else batch_size

            @partial(jax.pmap, devices=self.devices, backend="gpu")
            @jax.jit
            def partial_dμ_dθ(der):
                return jnp.mean(
                    (der[:, 1, :] - der[:, 0, :]) / δθ, axis=-1, dtype=jnp.float64
                ).astype(jnp.float32)

            derivatives = derivatives.reshape(
                N_dim // (n_d * batch_size), n_d, batch_size, 2, len(δθ), N_samples
            )
            dμ_dθ = jnp.concatenate(
                jnp.array(
                    [jnp.concatenate(partial_dμ_dθ(x), axis=0) for x in derivatives],
                    dtype=jnp.float32,
                ),
                axis=0,
            )

            if self.full_covariance:
                self.pca = PCA_m(devices=self.devices)
                self.pca.fit(
                    data, batch_size=batch_size, centering_data="GPU", use_SVD=use_SVD
                )
                self.Σ_inv = 1 / self.pca.eigenvalues

                @partial(jax.pmap, in_axes=(0, 0), devices=self.devices, backend="gpu")
                @jax.jit
                def partial_UtJ(u, j):
                    return jnp.einsum(
                        "ki,kj->ij", u, j, precision=jax.lax.Precision.HIGH
                    ).astype(jnp.float32)

                U = self.pca.U.reshape(
                    N_dim // (n_d * batch_size), n_d, batch_size, N_samples
                )
                dμ_dθ = dμ_dθ.reshape(
                    N_dim // (n_d * batch_size), n_d, batch_size, len(δθ)
                )
                self.UtJ = jnp.sum(
                    jnp.array(
                        [jnp.sum(partial_UtJ(u, j), axis=0) for u, j in zip(U, dμ_dθ)]
                    ),
                    axis=0,
                ).astype(jnp.float32)
            else:
                var = jnp.var(data, ddof=1, axis=-1)
                self.Σ_inv = 1 / var
                self.UtJ = dμ_dθ

        return self

    def compute(self, N=None, return_matrix=False):
        """Computing Fisher information.

        Args:
            N: number of principal components on which Fisher is computed. If not set,
                has a maximal value of `min(N_dim, N_samples)`. In the case `N = N_dim <= N_samples`
                no information is lost. However for `N_dim > N_samples`, compression is necessary.
                It is probably better for N to be largen than the number of parameters, i.e. `len(δθ)`.
            return_matrix (bool): either to return full Fisher matrix or not.

        Returns:
            `det(F)` if `return_matrix == False`, else `(det(F), F)`.
        """
        N = len(self.Σ_inv) if N is None else N
        if N < 1 or N > len(self.Σ_inv):
            raise ValueError(f"N should be between 1 and {len(self.Σ_inv)}.")
        if not self.full_covariance and N != len(self.Σ_inv):
            raise ValueError(
                "If only diagonal covariance is used, compression is not needed and not implemented."
            )
        F = jnp.einsum(
            "ki,k,kj->ij",
            self.UtJ[:N],
            self.Σ_inv[:N],
            self.UtJ[:N],
            precision=jax.lax.Precision.HIGH,
        ).astype(jnp.float32)

        if return_matrix:
            return jnp.linalg.det(F), F
        else:
            return jnp.linalg.det(F)
