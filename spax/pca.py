"""PCA compression."""
from decimal import Decimal
from sqlite3 import DataError
import warnings
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import h5py
from . import kernels


class PCA:
    """PCA in jax.

    Assuming data and intermediate results fit on a local device (RAM or GPU memory).
    No additional setup needed.

    Attributes:
        N: number of principal components. If not given, keeps maximal number of components.

    Methods:
        fit: computing principal vectors.
        transform: calculating principal components for given input.
        inverse_transform: inverse of the transform.
        save: saving the transform.
        load: loading the transform.
        sample: sampling multivariate gaussian distribution of the principal components
            and computing inverse_transform.
    """

    def __init__(self, N=None):
        self.N = N

    def fit(self, data, whiten=False, use_SVD=False):
        """Computing eigenvectors and eigenvalues of the data.

        Args:
            data (np.array): data to fit, of shape `(N_dim, N_samples)`.
            whiten (bool): scaling all dimensions to the unit variance.
            use_SVD (bool): If true, it uses SVD decomposition, which might be
                more stable numerically.
        Returns:
            An instance of itself.
        """
        data = jnp.array(data, dtype=jnp.float32)
        N_dim, N_samples = data.shape
        if self.N is None:
            self.N = min(N_dim, N_samples)

        self.μ = jnp.mean(data, axis=1, keepdims=True, dtype=jnp.float64).astype(
            jnp.float32
        )
        if whiten:
            self.σ = jnp.std(data, axis=1, keepdims=True, dtype=jnp.float64).astype(
                jnp.float32
            )
        else:
            self.σ = jnp.ones((N_dim, 1), dtype=jnp.float32)

        data = (data - self.μ) / self.σ

        if N_dim < N_samples:
            C = jnp.einsum(
                "ik,jk->ij", data, data, precision=jax.lax.Precision.HIGH
            ) / (N_samples - 1)
            try:
                C = C.astype(jnp.float64)
            except:
                warnings.warn("Couldn't use float64 precision for covariance.")
                C = C.astype(jnp.float32)

            if use_SVD:
                self.U, self.eigenvalues, _ = jnp.linalg.svd(
                    C, full_matrices=False, hermitian=True
                )
            else:
                self.eigenvalues, self.U = jnp.linalg.eigh(C)
                self.eigenvalues = self.eigenvalues[::-1]
                self.U = self.U[:, ::-1]

            self.eigenvalues = self.eigenvalues[: self.N]
            if jnp.any(self.eigenvalues < 0):
                warnings.warn("Some eigenvalues are negative.")
            self.λ = jnp.sqrt(self.eigenvalues)
            self.U = self.U[:, : self.N]
        else:
            D = (
                jnp.einsum("ki,kj->ij", data, data, precision=jax.lax.Precision.HIGH)
                / N_dim
            )
            try:
                D = D.astype(jnp.float64)
            except:
                warnings.warn("Couldn't use float64 precision for covariance.")
                D = D.astype(jnp.float32)

            if use_SVD:
                V, self.eigenvalues, _ = jnp.linalg.svd(
                    D, full_matrices=False, hermitian=True
                )
            else:
                self.eigenvalues, V = jnp.linalg.eigh(D)
                self.eigenvalues = self.eigenvalues[::-1]
                V = V[:, ::-1]

            self.eigenvalues = self.eigenvalues[: self.N] * (N_dim / (N_samples - 1))
            if jnp.any(self.eigenvalues < 0):
                warnings.warn("Some eigenvalues are negative.")
            self.λ = jnp.sqrt(self.eigenvalues)
            S_inv = (1 / jnp.sqrt(self.eigenvalues * (N_samples - 1)))[jnp.newaxis, :]
            VS_inv = V[:, : self.N] * S_inv
            self.U = jnp.einsum(
                "ij,jk->ik", data, VS_inv, precision=jax.lax.Precision.HIGH
            ).astype(jnp.float32)

        return self

    def transform(self, X):
        """Transforming X and computing principal components for each sample.

        Args:
            X: data to transform of shape `(N_dim, N_samples)`.

        Returns:
            X_t: transformed data of shape `(N, N_samples)`.
        """
        X = jnp.array(X, dtype=jnp.float32)
        X_t = jnp.einsum("ji,jk->ik", self.U, (X - self.μ) / self.σ)
        return np.array(X_t, dtype=np.float32)

    def inverse_transform(self, X_t):
        """Transforming X_t back to the original space.

        Args:
            X_t: data in principal-components space, of shape `(N, N_samples)`.

        Returns:
            X: transformed data in original space, of shape `(N_dim, N_samples)`.
        """
        X_t = jnp.array(X_t, dtype=jnp.float32)
        X = jnp.einsum("ij,jk->ik", self.U, X_t) * self.σ + self.μ
        return np.array(X, dtype=np.float32)

    def sample(self, n=1):
        """Sample from the multivariate Gaussian prior
        and compute the inverse_transofrm of the pulled samples.

        Args:
            n: number of samples.

        Returns:
            X: sampled data in original space, of shape `(N_dim, n)`.
        """
        X_t = np.random.normal(size=(self.N, n)) * np.array(self.λ)[:, np.newaxis]
        return self.inverse_transform(X_t)

    def save(self, filename, compression_scheme={}):
        """Save the PCA fit as hdf5 file.

        Args:
            filename: name of the file.
            compression_scheme: dictionary containing compression options,
                eg. {"compression": "gzip", "compression_opts": 7, "shuffle": True}

        Returns:
            `None`
        """
        with h5py.File(filename, "w") as f:
            f.attrs["N"] = self.N
            for k in ["λ", "σ", "μ", "U"]:
                f.create_dataset(
                    k,
                    data=np.array(getattr(self, k), dtype=np.float32),
                    **compression_scheme,
                )

    def load(self, filename):
        """Load the PCA fit.

        Args:
            filename: name of the file.

        Returns:
            `None`
        """
        with h5py.File(filename, "r") as f:
            if self.N != f.attrs["N"]:
                raise ValueError(
                    "File contains PCA of order {}, which is different from {}.".format(
                        f.attrs["N"], self.N
                    )
                )
            for k in ["λ", "σ", "μ", "U"]:
                setattr(self, k, jnp.array(f[k], dtype=jnp.float32))


class PCA_m(PCA):
    """PCA in jax, for CPU + GPU environments.

    Designed to use CPU as a host. To configure properly, put the following
    at the beginnning of your script:
    ```
    import jax
    jax.config.update('jax_platform_name', 'cpu')
    ```
    and pass `jax.devices("gpu")` as devices.

    Attributes:
        N: number of principal components. If not given, keeps maximal number of components.
        devices: list of `jax.devices`.

    Methods:
        fit: computing principal vectors.
        transform: calculating principal components for given input.
        inverse_transform: inverse of the transform.
        save: saving the transform.
        load: loading the transform.
        sample: sampling multivariate gaussian distribution of the principal components
            and computing inverse_transform.
    """

    def __init__(self, devices, N=None):
        super().__init__(N)
        self.devices = devices

    def fit(
        self, data, batch_size=None, whiten=False, centering_data="CPU", use_SVD=False
    ):
        """Computing eigenvectors and eigenvalues of the data.

        Args:
            data: data to fit on of shape `(N_dim, N_samples)`.
            batch_size: splitting calculation in data chunks of `(N_dim / n_devices / batch_size, N_samples)`.
                Take care such matrix (+ data) can fit on one device.
                `N_dim % (N_devices * batch_size) == 0`, defaults to `N_dim / n_devices`.
            whiten (bool): scaling all dimensions to the unit variance.
            centering_data (str): either "CPU" or "GPU", where to perform data centering/whitening.
            use_SVD (bool): If true, it uses SVD decomposition, which might be
                more stable numerically.
        Returns:
            An instance of itself.
        """
        n_d = len(self.devices)
        N_dim, N_samples = data.shape
        if self.N is None:
            self.N = min(N_dim, N_samples)
        batch_size = N_dim // n_d if batch_size is None else batch_size
        if N_dim % (n_d * batch_size) != 0:
            raise ValueError(
                "N_dim of the data should be divisible by the n_devices * batch_size."
            )

        if centering_data == "CPU":
            data = data.astype(np.float32)
            self.μ = jnp.mean(data, axis=1, keepdims=True, dtype=jnp.float64).astype(
                jnp.float32
            )
            if whiten:
                self.σ = jnp.std(data, axis=1, keepdims=True, dtype=jnp.float64).astype(
                    jnp.float32
                )
            else:
                self.σ = jnp.ones(shape=self.μ.shape, dtype=np.float32)
            data = (data - self.μ) / self.σ
            data = data.reshape(N_dim // (n_d * batch_size), n_d, batch_size, N_samples)
        elif centering_data == "GPU":
            data = data.astype(np.float32).reshape(
                N_dim // (n_d * batch_size), n_d, batch_size, N_samples
            )

            @partial(jax.pmap, devices=self.devices, backend="gpu")
            @jax.jit
            def data_transform(d_part):
                μ_part = jnp.mean(
                    d_part, axis=1, keepdims=True, dtype=jnp.float64
                ).astype(jnp.float32)
                if whiten:
                    σ_part = jnp.std(
                        d_part, axis=1, keepdims=True, dtype=jnp.float64
                    ).astype(jnp.float32)
                    d_part = (d_part - μ_part) / σ_part
                else:
                    σ_part = jnp.ones(shape=μ_part.shape, dtype=jnp.float32)
                    d_part = d_part - μ_part
                return d_part, μ_part, σ_part

            data_transformed, μ, σ = [], [], []
            for d in data:
                d_part, μ_part, σ_part = data_transform(d)
                data_transformed.append(jnp.array(d_part, dtype=jnp.float32))
                μ.append(jnp.array(μ_part, dtype=jnp.float32))
                σ.append(jnp.array(σ_part, dtype=jnp.float32))
            self.μ = jnp.array(μ, dtype=jnp.float32).flatten()[:, jnp.newaxis]
            self.σ = jnp.array(σ, dtype=jnp.float32).flatten()[:, jnp.newaxis]
            data = jnp.array(data_transformed, dtype=jnp.float32)
        else:
            raise ValueError(
                f"centering_data is {centering_data}, should be either CPU or GPU."
            )

        if N_dim < N_samples:

            @partial(jax.pmap, in_axes=(0, None), devices=self.devices, backend="gpu")
            @jax.jit
            def partial_C(d1, d2):
                return (
                    jnp.einsum("ik,jk->ij", d1, d2, precision=jax.lax.Precision.HIGH)
                    / (N_samples - 1)
                ).astype(jnp.float32)

            C = []
            for d_y in data.reshape(N_dim // batch_size, batch_size, N_samples):
                row_C = []
                for d_x in data:
                    p_C = jnp.concatenate(partial_C(d_x, d_y), axis=1)
                    row_C.append(p_C)
                row_C = jnp.concatenate(row_C, axis=1)
                C.append(row_C)
            C = jnp.concatenate(C, axis=0)
            C = jax.device_put(C, self.devices[0])
            try:
                C = C.astype(jnp.float64)
            except:
                warnings.warn("Couldn't use float64 precision for covariance.")
                C = C.astype(jnp.float32)
            if use_SVD:
                self.U, self.eigenvalues, _ = jnp.linalg.svd(
                    C, full_matrices=False, hermitian=True
                )
            else:
                self.eigenvalues, self.U = jnp.linalg.eigh(C)
                self.eigenvalues = self.eigenvalues[::-1]
                self.U = self.U[:, ::-1]

            self.eigenvalues = self.eigenvalues[: self.N]
            if jnp.any(self.eigenvalues < 0):
                warnings.warn("Some eigenvalues are negative.")
            self.λ = jnp.sqrt(self.eigenvalues)
            self.U = self.U[:, : self.N]
        else:

            @partial(jax.pmap, in_axes=(0, 0), devices=self.devices, backend="gpu")
            @jax.jit
            def partial_D(d1, d2):
                return (
                    jnp.einsum("ki,kj->ij", d1, d2, precision=jax.lax.Precision.HIGH)
                    / N_dim
                ).astype(jnp.float32)

            D = jnp.sum(
                jnp.array(
                    [jnp.sum(partial_D(d1, d2), axis=0) for d1, d2, in zip(data, data)]
                ),
                axis=0,
            )
            D = jax.device_put(D, self.devices[0])
            try:
                D = D.astype(jnp.float64)
            except:
                warnings.warn("Couldn't use float64 precision for covariance.")
                D = D.astype(jnp.float32)
            if use_SVD:
                V, self.eigenvalues, _ = jnp.linalg.svd(
                    D, full_matrices=False, hermitian=True
                )
            else:
                self.eigenvalues, V = jnp.linalg.eigh(D)
                self.eigenvalues = self.eigenvalues[::-1]
                V = V[:, ::-1]

            self.eigenvalues = self.eigenvalues[: self.N] * (N_dim / (N_samples - 1))
            if jnp.any(self.eigenvalues < 0):
                warnings.warn("Some eigenvalues are negative.")
            self.λ = jnp.sqrt(self.eigenvalues)
            S_inv = (1 / jnp.sqrt(self.eigenvalues * (N_samples - 1)))[jnp.newaxis, :]
            VS_inv = V[:, : self.N] * S_inv

            @partial(jax.pmap, devices=self.devices, backend="gpu")
            @jax.jit
            def partial_U(d):
                return jnp.einsum(
                    "ij,jk->ik", d, VS_inv, precision=jax.lax.Precision.HIGH
                ).astype(jnp.float32)

            self.U = jnp.concatenate(
                [jnp.concatenate(partial_U(d)) for d in data], axis=0
            )

        return self

    def transform(self, X, batch_size=None):
        """Transforming X and computing principal components for each sample.

        Args:
            X: data to transform of shape `(N_dim, N_samples)`.
            batch_size: splitting calculation in data chunks of `(N_dim / n_devices / batch_size, N_samples)`.

        Returns:
            X_t: transformed data of shape `(N, N_samples)`.
        """
        n_d = len(self.devices)
        N_dim, N_samples = X.shape
        batch_size = N_dim // n_d if batch_size is None else batch_size
        X = X.astype(np.float32).reshape(
            N_dim // (n_d * batch_size), n_d, batch_size, N_samples
        )
        μ = self.μ.reshape(N_dim // (n_d * batch_size), n_d, batch_size, 1)
        σ = self.σ.reshape(N_dim // (n_d * batch_size), n_d, batch_size, 1)
        U = self.U.reshape(N_dim // (n_d * batch_size), n_d, batch_size, self.N)

        @partial(jax.pmap, in_axes=(0, 0, 0, 0), devices=self.devices, backend="gpu")
        @jax.jit
        def partial_transform(U, x, μ, σ):
            return jnp.einsum(
                "ji,jk->ik", U, (x - μ) / σ, precision=jax.lax.Precision.HIGH
            ).astype(jnp.float32)

        X_t = jnp.sum(
            jnp.array(
                [
                    jnp.sum(partial_transform(_u, _x, _μ, _σ), axis=0)
                    for _u, _x, _μ, _σ in zip(U, X, μ, σ)
                ]
            ),
            axis=0,
        )
        return np.array(X_t, dtype=np.float32)

    def inverse_transform(self, X_t, batch_size=None):
        """Transforming X_t back to the original space.

        Args:
            X_t: data in principal-components space, of shape `(N, N_samples)`.
            batch_size: splitting calculation in data chunks of `(N_dim / n_devices / batch_size, N_samples)`.

        Returns:
            X: transformed data in original space, of shape `(N_dim, N_samples)`.
        """
        n_d = len(self.devices)
        N_dim = self.U.shape[0]
        batch_size = N_dim // n_d if batch_size is None else batch_size
        X_t = jnp.array(X_t, dtype=jnp.float32)
        μ = self.μ.reshape(N_dim // (n_d * batch_size), n_d, batch_size, 1)
        σ = self.σ.reshape(N_dim // (n_d * batch_size), n_d, batch_size, 1)
        U = self.U.reshape(N_dim // (n_d * batch_size), n_d, batch_size, self.N)

        @partial(jax.pmap, in_axes=(0, 0, 0), devices=self.devices, backend="gpu")
        @jax.jit
        def partial_inv_transform(U, μ, σ):
            return (
                jnp.einsum(
                    "ij,jk->ik", U, X_t, precision=jax.lax.Precision.HIGH
                ).astype(jnp.float32)
                * σ
                + μ
            )

        X = jnp.concatenate(
            jnp.array(
                [
                    jnp.concatenate(partial_inv_transform(_u, _μ, _σ), axis=0)
                    for _u, _μ, _σ in zip(U, μ, σ)
                ]
            ),
            axis=0,
        )
        return np.array(X, dtype=np.float32)

    def sample(self, n=1, batch_size=None):
        """Sample from the multivariate Gaussian prior
        and compute the inverse_transofrm of the pulled samples.

        Args:
            n: number of samples.
            batch_size: splitting calculation in data chunks of `(N_dim / n_devices / batch_size, N_samples)`.
                Used only for the inverse_transform calculation.

        Returns:
            X: sampled data in original space, of shape `(N_dim, n)`.
        """
        X_t = np.random.normal(size=(self.N, n)) * np.array(self.λ)[:, np.newaxis]
        return self.inverse_transform(X_t, batch_size)


class KernelPCA:
    """Kernel PCA in jax.

    Assuming data and intermediate results fit on a local device (RAM or GPU memory).
    No additional setup needed.

    Attributes:
        N: number of principal components. If not given, keeps N_samples components.
        kernel: a kernel function, or string ["rbf", ]
        kernel_kwargs: arguments needed for a kernel specification
        inverse_kernel: a kernel function or string ["rbf", ]
            Which kernel to use for the inverse transform, None does simplest transformation, "same" takes the same function as kernel
        inverse_kernel_kwargs: possible arguments for inverse_kernel.

    Methods:
        fit: computing principal vectors.
        transform: calculating principal components for a given input.
        inverse_transform: inverse of the transform.
    """

    def __init__(
        self,
        N=None,
        α=1.0,
        kernel="rbf",
        kernel_kwargs=None,
        inverse_kernel=None,
        inverse_kernel_kwargs=None,
    ):
        self.N = N
        self.α = α

        self.vectorized = {}
        self._init_kernel(kernel, kernel_kwargs, "kernel")
        if inverse_kernel == "same":
            self._init_kernel(kernel, kernel_kwargs, "inverse_kernel")
        else:
            self._init_kernel(inverse_kernel, inverse_kernel_kwargs, "inverse_kernel")

    def _init_kernel(self, kernel, kernel_kwargs, name="kernel"):
        if kernel is None or callable(kernel):
            setattr(self, name, kernel)
        else:
            try:
                setattr(self, name, getattr(kernels, f"_{kernel}"))
            except:
                raise ValueError(f"Kernel {kernel} not supported")
            if kernel_kwargs is not None:
                setattr(self, name, partial(getattr(self, name), **kernel_kwargs))
        # self._vectorize_kernel(name)
        self.vectorized[name] = False

    def _vectorize_kernel(self, name):
        # if getattr(self, name) is not None and self.vectorized[name] == False:
        #     setattr(self, name, jax.vmap(jax.vmap(getattr(self, name), in_axes = (None, 1)), in_axes = (1, None)))
        #     self.vectorized[name] = True
        pass

    def _kernel_normalization(self, K):
        K_columns = jnp.sum(K, axis=1)[:, jnp.newaxis] / self.N_samples
        return K - self.K_rows - K_columns - self.K_0

    def _init_kernel_normalization(self, K):
        self.K_rows = jnp.sum(K, axis=0)[jnp.newaxis, :] / self.N_samples
        self.K_0 = jnp.sum(self.K_rows) / self.N_samples
        return self._kernel_normalization(K)

    def _fit_inverse_transform(self, K):
        #         φ = self.transform(self.data)
        # shortened expression
        self.φ = (self.V * self.λ).T

        if self.inverse_kernel is None:
            regularized_K = K + self.α * jnp.eye(K.shape[0])
            A = jnp.linalg.solve(regularized_K, self.data.T)
            self.W = jnp.einsum("ij,jk->ik", self.φ, A)
        else:
            K = self.inverse_kernel(self.φ, self.φ)
            regularized_K = K + self.α * jnp.eye(K.shape[0])
            self.W = jnp.linalg.solve(regularized_K, self.data.T)

    def fit(self, data, use_SVD=False):
        """Computing eigenvectors and eigenvalues of the data.

        Args:
            data (np.array): data to fit, of shape `(N_dim, N_samples)`.
            use_SVD (bool): If true, it uses SVD decomposition, which might be
                more stable numerically.

        Returns:
            An instance of itself.
        """
        self.data = jnp.array(data, dtype=jnp.float32)
        self.N_dim, self.N_samples = data.shape
        self.N = self.N or self.N_samples

        # self._vectorize_kernel("kernel")

        K = self.kernel(self.data, self.data)
        K = self._init_kernel_normalization(K)

        if use_SVD:
            U, λ, _ = jnp.linalg.svd(K, full_matrices=False, hermitian=True)
        else:
            λ, U = jnp.linalg.eigh(K)
            λ = λ[::-1]
            U = U[:, ::-1]

        self.λ = λ[: self.N]
        if jnp.any(λ < 0):
            warnings.warn("Some eigenvalues are negative.")
        self.V = U[:, : self.N] / jnp.sqrt(self.λ[jnp.newaxis, :])

        self._fit_inverse_transform(K)

        return self

    def transform(self, X):
        """Transforming X and computing principal components for each sample.

        Args:
            X: data to transform of shape `(N_dim, N_samples)`.

        Returns:
            X_t: transformed data of shape `(N, N_samples)`.
        """
        X = jnp.array(X, dtype=jnp.float32)
        K = self.kernel(X, self.data)
        K = self._kernel_normalization(K)
        X_t = jnp.einsum("ij,jk->ki", K, self.V)
        return np.array(X_t, dtype=np.float32)

    def inverse_transform(self, X_t):
        """Transforming X_t back to the original space.

        Args:
            X_t: data in principal-components space, of shape `(N, N_samples)`.

        Returns:
            X: transformed data in original space, of shape `(N_dim, N_samples)`.
        """
        X_t = jnp.array(X_t, dtype=jnp.float32)

        # self._vectorize_kernel("inverse_kernel")

        if self.inverse_kernel is None:
            X = jnp.einsum("ij,ik->kj", X_t, self.W)
            return np.array(X, dtype=np.float32)
        else:
            K = self.inverse_kernel(X_t, self.φ)
            # print(K.shape, self.W.shape)
            X = jnp.einsum("ij,jk->ki", K, self.W)
            return np.array(X, dtype=np.float32)

    def sample(self, n=1, compute_inverse=False):
        """Sample from the multivariate Gaussian prior
        and (optionally) compute the inverse_transofrm of the pulled samples.

        Args:
            n: number of samples.
            compute_inverse: either to compute the inverse of pulled samples or not.

        Returns:
            x: pulled samples, or sampled data in the original space, depending on the `compute_inverse` flag.
        """
        X_t = (
            np.random.normal(size=(self.N, n))
            * np.sqrt(np.array(self.λ))[:, np.newaxis]
        )

        if compute_inverse:
            return self.inverse_transform(X_t)
        else:
            return X_t

    def save(self, filename, compression_scheme={}, save_data=False):
        """Save the PCA fit as hdf5 file.

        Args:
            filename: name of the file.
            compression_scheme: dictionary containing compression options,
                eg. {"compression": "gzip", "compression_opts": 7, "shuffle": True}
            save_data: bool, either to save the whole data array. If `False`, one
                needs to provide the data while loading.

        Returns:
            `None`
        """
        datasets = ["K_rows", "K_0", "V", "W", "φ", "λ"]
        attrs = ["N_dim", "N_samples", "N"]
        if save_data:
            datasets.append("data")

        with h5py.File(filename, "w") as f:
            for attr in attrs:
                f.attrs[attr] = getattr(self, attr)
            for k in datasets:
                f.create_dataset(
                    k,
                    data=np.array(getattr(self, k), dtype=np.float32),
                    **compression_scheme,
                )

    def load(self, filename, data=None):
        """Load the PCA fit.

        Args:
            filename: name of the file.
            data: data of the model. Needed only if `save_data == False` while saving.

        Returns:
            `None`
        """
        datasets = ["K_rows", "K_0", "V", "W", "φ", "λ"]
        attrs = ["N_dim", "N_samples", "N"]
        if data is None:
            datasets.append("data")
        else:
            self.data = data

        with h5py.File(filename, "r") as f:
            if self.N != f.attrs["N"]:
                raise ValueError(
                    "File contains KPCA of order {}, which is different from {}.".format(
                        f.attrs["N"], self.N
                    )
                )
            for attr in attrs:
                setattr(self, attr, f.attrs[attr])
            for k in datasets:
                setattr(self, k, jnp.array(f[k], dtype=jnp.float32))
        if (self.N_dim, self.N_samples) != self.data.shape:
            raise ValueError(
                f"Data shape is {data.shape}, which differs from saved "
                f"N_dim, N_samples: {(self.N_dim, self.N_samples)}"
            )


class KernelPCA_m(KernelPCA):
    """Kernel PCA in jax, for CPU + GPU environments.

    Designed to use CPU as a host. To configure properly, put the following
    at the beginnning of your script:
    ```
    import jax
    jax.config.update('jax_platform_name', 'cpu')
    ```
    and pass `jax.devices("gpu")` as devices.

    Attributes:
        N: number of principal components. If not given, keeps N_samples components.
        kernel: a kernel function, or string ["rbf", ]
        kernel_kwargs: arguments needed for a kernel specification
        inverse_kernel: a kernel function or string ["rbf", ]
            Which kernel to use for the inverse transform, None does the simplest transformation, "same" takes the same function as `kernel`.
        inverse_kernel_kwargs: possible arguments for inverse_kernel.

    Methods:
        fit: computing principal vectors.
        transform: calculating principal components for a given input.
        inverse_transform: inverse of the transform.
    """

    def __init__(
        self,
        devices,
        N=None,
        α=1.0,
        kernel="rbf",
        kernel_kwargs=None,
        inverse_kernel=None,
        inverse_kernel_kwargs=None,
    ):
        super().__init__(
            N, α, kernel, kernel_kwargs, inverse_kernel, inverse_kernel_kwargs
        )
        self.devices = devices

    def _collect_kernel(self, name, X, Y, batch_size_X, batch_size_Y):
        if self.vectorized[name] == False:
            self._vectorize_kernel(name)
        #         X = jnp.array(X, dtype = jnp.float32)
        #         Y = jnp.array(Y, dtype = jnp.float32)
        #         print(X.shape, Y.shape, batch_size_X, batch_size_Y)
        X = X.reshape(X.shape[0], X.shape[1] // batch_size_X, batch_size_X)
        X = np.moveaxis(X, 0, -2)
        #         print(X.shape)
        Y = Y.reshape(
            Y.shape[0],
            Y.shape[1] // batch_size_Y // len(self.devices),
            len(self.devices),
            batch_size_Y,
        )
        Y = np.moveaxis(Y, 0, -2)
        #         print(Y.shape)

        K = []
        for x in X:
            row_K = []
            for y in Y:
                p_K = jnp.concatenate(
                    getattr(self, name)(x, y), axis=1
                )  # concatenating pmapped axis
                row_K.append(p_K)
            row_K = jnp.concatenate(row_K, axis=1)
            K.append(row_K)
        K = jnp.concatenate(K, axis=0)
        return K

    def _vectorize_kernel(self, name):
        if self.kernel is not None and self.vectorized[name] == False:
            setattr(
                self,
                name,
                jax.pmap(
                    getattr(self, name),
                    in_axes=(None, 0),
                    devices=self.devices,
                    backend="gpu",
                ),
            )
            self.vectorized[name] = True

    def _fit_inverse_transform(self, K):
        # φ = self.transform(self.data)
        # shortened expression
        self.φ = (self.V * self.λ).T

        if self.N_dim < len(self.devices):
            devices = self.devices[: self.N_dim]
        else:
            devices = self.devices

        # _partial_solve = jax.pmap(
        #     jnp.linalg.solve, in_axes=(None, 0), devices=devices, backend="gpu"
        # )

        @partial(jax.pmap, in_axes=(None, 0), devices=devices, backend="gpu")
        @jax.jit
        def _partial_einsum(x, y):
            return jnp.einsum("ij,jk->ik", x, y)

        B = self.data.reshape(
            self.N_dim // self.batch_size_dim // len(devices),
            len(devices),
            self.batch_size_dim,
            self.N_samples,
        )
        B = jnp.moveaxis(B, -1, -2)

        if self.inverse_kernel is None:
            regularized_K = K + self.α * jnp.eye(K.shape[0])
            regularized_K = jax.device_put(regularized_K, self.devices[0])
            # regularized_K_inv = jax.scipy.linalg.pinvh(regularized_K)
            S, U = jnp.linalg.eigh(regularized_K)
            regularized_K_inv = jnp.einsum("ik,k,jk->ij", U, 1 / S, U)
            # X = jnp.concatenate([jnp.concatenate(_partial_solve(regularized_K, b), axis = 1) for b in B], axis = 1)
            X = jnp.concatenate(
                [
                    jnp.concatenate(_partial_einsum(regularized_K_inv, b), axis=1)
                    for b in B
                ],
                axis=1,
            )
            self.W = jnp.einsum("ij,jk->ik", self.φ, X)
        else:
            K = self._collect_kernel(
                "inverse_kernel",
                self.φ,
                self.φ,
                self.batch_size_samples,
                self.batch_size_samples,
            )
            regularized_K = K + self.α * jnp.eye(K.shape[0])
            regularized_K = jax.device_put(regularized_K, self.devices[0])
            # regularized_K_inv = jax.scipy.linalg.pinvh(regularized_K)
            S, U = jnp.linalg.eigh(regularized_K)
            regularized_K_inv = jnp.einsum("ik,k,jk->ij", U, 1 / S, U)
            # self.W = jnp.concatenate([jnp.concatenate(_partial_solve(regularized_K, b), axis = 1) for b in B], axis = 1)
            self.W = jnp.concatenate(
                [
                    jnp.concatenate(_partial_einsum(regularized_K_inv, b), axis=1)
                    for b in B
                ],
                axis=1,
            )

    def fit(self, data, batch_size_samples=None, batch_size_dim=None, use_SVD=False):
        """Computing eigenvectors and eigenvalues of the data.

        Args:
            data (np.array): data to fit, of shape `(N_dim, N_samples)`.
            batch_size_samples: size of batches for `N_samples`, defaults to `N_samples // N_devices`.
                `N_samples` should be divisible by `batch_size_samples * N_devices`
            batch_size_dim: size of batches for `N_dim`, defaults to `N_dim // N_devices` or 1 for small `N_dim`.
                `N_dim` should be divisible by `batch_size_dim * N_devices`
            use_SVD (bool): If true, it uses SVD decomposition, which might be
                more stable numerically.

        Returns:
            An instance of itself.
        """
        self.data = np.array(data, dtype=np.float32)
        self.N_dim, self.N_samples = data.shape
        self.N = self.N or self.N_samples
        self.batch_size_samples = batch_size_samples or self.N_samples // len(
            self.devices
        )
        if batch_size_dim is None:
            if self.N_dim <= len(self.devices):
                self.batch_size_dim = 1
            else:
                self.batch_size_dim = self.N_dim // len(self.devices)
        else:
            self.batch_size_dim = batch_size_dim

        K = self._collect_kernel(
            "kernel",
            self.data,
            self.data,
            self.batch_size_samples,
            self.batch_size_samples,
        )
        K = jax.device_put(K, self.devices[0])
        K = self._init_kernel_normalization(K)

        if use_SVD:
            U, λ, _ = jnp.linalg.svd(K, full_matrices=False, hermitian=True)
        else:
            λ, U = jnp.linalg.eigh(K)
            λ = λ[::-1]
            U = U[:, ::-1]

        self.λ = λ[: self.N]
        if jnp.any(λ < 0):
            warnings.warn("Some eigenvalues are negative.")
        self.V = U[:, : self.N] / jnp.sqrt(self.λ[jnp.newaxis, :])

        self._fit_inverse_transform(K)

        return self

    def transform(self, X, batch_size_samples=None):
        """Transforming X and computing principal components for each sample.

        Args:
            X: data to transform of shape `(N_dim, N_samples)`.
            batch_size_samples: size of batches for `N_samples` of `X`, defaults to `N_samples`.
                `N_samples` should be divisible by `batch_size_samples`

        Returns:
            X_t: transformed data of shape `(N, N_samples)`.
        """
        batch_size_samples = batch_size_samples or X.shape[1]

        X = np.array(X, dtype=np.float32)
        K = self._collect_kernel(
            "kernel", X, self.data, batch_size_samples, self.batch_size_samples
        )
        K = jax.device_put(K, self.devices[0])
        K = self._kernel_normalization(K)
        X_t = jnp.einsum("ij,jk->ki", K, self.V)
        return np.array(X_t, dtype=np.float32)

    def inverse_transform(self, X_t, batch_size_samples=None):
        """Transforming X_t back to the original space.

        Args:
            X_t: data in principal-components space, of shape `(N, N_samples)`.
            batch_size_samples: size of batches for `N_samples` of `X_t`, defaults to `N_samples`.
                `N_samples` should be divisible by `batch_size_samples`

        Returns:
            X: transformed data in original space, of shape `(N_dim, N_samples)`.
        """
        batch_size_samples = batch_size_samples or X_t.shape[1]

        X_t = np.array(X_t, dtype=np.float32)

        if self.inverse_kernel is None:
            X = jnp.einsum("ij,ik->kj", X_t, self.W)
            return np.array(X, dtype=np.float32)
        else:
            K = self._collect_kernel(
                "inverse_kernel",
                X_t,
                self.φ,
                batch_size_samples,
                self.batch_size_samples,
            )
            if self.N_dim < len(self.devices):
                devices = self.devices[: self.N_dim]
            else:
                devices = self.devices

            @partial(jax.pmap, in_axes=(None, 0), devices=devices, backend="gpu")
            @jax.jit
            def _partial_einsum(x, y):
                return jnp.einsum("ij,jk->ki", x, y)

            # X = jnp.einsum("ij,jk->ki", K, self.W)
            W = self.W.reshape(
                self.N_samples,
                self.N_dim // len(devices) // self.batch_size_dim,
                len(devices),
                self.batch_size_dim,
            )
            W = np.moveaxis(W, 0, -2)
            X = jnp.concatenate(
                [jnp.concatenate(_partial_einsum(K, w), axis=0) for w in W], axis=0
            )
            return np.array(X, dtype=np.float32)

    def sample(self, n=1, compute_inverse=False, batch_size_samples=None):
        """Sample from the multivariate Gaussian prior
        and (optionally) compute the inverse_transofrm of the pulled samples.

        Args:
            n: number of samples.
            compute_inverse: either to compute the inverse of pulled samples or not.
            batch_size_samples: `batch_size_samples` for `inverse_transform`, ignored if `compute_inverse` is `False`.

        Returns:
            x: pulled samples, or sampled data in the original space, depending on the `compute_inverse` flag.
        """
        X_t = (
            np.random.normal(size=(self.N, n))
            * np.sqrt(np.array(self.λ))[:, np.newaxis]
        )

        if compute_inverse:
            return self.inverse_transform(X_t, batch_size_samples)
        else:
            return X_t
