import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import h5py
from . import kernels

class PCA():
    '''PCA in jax.

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
    '''
    def __init__(self, N = None):
        self.N = N
        
    def fit(self, data, whiten = False):
        '''Computing eigenvectors and eigenvalues of the data.

        Args:
            data (np.array): data to fit, of shape `(N_dim, N_samples)`.
            whiten (bool): scaling all dimensions to the unit variance.

        Returns:
            An instance of itself.
        '''
        data = jnp.array(data, dtype = jnp.float32)
        N_dim, N_samples = data.shape
        if self.N is None:
            self.N = min(N_dim, N_samples)

        self.μ = jnp.mean(data, axis = 1, keepdims = True, dtype = jnp.float64).astype(jnp.float32)
        if whiten:
            self.σ = jnp.std(data, axis = 1, keepdims = True, dtype = jnp.float64).astype(jnp.float32)
        else:
            self.σ = jnp.ones((N_dim, 1), dtype = jnp.float32)

        data = (data - self.μ) / self.σ

        if N_dim < N_samples:
            C = (jnp.einsum("ik,jk->ij", data, data, precision = jax.lax.Precision.HIGH) / (N_samples - 1)).astype(jnp.float32)
            self.λ, self.U = jnp.linalg.eigh(C)
            self.λ = jnp.sqrt(self.λ[-self.N:])
            self.U = self.U[:, -self.N:]

            self.U = self.U[:, ::-1]
            self.λ = self.λ[::-1]
        else:
            D = (jnp.einsum("ki,kj->ij", data, data, precision = jax.lax.Precision.HIGH) / N_dim).astype(jnp.float32)
            λ, V = jnp.linalg.eigh(D)
            self.λ = jnp.sqrt(λ[-self.N:]) * jnp.sqrt(N_dim / (N_samples - 1))
            S_inv = (1 / (jnp.sqrt(λ[-self.N:]) * jnp.sqrt(N_dim)))[jnp.newaxis, :]
            VS_inv = V[:, -self.N:] * S_inv
            self.U = jnp.einsum("ij,jk->ik", data, VS_inv, precision = jax.lax.Precision.HIGH).astype(jnp.float32)

            self.U = self.U[:, ::-1]
            self.λ = self.λ[::-1]

        return self

    def transform(self, X):
        '''Transforming X and computing principal components for each sample.

        Args:
            X: data to transform of shape `(N_dim, N_samples)`.
        
        Returns:
            X_t: transformed data of shape `(N, N_samples)`.
        '''
        X = jnp.array(X, dtype = jnp.float32)
        X_t = jnp.einsum("ji,jk->ik", self.U, (X - self.μ) / self.σ)
        return np.array(X_t, dtype = np.float32)

    def inverse_transform(self, X_t):
        '''Transforming X_t back to the original space.

        Args:
            X_t: data in principal-components space, of shape `(N, N_samples)`.
        
        Returns:
            X: transformed data in original space, of shape `(N_dim, N_samples)`.
        '''
        X_t = jnp.array(X_t, dtype = jnp.float32)
        X = jnp.einsum("ij,jk->ik", self.U, X_t) * self.σ + self.μ
        return np.array(X, dtype = np.float32)

    def sample(self, n = 1):
        '''Sample from the multivariate Gaussian prior 
        and compute the inverse_transofrm of the pulled samples.

        Args:
            n: number of samples.

        Returns:
            X: sampled data in original space, of shape `(N_dim, n)`.
        '''
        X_t = np.random.normal(size = (self.N, n)) * np.array(self.λ)[:, np.newaxis]
        return self.inverse_transform(X_t)

    def save(self, filename, compression_scheme = {}):
        '''Save the PCA fit as hdf5 file.
        
        Args:
            filename: name of the file.
            compression_scheme: dictionary containing compression options, 
                eg. {"compression": "gzip", "compression_opts": 7, "shuffle": True}
        
        Returns:
            `None`
        '''
        with h5py.File(filename, "w") as f:
            f.attrs["N"] = self.N
            for k in ["λ", "σ", "μ", "U"]:
                f.create_dataset(k, data = np.array(getattr(self, k), dtype = np.float32), **compression_scheme)
        
    def load(self, filename):
        '''Load the PCA fit. 
        
        Args:
            filename: name of the file.
        
        Returns:
            `None`
        '''
        with h5py.File(filename, "r") as f:
            if self.N != f.attrs["N"]:
                raise ValueError(
                    "File contains PCA of order {}, which is different from {}.".format(f.attrs["N"], self.N)
                    )
            for k in ["λ", "σ", "μ", "U"]:
                setattr(self, k, jnp.array(f[k], dtype = jnp.float32))


class PCA_m(PCA):
    '''PCA in jax, for CPU + GPU environments.

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
    '''
    def __init__(self, devices, N = None):
        super().__init__(N)
        self.devices = devices
        
    def fit(self, data, batch_size = None, whiten = False, centering_data = "CPU"):
        '''Computing eigenvectors and eigenvalues of the data.

        Args:
            data: data to fit on of shape `(N_dim, N_samples)`.
            batch_size: splitting calculation in data chunks of `(N_dim / n_devices / batch_size, N_samples)`. 
                Take care such matrix (+ data) can fit on one device.
                `N_dim % (N_devices * batch_size) == 0`, defaults to `N_dim / n_devices`.
            whiten (bool): scaling all dimensions to the unit variance.
            centering_data (str): either "CPU" or "GPU", where to perform data centering/whitening.

        Returns:
            An instance of itself.
        '''
        n_d = len(self.devices)
        N_dim, N_samples = data.shape
        if self.N is None:
            self.N = min(N_dim, N_samples)
        batch_size = N_dim // n_d if batch_size is None else batch_size
        if N_dim % (n_d * batch_size) != 0:
            raise ValueError("N_dim of the data should be divisible by the n_devices * batch_size.")

        if centering_data == "CPU":
            data = data.astype(np.float32)
            self.μ = jnp.mean(data, axis = 1, keepdims = True, dtype = jnp.float64).astype(jnp.float32)
            if whiten:
                self.σ = jnp.std(data, axis = 1, keepdims = True, dtype = jnp.float64).astype(jnp.float32)
            else:
                self.σ = jnp.ones(shape = self.μ.shape, dtype = np.float32)
            data = (data - self.μ) / self.σ
            data = data.reshape(N_dim // (n_d * batch_size), n_d, batch_size, N_samples)
        elif centering_data == "GPU":
            data = data.astype(np.float32).reshape(N_dim // (n_d * batch_size), n_d, batch_size, N_samples)
            @partial(jax.pmap, devices = self.devices, backend = "gpu")
            @jax.jit
            def data_transform(d_part):
                μ_part = jnp.mean(d_part, axis = 1, keepdims = True, dtype = jnp.float64).astype(jnp.float32)
                if whiten:
                    σ_part = jnp.std(d_part, axis = 1, keepdims = True, dtype = jnp.float64).astype(jnp.float32)
                    d_part = (d_part - μ_part) / σ_part
                else:
                    σ_part = jnp.ones(shape = μ_part.shape, dtype = jnp.float32)
                    d_part = d_part - μ_part
                return d_part, μ_part, σ_part

            data_transformed, μ, σ = [], [], []
            for d in data:
                d_part, μ_part, σ_part = data_transform(d)
                data_transformed.append(jnp.array(d_part, dtype = jnp.float32))
                μ.append(jnp.array(μ_part, dtype = jnp.float32))
                σ.append(jnp.array(σ_part, dtype = jnp.float32))
            self.μ = jnp.array(μ, dtype = jnp.float32).flatten()[:, jnp.newaxis]
            self.σ = jnp.array(σ, dtype = jnp.float32).flatten()[:, jnp.newaxis]
            data = jnp.array(data_transformed, dtype = jnp.float32)
        else:
            raise ValueError(f"centering_data is {centering_data}, should be either CPU or GPU.")

        if N_dim < N_samples:
            @partial(jax.pmap, in_axes = (0, None), devices = self.devices, backend = "gpu")
            @jax.jit
            def partial_C(d1, d2):
                return (jnp.einsum("ik,jk->ij", d1, d2, precision = jax.lax.Precision.HIGH) / (N_samples - 1)).astype(jnp.float32)
            C = []
            for d_y in data.reshape(N_dim // batch_size, batch_size, N_samples):
                row_C = []
                for d_x in data:
                    p_C = jnp.concatenate(partial_C(d_x, d_y), axis = 1)
                    row_C.append(p_C)
                row_C = jnp.concatenate(row_C, axis = 1)
                C.append(row_C)
            C = jnp.concatenate(C, axis = 0)
            C = jax.device_put(C, self.devices[0])
            self.λ, self.U = jnp.linalg.eigh(C)
            self.λ = jnp.sqrt(self.λ[-self.N:])
            self.U = self.U[:, -self.N:]

            self.U = self.U[:, ::-1]
            self.λ = self.λ[::-1]
        else:
            @partial(jax.pmap, in_axes = (0, 0), devices = self.devices, backend = "gpu")
            @jax.jit
            def partial_D(d1, d2):
                return (jnp.einsum("ki,kj->ij", d1, d2, precision = jax.lax.Precision.HIGH) / N_dim).astype(jnp.float32)
            D = jnp.sum(jnp.array([jnp.sum(partial_D(d1, d2), axis = 0) for d1, d2, in zip(data, data)]), axis = 0)
            D = jax.device_put(D, self.devices[0])
            λ, V = jnp.linalg.eigh(D)
            self.λ = jnp.sqrt(λ[-self.N:]) * jnp.sqrt(N_dim / (N_samples - 1))
            S_inv = (1 / (jnp.sqrt(λ[-self.N:]) * jnp.sqrt(N_dim)))[jnp.newaxis, :]
            VS_inv = V[:, -self.N:] * S_inv

            @partial(jax.pmap, devices = self.devices, backend = "gpu")
            @jax.jit
            def partial_U(d):
                return jnp.einsum("ij,jk->ik", d, VS_inv, precision = jax.lax.Precision.HIGH).astype(jnp.float32)

            self.U = jnp.concatenate([jnp.concatenate(partial_U(d)) for d in data], axis = 0)

            self.U = self.U[:, ::-1]
            self.λ = self.λ[::-1]
        
        return self

    def transform(self, X, batch_size = None):
        '''Transforming X and computing principal components for each sample.

        Args:
            X: data to transform of shape `(N_dim, N_samples)`.
            batch_size: splitting calculation in data chunks of `(N_dim / n_devices / batch_size, N_samples)`. 

        Returns:
            X_t: transformed data of shape `(N, N_samples)`.
        '''
        n_d = len(self.devices)
        N_dim, N_samples = X.shape
        batch_size = N_dim // n_d if batch_size is None else batch_size
        X = X.astype(np.float32).reshape(N_dim // (n_d * batch_size), n_d, batch_size, N_samples)
        μ = self.μ.reshape(N_dim // (n_d * batch_size), n_d, batch_size, 1)
        σ = self.σ.reshape(N_dim // (n_d * batch_size), n_d, batch_size, 1)
        U = self.U.reshape(N_dim // (n_d * batch_size), n_d, batch_size, self.N)

        @partial(jax.pmap, in_axes = (0, 0, 0, 0), devices = self.devices, backend = "gpu")
        @jax.jit
        def partial_transform(U, x, μ, σ):
            return jnp.einsum("ji,jk->ik", U, (x - μ) / σ, precision = jax.lax.Precision.HIGH).astype(jnp.float32)

        X_t = jnp.sum(
            jnp.array([jnp.sum(partial_transform(_u, _x, _μ, _σ), axis = 0) for _u, _x, _μ, _σ in zip(U, X, μ, σ)]), axis = 0)
        return np.array(X_t, dtype = np.float32)

    def inverse_transform(self, X_t, batch_size = None):
        '''Transforming X_t back to the original space.

        Args:
            X_t: data in principal-components space, of shape `(N, N_samples)`.
            batch_size: splitting calculation in data chunks of `(N_dim / n_devices / batch_size, N_samples)`. 

        Returns:
            X: transformed data in original space, of shape `(N_dim, N_samples)`.
        '''
        n_d = len(self.devices)
        N_dim = self.U.shape[0]
        batch_size = N_dim // n_d if batch_size is None else batch_size
        X_t = jnp.array(X_t, dtype = jnp.float32)
        μ = self.μ.reshape(N_dim // (n_d * batch_size), n_d, batch_size, 1)
        σ = self.σ.reshape(N_dim // (n_d * batch_size), n_d, batch_size, 1)
        U = self.U.reshape(N_dim // (n_d * batch_size), n_d, batch_size, self.N)

        @partial(jax.pmap, in_axes = (0, 0, 0), devices = self.devices, backend = "gpu")
        @jax.jit
        def partial_inv_transform(U, μ, σ):
            return jnp.einsum("ij,jk->ik", U, X_t, precision = jax.lax.Precision.HIGH).astype(jnp.float32) * σ + μ

        X = jnp.concatenate(
            jnp.array([jnp.concatenate(partial_inv_transform(_u, _μ, _σ), axis = 0) for _u, _μ, _σ in zip(U, μ, σ)]), axis = 0)
        return np.array(X, dtype = np.float32)

    def sample(self, n = 1, batch_size = None):
        '''Sample from the multivariate Gaussian prior 
        and compute the inverse_transofrm of the pulled samples.

        Args:
            n: number of samples.
            batch_size: splitting calculation in data chunks of `(N_dim / n_devices / batch_size, N_samples)`.
                Used only for the inverse_transform calculation. 

        Returns:
            X: sampled data in original space, of shape `(N_dim, n)`.
        '''
        X_t = np.random.normal(size = (self.N, n)) * np.array(self.λ)[:, np.newaxis]
        return self.inverse_transform(X_t, batch_size)
    

class KernelPCA():
    '''Kernel PCA in jax.

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
    '''
    def __init__(self, N = None, α = 1.0, kernel = "rbf", kernel_kwargs = None, inverse_kernel = None, inverse_kernel_kwargs = None):
        self.N = N
        self.α = α
        
        self.vectorized = {}
        if inverse_kernel == "same":
            inverse_kernel = kernel
            inverse_kernel_kwargs = kernel_kwargs

        self._init_kernel(kernel, kernel_kwargs, "kernel")
        self._init_kernel(inverse_kernel, inverse_kernel_kwargs, "inverse_kernel")
    
    def _init_kernel(self, kernel, kernel_kwargs, name = "kernel"):
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
        if self.kernel is not None and self.vectorized[name] == False:
            setattr(self, name, jax.vmap(jax.vmap(getattr(self, name), in_axes = (None, 1)), in_axes = (1, None)))
        
    def _kernel_normalization(self, K):
        K_columns = jnp.sum(K, axis = 1)[:, jnp.newaxis] / self.N_samples
        return K - self.K_rows - K_columns - self.K_0
    
    def _init_kernel_normalization(self, K):
        self.K_rows = jnp.sum(K, axis = 0)[jnp.newaxis, :] / self.N_samples
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
            self.W = jax.scipy.linalg.solve(regularized_K, self.data.T, sym_pos = True, overwrite_a = True)
            
        
    def fit(self, data):
        '''Computing eigenvectors and eigenvalues of the data.

        Args:
            data (np.array): data to fit, of shape `(N_dim, N_samples)`.

        Returns:
            An instance of itself.
        '''
        self.data = jnp.array(data, dtype = jnp.float32)
        self.N_dim, self.N_samples = data.shape
        self.N = self.N or self.N_samples
        
        self._vectorize_kernel("kernel")

        K = self.kernel(self.data, self.data)
#         print(K.shape)
        K = self._init_kernel_normalization(K)
        λ, U = jnp.linalg.eigh(K)
        λ = λ[-self.N:]
        V = U[:, -self.N:] / jnp.sqrt(λ[jnp.newaxis, :])
                
        self.V = V[:, ::-1]
        self.λ = λ[::-1]
        
        self._fit_inverse_transform(K)

        return self

    def transform(self, X):
        '''Transforming X and computing principal components for each sample.

        Args:
            X: data to transform of shape `(N_dim, N_samples)`.
        
        Returns:
            X_t: transformed data of shape `(N, N_samples)`.
        '''
        X = jnp.array(X, dtype = jnp.float32)
        K = self.kernel(X, self.data)
        K = self._kernel_normalization(K)
        X_t = jnp.einsum("ij,jk->ki", K, self.V)
        return np.array(X_t, dtype = np.float32)
    
    def inverse_transform(self, X_t):
        '''Transforming X_t back to the original space.

        Args:
            X_t: data in principal-components space, of shape `(N, N_samples)`.
        
        Returns:
            X: transformed data in original space, of shape `(N_dim, N_samples)`.
        '''        
        X_t = jnp.array(X_t, dtype = jnp.float32)
        
        self._vectorize_kernel("inverse_kernel")
        
        if self.inverse_kernel is None:
            X = jnp.einsum("ij,ik->kj", X_t, self.W)
            return np.array(X, dtype = np.float32)
        else:
            K = self.inverse_kernel(X_t, self.φ)
            # print(K.shape, self.W.shape)
            X = jnp.einsum("ij,jk->ki", K, self.W)
            return np.array(X, dtype = np.float32)


class KernelPCA_m(KernelPCA):
    '''Kernel PCA in jax, for CPU + GPU environments.

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
    '''
    def __init__(self, devices, N = None, α = 1.0, kernel = "rbf", kernel_kwargs = None, inverse_kernel = None, inverse_kernel_kwargs = None):
        super().__init__(N, α, kernel, kernel_kwargs, inverse_kernel, inverse_kernel_kwargs)
        self.devices = devices
    
    def _vectorize_kernel(self, name):
        if self.kernel is not None and self.vectorized[name] == False:
            setattr(self, name, jax.pmap(getattr(self, name), in_axes = (None, 0), devices = self.devices, backend = "gpu"))
    
    def _collect_kernel(self, name, X, Y):
        #TODO: Finish this
        K = [getattr(self, name)(x, Y) for x in X]
        K = jnp.array(K, dtype = jnp.float32).reshape(self.N_samples, self.N_samples)
    
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
            self.W = jax.scipy.linalg.solve(regularized_K, self.data.T, sym_pos = True, overwrite_a = True)
            
        
    def fit(self, data):
        '''Computing eigenvectors and eigenvalues of the data.

        Args:
            data (np.array): data to fit, of shape `(N_dim, N_samples)`.

        Returns:
            An instance of itself.
        '''
        self.data = jnp.array(data, dtype = jnp.float32)
        self.N_dim, self.N_samples = data.shape
        self.N = self.N or self.N_samples
        
        self._vectorize_kernel("kernel")

        K = self.kernel(self.data, self.data)
#         print(K.shape)
        K = self._init_kernel_normalization(K)
        λ, U = jnp.linalg.eigh(K)
        λ = λ[-self.N:]
        V = U[:, -self.N:] / jnp.sqrt(λ[jnp.newaxis, :])
        
#         V = U / jnp.sqrt(λ[jnp.newaxis, :])
        
        self.V = V[:, ::-1]
        self.λ = λ[::-1]
        
        self._fit_inverse_transform(K)

        return self

    def transform(self, X):
        '''Transforming X and computing principal components for each sample.

        Args:
            X: data to transform of shape `(N_dim, N_samples)`.
        
        Returns:
            X_t: transformed data of shape `(N, N_samples)`.
        '''
        X = jnp.array(X, dtype = jnp.float32)
        K = self.kernel(X, self.data)
        K = self._kernel_normalization(K)
        X_t = jnp.einsum("ij,jk->ki", K, self.V)
        return np.array(X_t, dtype = np.float32)
    
    def inverse_transform(self, X_t):
        '''Transforming X_t back to the original space.

        Args:
            X_t: data in principal-components space, of shape `(N, N_samples)`.
        
        Returns:
            X: transformed data in original space, of shape `(N_dim, N_samples)`.
        '''        
        X_t = jnp.array(X_t, dtype = jnp.float32)
        
        self._vectorize_kernel("inverse_kernel")
        
        if self.inverse_kernel is None:
            X = jnp.einsum("ij,ik->kj", X_t, self.W)
            return np.array(X, dtype = np.float32)
        else:
            K = self.inverse_kernel(X_t, self.φ)
            # print(K.shape, self.W.shape)
            X = jnp.einsum("ij,jk->ki", K, self.W)
            return np.array(X, dtype = np.float32)
