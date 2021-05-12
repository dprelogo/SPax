import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import h5py

class PCA():
    '''PCA in jax.

    Assuming data and intermediate results fit on a local device (RAM or GPU memory).
    No additional setup needed.

    Attributes:
        N: number of principal components.
    
    Methods:
        fit: computing principal vectors.
        transform: calculating principal components for given input.
        inverse_transform: inverse of the transform.
        sample: sampling multivariate gaussian distribution of the principal components
            and computing inverse_transform.
    '''
    def __init__(self, N):
        self.N = N
        
    def fit(self, data, whiten = False):
        '''Computing eigenvectors and eigenvalues of the data.

        Args:
            data (np.array): data to fit, of shape `(N_dim, N_samples)`.
            whiten (bool): scaling all dimensions to the unit variance.

        Returns:
            `None`
        '''
        data = jnp.array(data, dtype = jnp.float32)
        N_dim, N_samples = data.shape
        self.μ = jnp.mean(data, axis = 1, keepdims = True, dtype = jnp.float64).astype(jnp.float32)
        if whiten:
            self.σ = jnp.std(data, axis = 1, keepdims = True, dtype = jnp.float64).astype(jnp.float32)
        else:
            self.σ = jnp.ones((N_dim, 1), dtype = jnp.float32)

        data = (data - self.μ) / self.σ

        if N_dim <= N_samples:
            C = (jnp.einsum("ik,jk->ij", data, data, precision = jax.lax.Precision.HIGH) / (N_dim - 1)).astype(jnp.float32)
            self.λ, self.U = jnp.linalg.eigh(C)
            self.λ = jnp.sqrt(self.λ[-self.N:])
            self.U = self.U[:, -self.N:]
        else:
            D = (jnp.einsum("ki,kj->ij", data, data, precision = jax.lax.Precision.HIGH) / N_samples).astype(jnp.float32)
            λ, V = jnp.linalg.eigh(D)
            self.λ = jnp.sqrt(λ[-self.N:]) * jnp.sqrt(N_samples / (N_dim - 1))
            S_inv = (1 / (jnp.sqrt(λ[-self.N:]) * jnp.sqrt(N_samples)))[jnp.newaxis, :]
            VS_inv = V[:, -self.N:] * S_inv
            self.U = jnp.einsum("ij,jk->ik", data, VS_inv, precision = jax.lax.Precision.HIGH).astype(jnp.float32)

    def transform(self, X):
        '''Transforming X and computing principal components for each sample.

        Args:
            X: data to transform of shape `(N_dim, N_samples)`.
        
        Returns:
            X_t: transformed data of shape `(N, N_samples)`.
        '''
        #self.v.T == R, self.v == R^{-1}, where R is a rotation matrix.
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
            f.create_dataset("λ", data = np.array(self.λ, dtype = np.float32), **compression_scheme)
            f.create_dataset("σ", data = np.array(self.σ, dtype = np.float32), **compression_scheme)
            f.create_dataset("μ", data = np.array(self.μ, dtype = np.float32), **compression_scheme)
            f.create_dataset("U", data = np.array(self.U, dtype = np.float32), **compression_scheme)
        
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
            self.λ = jnp.array(f["λ"], dtype = jnp.float32)
            self.σ = jnp.array(f["σ"], dtype = jnp.float32)
            self.μ = jnp.array(f["μ"], dtype = jnp.float32)
            self.U = jnp.array(f["U"], dtype = jnp.float32)


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
        N: number of principal components.
        devices: list of `jax.devices`.
    
    Methods:
        fit: computing principal vectors.
        transform: calculating principal components for given input.
        inverse_transform: inverse of the transform.
        sample: sampling multivariate gaussian distribution of the principal components
            and computing inverse_transform.
    '''
    def __init__(self, N, devices):
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
            centering_data (str): either "CPU" or "GPU", where to center the data.

        Returns:
            `None`
        '''
        n_d = len(self.devices)
        N_dim, N_samples = data.shape
        batch_size = N_dim // n_d if batch_size is None else batch_size
        if N_dim % (n_d * batch_size) != 0:
            raise ValueError("N_dim of the data should be divisible by the n_devices * batch_size.")

        if centering_data == "CPU":
            data = data.astype(np.float32)
            self.μ = jnp.mean(data, axis = 1, keepdims = True)
            if whiten:
                self.σ = jnp.std(data, axis = 1, keepdims = True)
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
            for i, d in enumerate(data):
                d_part, μ_part, σ_part = data_transform(d)
                data_transformed.append(jnp.array(d_part, dtype = jnp.float32))
                μ.append(jnp.array(μ_part, dtype = jnp.float32))
                σ.append(jnp.array(σ_part, dtype = jnp.float32))
            self.μ = jnp.array(μ, dtype = jnp.float32).flatten()[:, jnp.newaxis]
            self.σ = jnp.array(σ, dtype = jnp.float32).flatten()[:, jnp.newaxis]
            data = jnp.array(data_transformed, dtype = jnp.float32)
        else:
            raise ValueError(f"centering_data is {centering_data}, should be either CPU or GPU.")

        if N_dim <= N_samples:
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