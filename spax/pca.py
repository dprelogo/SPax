import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

class PCA():
    '''
    PCA in jax.

    For CPU only environments, use as is.
    For environment with 1 GPU, if data and computations can fit on a GPU, use as is.
    For general CPU + GPU environment, put the following at the beginning of your code:
    ```
    import jax
    jax.config.update('jax_platform_name', 'cpu')
    ```
    and pass `jax.devices("gpu")` as devices.
    '''
    def __init__(self, N, devices = None):
        self.N = N
        self.devices = devices
        self.n_d = len(devices)
    
    def _fit(self, data):
        self.μ = jnp.mean(data, axis = 1, keepdims = True, dtype = jnp.float64).astype(jnp.float32)
        Σ = jnp.cov(data)
        self.λ, self.v = jnp.linalg.eigh(Σ)
        self.λ = self.λ[-self.N:]
        self.v = self.v[:, -self.N:]

    def _fit_pmap(self, data, batch_size):
        @partial(jax.pmap, in_axes = (0, 0, None, None), devices = self.devices, backend = "gpu")
        @jax.jit
        def cov(y, μ_y, x, μ_x):
            n = x.shape[-1]
            return (x - μ_x) @ (y - μ_y).T / (n - 1)

        N_dim, N_samples = data.shape
        d_y = data.reshape(self.n_d, N_dim // self.n_d, N_samples)
        d_x = data.reshape(N_dim // batch_size, batch_size, N_samples)

        partial_mean = jax.pmap(
            partial(jnp.mean, axis = 1, keepdims = True, dtype = jnp.float64), 
            devices = self.devices, backend = "gpu")
        self.μ = jnp.concatenate(partial_mean(d_y), axis = 0).astype(jnp.float32)
        μ_y = self.μ.reshape(self.n_d, N_dim // self.n_d, 1)
        μ_x = self.μ.reshape(N_dim // batch_size, batch_size, 1)

        Σ = jnp.concatenate([jnp.concatenate(cov(d_y, μ_y, d_xi, μ_xi), axis = 1) for d_xi, μ_xi in zip(d_x, μ_x)], axis = 0)
        self.λ, self.v = jnp.linalg.eigh(Σ)
        self.λ = self.λ[-self.N:]
        self.v = self.v[:, -self.N:]
        
    def fit(self, data, batch_size = None):
        '''
        Computing eigenvectors and eigenvalues of the data.
        `data.shape == (N_dim, N_samples)`
        '''
        data = data.astype(np.float32)
        if self.devices is None:
            self._fit(data)
        else:
            batch_size = data.shape[0] if batch_size is None else batch_size
            self._fit_pmap(data, batch_size)

    def transform(self, X):
        '''
        Transforming X and computing principal components for each sample.
        `X.shape == (N_dim, N_samples)`
        '''
        #self.v.T == R, self.v == R^{-1}

        X = X.astype(np.float32)
        X_t = self.v.T @ (X - self.μ)
        return np.array(X_t, dtype = np.float32)

    def inverse_transform(self, X_t):
        '''
        Transforming X_t back to original space.
        `X_t.shape == (N, N_samples)`
        '''
        X_t = X_t.astype(np.float32)
        X = self.v @ X_t + self.μ
        return np.array(X, dtype = np.float32)

    def sample(self, n = 1):
        '''
        Sample from the Gaussian prior and compute the inverse_transofrm of pulled samples.
        '''
        X_t = np.random.normal(size = (self.N, n)) * np.sqrt(np.array(self.λ)[:, np.newaxis])
        return self.inverse_transform(X_t)
