import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
import numpy as np
import spax

pca = spax.PCA(5, devices = jax.devices("gpu"))
data = np.random.normal(0, 1, size = (10, 100000)) * np.sqrt(np.arange(1, 11))[:, np.newaxis]

pca.fit(data, batch_size = 2) # N_dim % batch_size == 0
sampled_data = pca.sample(100000)
print(np.std(pca.transform(sampled_data), axis = 1, ddof = 1)**2) # should be [6, 7, 8, 9, 10]
print(pca.λ) # should be the same
print(np.round(pca.v.T, 1)) # should be a +-unit matrix on last 5 dimensions