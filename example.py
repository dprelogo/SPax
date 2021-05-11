import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
import numpy as np
import spax

pca = spax.PCA_m(5, devices = jax.devices("gpu"))
data = np.random.normal(0, 1, size = (16, 100000)) * np.sqrt(np.arange(1, 17))[:, np.newaxis]

pca.fit(data, batch_size = 2) # N_dim % (N_devices * batch_size) == 0
sampled_data = pca.sample(100000)
print(np.std(pca.transform(sampled_data), axis = 1, ddof = 1)**2) # should be [12, 13, 14, 15, 16]
print(pca.λ ** 2) # should be the same
print(np.round(pca.U.T, 1)) # should be a +-unit matrix on last 5 dimensions

pca = spax.PCA_m(5, devices = jax.devices("gpu"))
data = np.random.normal(0, 1, size = (100000, 16)) * np.sqrt(np.arange(1, 100001))[:, np.newaxis]

pca.fit(data, batch_size = 10000) # N_dim % (N_devices * batch_size) == 0
sampled_data = pca.sample(100)
print(np.std(pca.transform(sampled_data), axis = 1, ddof = 1)**2)
print(pca.λ ** 2)
print(np.round(pca.U.T, 1))