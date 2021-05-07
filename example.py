import jax
jax.config.update("jax_platform_name", "cpu")
import numpy as np
import spax

pca = spax.PCA(5, devices = jax.devices("gpu"))
data = np.random.normal(0, 1, size = (100, 1000)) * np.arange(1, 101)[:, np.newaxis]

pca.fit(data, batch_size = 10) # N_dims % batch_size == 0
sampled_data = pca.sample(1000)
print(np.std(sampled_data, axis = 1, ddof = 1)) # should be [100, 99, 98, 97, 96]