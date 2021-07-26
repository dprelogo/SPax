from functools import partial
import jax
import jax.numpy as jnp

@jax.jit
def _rbf(x, y, σ = 1.):
    d = x - y
    return jnp.exp(-jnp.dot(d, d) / (2 * σ**2))