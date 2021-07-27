from functools import partial
import jax
import jax.numpy as jnp

# def _dimension_check(x, y):
#     if x.shape != y.shape:
#         raise ValueError("x and y shapes are not the same")

@jax.jit
def _linear(x, y):
    return jnp.einsum("ij,ik->jk", x, y)

@jax.jit
def _rbf(x, y, gamma = 1.):
    # if jnp.ndim(x) == 1:
    #     x = x.reshape(-1, 1)
    #     y = y.reshape(-1, 1)
    xx = jnp.einsum("ik,ik->k", x, x).reshape(-1, 1)
    yy = jnp.einsum("ik,ik->k", y, y).reshape(1, -1)
    xy = _linear(x, y)
    d2 = xx + yy - 2 * xy
    return jnp.exp(-gamma * d2)

@jax.jit
def _poly(x, y, gamma = 1., r = 1., d = 2.):
    return jnp.power(gamma * _linear(x, y) + r, d)

@jax.jit
def _tanh(x, y, gamma = 1., r = 1.): 
    return jnp.tanh(gamma * _linear(x, y) + r) # often called `sigmoid`

@jax.jit
def _cosine(x, y):
    '''
    Computing cosine similarity between x & y.
    '''
    xx = jnp.einsum("ik,ik->k", x, x).reshape(-1, 1)
    yy = jnp.einsum("ik,ik->k", y, y).reshape(1, -1)
    xy = _linear(x, y)
    return xy / (xx @ yy)