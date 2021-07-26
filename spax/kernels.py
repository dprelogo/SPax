from functools import partial
import jax
import jax.numpy as jnp

# def _dimension_check(x, y):
#     if x.shape != y.shape:
#         raise ValueError("x and y shapes are not the same")

@jax.jit
def _rbf(x, y, gamma = 1.):
    d = x - y
    return jnp.exp(-gamma * d.T @ d)

@jax.jit
def _linear(x, y):
    return x.T @ y

@jax.jit
def _poly(x, y, gamma = 1., r = 1., d = 2.):
    return jnp.power(gamma * x.T @ y + 1, d)

@jax.jit
def _tanh(x, y, gamma = 1., r = 1.): 
    return jnp.tanh(gamma * x.T @ y + r) # often called `sigmoid`

@jax.jit
def _cosine(x, y):
    '''
    Computing cosine similarity between x & y.
    '''
    return x.T @ y / jnp.sqrt(x.T @ x * y.T @ y)