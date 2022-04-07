# SPax: Signal comPression in [JAX](https://github.com/google/jax)
Supporting both CPU and (multi) GPU operations.


## Algorithms
- **Principal Component Analysis**
  - using eigenvalue decomposition on a covariance/data matrix, reducing the memory footprint
- **Kernel Principal Component Analysis**
  - flexibility in the inverse transform definition
- **MOPED compression**
  - (normalized) compression for the Gaussian likelihood, with fixed covariance
  - generalized MOPED for any likelihood function
- **Fisher information calculation**
  - with additional PCA compression for highly dimensional datasets

## Installation
For installation on CPU-only machine, clone the repository and install as:
```
pip install ".[cpu]"
```
For the GPU support:
```
# for CUDA>=11.1 and cuDNN>=8.2
pip install ".[gpu]" -f https://storage.googleapis.com/jax-releases/jax_releases.html 
```
See [JAX installation](https://github.com/google/jax#installation) instructions for other CUDA/cuDNN versions and update JAX accordingly.
## Additional flags
In the case CUDA is not recognized, run as
```
XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda python your_script.py
```

In the case of GPU memory problems, `XLA_PYTHON_CLIENT_MEM_FRACTION=.XX` or `XLA_PYTHON_CLIENT_PREALLOCATE=false` might get handy.
