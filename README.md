# SPax
Signal Processing in [Jax](https://github.com/google/jax). 

Supporting both CPU and (multi) GPU operations.

## Additional flags
In the case CUDA is not recognized, run as
```
XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda python example.py
```

In the case of GPU memory problems, `XLA_PYTHON_CLIENT_MEM_FRACTION=.XX` or `XLA_PYTHON_CLIENT_PREALLOCATE=false` might get handy.
