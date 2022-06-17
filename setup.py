from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

_current_jaxlib_version = "0.3.10"
_default_cuda_version = "11"
_default_cudnn_version = "805"

setup(
    name="SPax",
    version="0.1dev",
    author="David Prelogović",
    author_email="david.prelogovic@gmail.com",
    description="Signal processing in Jax.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dprelogo/SPax",
    packages=["spax"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "jax",
        "numpy",
        "h5py",
    ],
    extras_require={
        "cpu": [f"jaxlib=={_current_jaxlib_version}"],
        "gpu": [
            f"jaxlib=={_current_jaxlib_version}+cuda{_default_cuda_version}.cudnn{_default_cudnn_version}"
        ],
    },
)
