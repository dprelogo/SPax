from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="SPax",
    version="0.1dev",
    author="David Prelogovic",
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
    python_requires=">=3.6",
    install_requires=[
        "jax",
        "jaxlib",
        "numpy",
        "h5py",
        "scikit-learn",
        ],
)