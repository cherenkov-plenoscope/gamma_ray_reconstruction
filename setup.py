import setuptools
import os

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="gamma_ray_reconstruction",
    version="0.0.3",
    author="Sebastian Achim Mueller",
    author_email="sebastian-achim.mueller@mpi-hd.mpg.de",
    description="Reconstruct cosmic gamma-rays from Cherenkov-light-fields",
    long_description=long_description,
    long_description_content_type="text/md",
    url="https://github.com/cherenkov-plenoscope/starter_kit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
    packages=["gamma_ray_reconstruction",],
    python_requires=">=3.0",
    install_requires=[
        "iminuit",  # ==1.4.9
        "binning_utils-sebastian-achim-mueller",
    ],
)
