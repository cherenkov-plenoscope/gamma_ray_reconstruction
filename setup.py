import setuptools
import os

with open("README.rst", "r", encoding="utf-8") as f:
    long_description = f.read()


with open(os.path.join("gamma_ray_reconstruction", "version.py")) as f:
    txt = f.read()
    last_line = txt.splitlines()[-1]
    version_string = last_line.split()[-1]
    version = version_string.strip("\"'")


setuptools.setup(
    name="gamma_ray_reconstruction_cherenkov-plenoscope-project",
    version=version,
    author="Sebastian Achim Mueller",
    author_email="sebastian-achim.mueller@mpi-hd.mpg.de",
    description="Reconstruct cosmic gamma-rays from Cherenkov-light-fields",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/cherenkov-plenoscope/gamma_ray_reconstruction",
    packages=[
        "gamma_ray_reconstruction",
        "gamma_ray_reconstruction.energy",
        "gamma_ray_reconstruction.utils",
        "gamma_ray_reconstruction.trajectory",
        "gamma_ray_reconstruction.trajectory.v2020nov12fuzzy0",
        "gamma_ray_reconstruction.trajectory.v2020dec04iron0b",
        "gamma_ray_reconstruction.gamma_hadron",
    ],
    install_requires=[
        "iminuit",  # ==1.4.9
        "binning_utils-sebastian-achim-mueller",
    ],
    package_data={"gamma_ray_reconstruction": []},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
