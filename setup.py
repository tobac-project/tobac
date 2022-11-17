from setuptools import setup


"""
This code is from the python documentation and is
designed to read in the version number.
See: https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
"""
from pathlib import Path


def read(pkg_name):
    init_fname = Path(__file__).parent / pkg_name / "__init__.py"
    with open(init_fname, "r") as fp:
        return fp.read()


def get_version(pkg_name):
    for line in read(pkg_name).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


PACKAGE_NAME = "tobac"

setup(
    name=PACKAGE_NAME,
    version=get_version(PACKAGE_NAME),
    description="Tracking and object-based analysis of clouds",
    url="http://github.com/tobac-project/tobac",
    author=[
        "Max Heikenfeld",
        "William Jones",
        "Fabian Senf",
        "Sean Freeman",
        "Julia Kukulies",
        "Peter Marinescu",
    ],
    author_email=[
        "max.heikenfeld@physics.ox.ac.uk",
        "william.jones@physics.ox.ac.uk",
        "senf@tropos.de",
        "sean.freeman@colostate.edu",
        "julia.kukulies@gu.se",
        "peter.marinescu@colostate.edu",
    ],
    license="BSD-3-Clause License",
    packages=[PACKAGE_NAME],
    install_requires=[
        "numpy",
        "scipy",
        "scikit-image",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "scitools-iris",
        "xarray",
        "cartopy",
        "trackpy",
    ],
    zip_safe=False,
)
