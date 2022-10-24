from setuptools import setup

import sys
import warnings

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


if sys.version_info < (3, 7):
    warnings.warn(
        "Support for Python versions less than 3.7 is deprecated. "
        "Version 1.5 of tobac will require Python 3.7 or later.",
        DeprecationWarning,
    )


PACKAGE_NAME = "tobac"

# See classifiers list at: https://pypi.org/classifiers/
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Environment :: Console",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: BSD License",
    "Operating System :: POSIX :: Linux",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Atmospheric Science",
]


setup(
    name=PACKAGE_NAME,
    version=get_version(PACKAGE_NAME),
    description="Tracking and object-based analysis of clouds",
    url="http://github.com/tobac-project/tobac",
    classifiers=CLASSIFIERS,
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
    install_requires=[],
    zip_safe=False,
)
