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


def get_requirements(requirements_filename):
    requirements_file = Path(__file__).parent / requirements_filename
    assert requirements_file.exists()
    with open(requirements_file) as f:
        requirements = [
            line.strip() for line in f.readlines() if not line.startswith("#")
        ]
    # Iris has a different name on PyPI...
    if "iris" in requirements:
        requirements.remove("iris")
        requirements.append("scitools-iris")
    return requirements


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
        "sean.freeman@uah.edu",
        "julia.kukulies@gu.se",
        "peter.marinescu@colostate.edu",
    ],
    license="BSD-3-Clause License",
    packages=[PACKAGE_NAME, PACKAGE_NAME + ".utils"],
    install_requires=get_requirements("requirements.txt"),
    test_requires=["pytest"],
    zip_safe=False,
)
