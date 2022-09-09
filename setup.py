from setuptools import setup


"""
This code is from the python documentation and is
designed to read in the version number.
See: https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
"""
import codecs
import os.path


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="tobac",
    version=get_version("tobac/__init__.py"),
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
    packages=["tobac"],
    install_requires=[],
    zip_safe=False,
)
