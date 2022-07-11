from setuptools import setup

setup(
    name="tobac",
    version="1.3.2",
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
