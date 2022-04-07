from setuptools import setup

setup(
    name="tobac",
    version="1.3",
    description="Tracking and object-based analysis of clouds",
    url="http://github.com/climate-processes/tobac",
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
    license="GNU",
    packages=["tobac"],
    install_requires=[],
    zip_safe=False,
)
