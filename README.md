tobac - Tracking and Object-based Analysis of Clouds
======
[![Documentation Status](https://readthedocs.org/projects/tobac/badge/?version=latest)](https://tobac.readthedocs.io/en/latest/?badge=latest)[![Download Counter](https://anaconda.org/conda-forge/tobac/badges/downloads.svg)](https://anaconda.org/conda-forge/tobac/)

What is it?
-----------

*tobac* is a Python package for identifiying, tracking and analysing of clouds and other meteorological phenomena in different types of gridded datasets. *tobac* is unique in its ability to track phenomena using **any** variable on **any** grid, including radar data, satellite observations, and numerical model output. *tobac* has been used in a variety of peer-reviewed [publications](https://tobac.readthedocs.io/en/rc_v1.4.0/publications.html) and is an international, multi-institutional collaboration. 

Documentation
-------------
Individual features are identified as either maxima or minima in a two dimensional time varying field.
The volume/area associated with the identified objects can be determined based on a time-varying 2D or 3D field and a threshold value. The in thre tracking step, the identified objects are linked into consistent trajectories representing the cloud over its lifecycle.

Detailed documentation of the package can be found at https://tobac.readthedocs.io.

Release announcements, workshop and conference announcements, and other information of interest to the broader *tobac* users group are sent to the [tobac core group](https://groups.google.com/g/tobac/about) mailing list. If you are interested in contributing to the development of *tobac*, we invite you to join the [tobac developers](https://groups.google.com/u/1/g/tobac-developers) mailing list. Information on monthly developers' meetings and other developer discussion and announcements are sent to that list. 

We also have a Slack server for both users and developers. For information on joining that, please contact the *tobac* developers mailing list, or see the information in the *tobac* release notes sent to the *tobac* mailing list. 

Installation
------------
tobac requires Python 3, and support for Python versions before 3.7 (i.e., 3.6 and lower) is deprecated and will be removed in tobac version 1.5.

The easiest way is to install the most recent version of tobac via conda and the conda-forge channel:
```
conda install -c conda-forge tobac 
```
This will take care of all necessary dependencies and should do the job for most users and also allows for an easy update of the installation by
```
conda update -c conda-forge tobac 
```


You can also install conda via git, either for development purposes or to use specific development branches for the Github repository.

If you are using anaconda, the following command from within the cloned repository should make sure all dependencies are met and up to date:
```
conda install -c conda-forge --yes --file conda-requirements.txt
```
You can directly install the package directly from github with pip and either of the two following commands:
```
pip install --upgrade git+ssh://git@github.com/tobac-project/tobac.git
pip install --upgrade git+https://github.com/tobac-project/tobac.git
```
You can also clone the package with any of the two following commands
```
git clone git@github.com:tobac-project/tobac.git
git clone https://github.com/tobac-project/tobac.git
```
and install the package from the locally cloned version:
```
pip install tobac/
```

Contributing
------------
We encourage bug reports, questions, and code contributions. For more details on contributing, please see https://github.com/tobac-project/tobac/blob/v2.0-dev/CONTRIBUTING.md

We are currently in a transition phase between versions 1.x and 2.x. v2.x will enable the use of multiple tracking methods (including TINT) and will use xarray for gridded data instead of Iris. Preliminary development on v2.x has taken place on the `v2.0-dev` branch, while work on the `main` and `RC_v1.x.x` branches (containing v1.x development) is ongoing to unify these development efforts. 

Roadmap
------------
A roadmap for the future development of tobac is available here: https://github.com/tobac-project/tobac-roadmap/blob/master/tobac-roadmap-main.md
