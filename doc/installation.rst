Installation
------------
tobac is now capable of working with both Python 2 and Python 3 (tested for 2.7,3.6 and 3.7) installations.

The easiest way is to install the most recent version of tobac via conda or mamba and the conda-forge channel:

```
conda install -c conda-forge tobac 
```
or
```
mamba install -c conda-forge tobac
```

This will take care of all necessary dependencies and should do the job for most users and also allows for an easy update of the installation by

```
conda update -c conda-forge tobac 
```
or
```
mamba update -c conda-forge tobac 
```


You can also install conda via pip, which is mainly interesting for development purposed or to use specific development branches for the Github repository.

The follwoing python packages are required (including dependencies of these packages):
   
*trackpy*, *scipy*, *numpy*, *iris*, *scikit-learn*, *scikit-image*, *cartopy*, *pandas*, *pytables* 


If you are using anaconda, the following command should make sure all dependencies are met and up to date:
    ``conda install -c conda-forge -y trackpy scipy numpy iris scikit-learn scikit-image cartopy pandas pytables``

You can directly install the package directly from github with pip and either of the two following commands: 

    ``pip install --upgrade git+ssh://git@github.com/climate-processes/tobac.git``

    ``pip install --upgrade git+https://github.com/climate-processes/tobac.git``

You can also clone the package with any of the two following commands: 

    ``git clone git@github.com:climate-processes/tobac.git``

    ``git clone https://github.com/climate-processes/tobac.git``

and install the package from the locally cloned version (The trailing slash is actually necessary):

    ``pip install --upgrade tobac/``
