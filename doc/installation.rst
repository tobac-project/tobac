Installation
------------
tobac is now capable of working with both Python 2 and Python 3 (tested for 2.7,3.6 and 3.7) installations.

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
