.. _installguide:

Installation
------------

tobac works with Python 3 installations.

The easiest way is to install the most recent version of tobac via conda or mamba and the conda-forge channel:

:code:`conda install -c conda-forge tobac` or :code:`mamba install -c conda-forge tobac`

This will take care of all necessary dependencies and should do the job for most users. It also allows for an easy update of the installation by

:code:`conda update -c conda-forge tobac` :code:`mamba update -c conda-forge tobac`


You can also install conda via pip, which is mainly interesting for development purposes or using specific development branches for the Github repository.

The following python packages are required (including dependencies of these packages):

*numpy*, *scipy*, *scikit-image*, *pandas*, *pytables*, *matplotlib*, *iris*, *xarray*, *cartopy*, *trackpy*
   
If you are using anaconda, the following command should make sure all dependencies are met and up to date:

.. code-block:: console

    conda install -c conda-forge -y numpy scipy scikit-image pandas pytables matplotlib iris xarray cartopy trackpy

You can directly install the package directly from github with pip and either of the two following commands: 

    ``pip install --upgrade git+ssh://git@github.com/tobac-project/tobac.git``

    ``pip install --upgrade git+https://github.com/tobac-project/tobac.git``

You can also clone the package with any of the two following commands: 

    ``git clone git@github.com:tobac-project/tobac.git``

    ``git clone https://github.com/tobac-project/tobac.git``

and install the package from the locally cloned version (The trailing slash is actually necessary):

    ``pip install --upgrade tobac/``