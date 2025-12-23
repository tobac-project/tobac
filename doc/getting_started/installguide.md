(installguide)=

# Installation

If you are new to `python`, we recommend the [Project Pythia Foundations Course](https://foundations.projectpythia.org/) and using the [Miniforge](https://conda-forge.org/download/) Python bundle.

Once you have Python and an installer, the easiest way to install the most recent version of tobac is via `mamba` or `conda` and the conda-forge channel:

::::{tab-set}
:::{tab-item} `mamba`
:sync: mamba
```shell
mamba install -c conda-forge tobac
```
:::
:::{tab-item} `conda`
:sync: conda
```shell
conda install -c conda-forge tobac
```
:::
::::

This will take care of all necessary dependencies and should do the job for most users. It also allows for an easy update of the installation by

::::{tab-set}
:::{tab-item} `mamba`
:sync: mamba
```shell
mamba update -c conda-forge tobac
```
:::
:::{tab-item} `conda`
:sync: conda
```shell
conda update -c conda-forge tobac
```
:::
::::

The following python packages are required (including dependencies of these packages):

`conda-forge numpy scipy scikit-image scikit-learn pandas matplotlib iris xarray cartopy trackpy typing_extensions`

If you are looking to install or run a development version of *tobac*, you can install the prerequisites using `mamba` or `conda` using the following commands:

::::{tab-set}
:::{tab-item} `mamba`
:sync: mamba
```shell
mamba install -c conda-forge numpy scipy scikit-image scikit-learn pandas matplotlib iris xarray cartopy trackpy typing_extensions
```
:::
:::{tab-item} `conda`
:sync: conda
```shell
conda install -c conda-forge numpy scipy scikit-image scikit-learn pandas matplotlib iris xarray cartopy trackpy typing_extensions
```
:::
::::


While we don't recommend it, you can directly install the package directly from github with pip and either of the two following commands:

> `pip install --upgrade git+ssh://git@github.com/tobac-project/tobac.git`
>
> `pip install --upgrade git+https://github.com/tobac-project/tobac.git`

You can also clone the package with any of the two following commands:

> `git clone git@github.com:tobac-project/tobac.git`
>
> `git clone https://github.com/tobac-project/tobac.git`

and install the package from the locally cloned version (The trailing slash is actually necessary):

> `pip install --upgrade tobac/`
