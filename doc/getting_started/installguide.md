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

*tobac* can also be installed from PyPI with `pip` or `uv`:

::::{tab-set}
:::{tab-item} `pip`
:sync: pip
```shell
pip install tobac
```
:::
:::{tab-item} `uv`
:sync: uv
```shell
uv pip install tobac
```
:::
::::

The required packages are listed in [`pyproject.toml`](https://github.com/tobac-project/tobac/blob/main/pyproject.toml) (PyPI names), with their conda-forge counterparts in [`pixi.toml`](https://github.com/tobac-project/tobac/blob/main/pixi.toml) and the `environment*.yml` files.

## Development installation

If you are looking to install or run a development version of *tobac*, clone the repository:

> `git clone https://github.com/tobac-project/tobac.git`

and, from within the cloned `tobac` directory, set up an environment with your preferred installer:

::::{tab-set}
:::{tab-item} `pixi`
:sync: pixi
```shell
pixi shell                   # dependencies from conda-forge + editable tobac
pixi run test                # run the test suite
pixi run -e examples jupyter lab   # environment for the example notebooks
pixi run -e docs docs        # build the documentation
```
:::
:::{tab-item} `mamba`
:sync: mamba
```shell
mamba env create -f environment.yml   # or environment-examples.yml for the notebooks
mamba activate tobac-dev
pip install -e . --no-deps
```
:::
:::{tab-item} `conda`
:sync: conda
```shell
conda env create -f environment.yml   # or environment-examples.yml for the notebooks
conda activate tobac-dev
pip install -e . --no-deps
```
:::
:::{tab-item} `pip`
:sync: pip
```shell
pip install -e . --group dev   # needs pip >= 25.1
```
:::
:::{tab-item} `uv`
:sync: uv
```shell
uv sync                        # installs the dev group by default
```
:::
::::

The development requirements are [dependency groups](https://packaging.python.org/en/latest/specifications/dependency-groups/): `test`, `lint`, `dev` (tests and linting) and `docs` (to build the documentation), e.g. `pip install -e . --group docs` or `uv sync --group docs`.
The example notebooks need the `examples` extra, e.g. `pip install -e ".[examples]"` or `uv sync --extra examples`.
Note that some example notebooks also need `ffmpeg`, which is not available from PyPI; it is included in the `pixi` and `conda` examples environments.

You can also directly install the development version from GitHub with pip:

> `pip install --upgrade git+https://github.com/tobac-project/tobac.git`
