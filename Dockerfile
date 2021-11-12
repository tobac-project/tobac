FROM mambaorg/micromamba

#WORKDIR .
COPY . ./

RUN micromamba install -y -n base -c conda-forge numpy \
    scipy scikit-image pandas pytables matplotlib iris \
    cf-units xarray cartopy trackpy numba pytest

# Make RUN commands use the new environment:
#SHELL ["micromamba", "run", "-n", "myenv", "/bin/bash", "-c"]

RUN pip install .

RUN pytest
