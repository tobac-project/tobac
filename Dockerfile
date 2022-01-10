FROM mambaorg/micromamba
ARG MAMBA_DOCKERFILE_ACTIVATE=1

#WORKDIR .

RUN micromamba install -y -n base -c conda-forge numpy \
    scipy scikit-image pandas pytables matplotlib iris \
    cf-units xarray cartopy trackpy numba pytest

COPY . ./

RUN pip install .

RUN pytest
