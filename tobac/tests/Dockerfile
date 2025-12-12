FROM mambaorg/micromamba
ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN micromamba install -y -n base -c conda-forge  python=3.13 numpy \
    scipy scikit-image pandas pytables matplotlib iris \
    cf-units xarray cartopy trackpy numba pytest pip scikit-learn

COPY . ./

RUN pip install .

RUN pytest
