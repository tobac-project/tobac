FROM mambaorg/micromamba

#WORKDIR .

RUN micromamba install -y -n base -c conda-forge numpy \
    scipy scikit-image pandas pytables matplotlib iris \
    cf-units xarray cartopy trackpy numba pytest pip

COPY . ./

RUN pip install .

RUN pytest
