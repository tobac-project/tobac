FROM mambaorg/micromamba
ARG MAMBA_DOCKERFILE_ACTIVATE=1

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml python=3.13 && \
    micromamba clean --all --yes

COPY . ./

RUN pip install . --no-deps

RUN pytest
