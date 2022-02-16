FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    git \
    zip \
    vim \
    gosu \
    hdf5-tools \
    libopenblas-base \
    libopenblas-dev \
    libhdf5-103 \
    libhdf5-dev \
    libeigen3-dev \
    cmake \
    pkg-config \
    gfortran \
    openmpi-bin \
    libopenmpi-dev \
    sudo \
    libboost-dev \
    libboost-mpi-dev \
    && \
    apt-get clean && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* # clean up


# Create non-root user
ARG NB_USER=vscode
ARG NB_UID=1000
RUN useradd -u $NB_UID -m $NB_USER -s /bin/bash && \
    echo 'vscode ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER $NB_USER
ENV PATH "/home/${NB_USER}/.local/bin:/opt/pomerol2dcore/bin:${PATH}"
ENV PYTHONPATH "/home/${NB_USER}/work/src:${PYTONPATH}"

# for vscode
RUN mkdir /home/${NB_USER}/work

RUN pip3 install -U pip && \
    pip3 install scipy h5py toml tomli matplotlib pytest mypy xprec irbasis
