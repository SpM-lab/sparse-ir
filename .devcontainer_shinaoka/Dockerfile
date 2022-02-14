FROM continuumio/anaconda3

ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    git \
    zip \
    vim \
    cmake pkg-config gfortran \
    && \
    apt-get clean && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* # clean up

# Create non-root user
#ARG NB_USER=vscode
#ARG NB_UID=1000
#ARG WORK_DIR=/home/${NB_USER}/work
#RUN useradd -u $NB_UID -m $NB_USER -s /bin/bash && \
    #echo 'vscode ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
#USER $NB_USER

#RUN mkdir /home/${NB_USER}/work
#RUN mkdir /vscode
