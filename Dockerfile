# Base image: Using a pre-built image of depmap pipeline run 
# from Google Container Registry
FROM ubuntu:noble

# Install tzdata non-interactively; else it asks for timezone
RUN apt-get update --fix-missing && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata software-properties-common pip rsync curl \
  build-essential pkg-config libhdf5-dev

# Install Python 3.9
RUN add-apt-repository ppa:deadsnakes/ppa && \
  apt-get update && \
  apt-get install -y \
  python3.9 \
  python3.9-distutils \
  python3.9-dev 
  
RUN apt-get install -y python3.9-venv

RUN apt-get install -y pipx && pipx install poetry==1.8.5

# pipx installed poetry into /root/.local/bin so add to path
ENV PATH="/root/.local/bin:${PATH}"

RUN mkdir -p /daintree
WORKDIR /daintree

# create a virtual env where we'll install the daintree command and add it to our path
RUN python3.9 -m venv /root/daintree 
ENV PATH="/root/daintree/bin:${PATH}"

# copy in the minimal number of files to get `poetry install` to succeed with no error
COPY poetry.lock /daintree/poetry.lock
COPY pyproject.toml /daintree/pyproject.toml
COPY README.md /daintree/README.md
COPY daintree_core/__init__.py /daintree/daintree_core/__init__.py
COPY daintree_runner/__init__.py /daintree/daintree_runner/__init__.py
# do an install before we've copied daintree_core and daintree_runner fully in place so that we can cache this
# layer and skip most of the work poetry install does when we change the source code under daintree_*
# RUN poetry config virtualenvs.create false && poetry env use /root/daintree/bin/python && poetry install
# RUN poetry config virtualenvs.create false && poetry env use /root/daintree/bin/python && poetry install
RUN poetry install

COPY daintree_core/ /daintree/daintree_core/
COPY daintree_runner/ /daintree/daintree_runner/
# reinstall now that the code is in place
RUN poetry install

# create symlinks for the daintree-* commands and put them into our path
RUN poetry run bash -c 'ln `which daintree-core` /root/.local/bin/daintree-core && ln `which daintree-runner` /root/.local/bin/daintree-runner'
