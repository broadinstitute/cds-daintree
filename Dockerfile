# Base image: Using a pre-built image of depmap pipeline run 
# from Google Container Registry
FROM ubuntu:noble

# Install tzdata non-interactively; else it asks for timezone
RUN apt-get update --fix-missing && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata software-properties-common pip rsync curl

# Download and Install Google Cloud SDK
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-458.0.1-linux-x86_64.tar.gz && \
   tar -xf google-cloud-cli-458.0.1-linux-x86_64.tar.gz && \
   ./google-cloud-sdk/install.sh -q && \
   rm google-cloud-cli-458.0.1-linux-x86_64.tar.gz

# Install Python 3.9
RUN add-apt-repository ppa:deadsnakes/ppa && \
  apt-get update && \
  apt-get install -y \
  python3.9 \
  python3.9-distutils \
  python3.9-dev 

RUN apt-get install -y pipx && pipx install poetry==1.8.5

# pipx installed poetry into /root/.local/bin so add to path
ENV PATH="/root/.local/bin:${PATH}"

RUN mkdir -p /daintree
WORKDIR /daintree
COPY daintree_core/ /daintree/daintree_core/
COPY daintree_scripts/ /daintree/daintree_scripts/
COPY poetry.lock /daintree/poetry.lock
COPY pyproject.toml /daintree/pyproject.toml
COPY README.md /daintree/README.md

RUN cd /daintree/ && poetry install

# # Set the entrypoint to run the Python script
# ENTRYPOINT ["/install/depmap-py/bin/python3.9", "-u", "/daintree/daintree_scripts/run_fit_models.py"]
