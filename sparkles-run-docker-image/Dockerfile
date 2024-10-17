# Base image: Using a pre-built image of depmap pipeline run 
# from Google Container Registry
FROM us.gcr.io/broad-achilles/depmap-pipeline-run:ga-build-38

RUN apt-get update --fix-missing

# Install tzdata non-interactively; else it asks for timezone
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata

RUN apt-get -y install software-properties-common littler pip openssh-client rsync

# Download public key from github.com
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

RUN /install/depmap-py/bin/pip install scikit-learn click
RUN echo "Test 8"
RUN /install/depmap-py/bin/pip install git+https://github.com/broadinstitute/cds-daintree.git@pearson-correlation#subdirectory=daintree/daintree_package

# Download and Install Google Cloud SDK
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-458.0.1-linux-x86_64.tar.gz
RUN tar -xf google-cloud-cli-458.0.1-linux-x86_64.tar.gz
RUN ./google-cloud-sdk/install.sh -q
