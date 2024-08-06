FROM us.gcr.io/broad-achilles/depmap-pipeline-run:ga-build-38

RUN apt-get update --fix-missing
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get -y install software-properties-common littler pip openssh-client rsync

# Download public key from github.com
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

RUN /install/depmap-py/bin/pip install scikit-learn click
RUN /install/depmap-py/bin/pip install git+https://github.com/broadinstitute/cds-ensemble.git@5c16319ce5edad8e0a99c57ac7090f258afe7adc

RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-458.0.1-linux-x86_64.tar.gz
RUN tar -xf google-cloud-cli-458.0.1-linux-x86_64.tar.gz
RUN ./google-cloud-sdk/install.sh -q
    