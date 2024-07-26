FROM us.gcr.io/broad-achilles/depmap-pipeline-run:ga-build-38

RUN apt-get update --fix-missing
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get -y install software-properties-common littler pip openssh-client rsync

# Download public key from github.com
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

RUN /install/depmap-py/bin/pip install scikit-learn click
RUN /install/depmap-py/bin/pip install git+https://github.com/broadinstitute/cds-ensemble.git@5c16319ce5edad8e0a99c57ac7090f258afe7adc

# ----- start of hermit-specific requirements ------
#
# The requirements for a docker image to work with hermitcrab:
# 1. There must be a user named "ubuntu" with uid 2000
# 2. They must be able to sudo without entering a password
# 3. sshd must be installed 

RUN apt-get update && \
  apt-get install -y openssh-server sudo && \
  mkdir /run/sshd && \
  adduser ubuntu --disabled-password --uid 2000 --gecos "" && \
  echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers 

RUN mkdir /home/ubuntu/.ssh && chmod go-rx /home/ubuntu/.ssh && chown ubuntu:ubuntu /home/ubuntu/.ssh

RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-458.0.1-linux-x86_64.tar.gz
RUN tar -xf google-cloud-cli-458.0.1-linux-x86_64.tar.gz
RUN ./google-cloud-sdk/install.sh -q
# ----- end of hermit-specific requirements ------
    