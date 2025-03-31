This python module contains code which was originally from a project called "CDS-ensemble"

The code is used to build Random Forest models and also included code for pre-processing 
and massaging the data before passing it to scikit learn's RF module.

This code is intended to solely to be used by a docker image, because daintree uses
sparkles to distribute the training process.

This directory also contains a Dockerfile for building that image.

To build:

```
docker build -t daintree-core .
```

