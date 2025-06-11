set -ex

DOCKER_IMAGE=us-central1-docker.pkg.dev/cds-docker-containers/docker/daintree:test
docker build . -t $DOCKER_IMAGE
docker push $DOCKER_IMAGE

daintree-runner create-sparkles-workflow --config example/model-map.json --models-per-task 10 --test-first-n-tasks 10 > workflow.tmp
sparkles workflow run --nodes 10 daintree-test-10 workflow.tmp -i $DOCKER_IMAGE -u/home/ubuntu/.taiga/token:.taiga-token -u example/model-map.json:model_config.json
