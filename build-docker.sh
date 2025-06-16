set -x 

VERSION=`poetry version --short`
DOCKER_IMAGE=us-central1-docker.pkg.dev/cds-docker-containers/docker/daintree:$VERSION
if [[ docker manifest inspect $DOCKER_IMAGE > /dev/null ]]; then
  echo "Image $DOCKER_IMAGE already exists. Did you bump the version with 'poetry version minor'?"
  echo "Not building"
  exit 1
else
  echo "Building $DOCKER_IMAGE"
  docker build -t $DOCKER_IMAGE .
  docker push $DOCKER_IMAGE
fi

