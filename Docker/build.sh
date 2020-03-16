PARENT=$(cd `dirname $0` && cd .. &&  pwd)

DOCKERFILE="$PARENT/Docker/Dockerfile"
IMAGE="robie2011/mlpipe"
cd $PARENT

echo "Dockerfile: $DOCKERFILE"
echo "Directory: $PARENT"
echo "Create Image: $IMAGE"


docker build . -f $DOCKERFILE -t $IMAGE
cd -