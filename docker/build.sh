#!/usr/bin/env bash

# The version of TensorFlow that we target
TENSORFLOW_VERSION='1.9.0'

# The version of Keras that we target
KERAS_VERSION='2.2.2'

# Determine if the Docker image for Keras already exists
EXISTING=`docker images keras:$KERAS_VERSION --format "{{.ID}}"`
if [ "$EXISTING" == "" ]; then
	
	# Clone the Keras source repo into a temporary directory
	KERAS_TEMPDIR='/tmp/keras'
	test -d "$KERAS_TEMPDIR" && rm -f -R "$KERAS_TEMPDIR"
	git clone --progress --depth 1 -b $KERAS_VERSION https://github.com/keras-team/keras.git "$KERAS_TEMPDIR"
	
	# Patch the Dockerfile to use the specified TensorFlow release instead of the latest release
	sed -i "s/tensorflow-gpu/tensorflow-gpu==$TENSORFLOW_VERSION/" "$KERAS_TEMPDIR/docker/Dockerfile"
	
	# Patch the Dockerfile to use the specified Keras release instead of the master branch
	sed -i "s/git clone/git clone -b $KERAS_VERSION/" "$KERAS_TEMPDIR/docker/Dockerfile"
	sed -i "s/keras.git \\&\\&/keras.git@$KERAS_VERSION \\&\\&/" "$KERAS_TEMPDIR/docker/Dockerfile"
	
	# Patch the Dockerfile to use the appropriate versions of the dependencies for Keras
	sed -i "s/sklearn_pandas/sklearn_pandas==1.7.0 scikit-learn==0.19.2 scipy==1.1.0 pandas==0.22.0 numpy==1.14.5/" "$KERAS_TEMPDIR/docker/Dockerfile"
	sed -i "s/bcolz \\\\/bcolz=1.2.1 \\\\/" "$KERAS_TEMPDIR/docker/Dockerfile"
	sed -i "s/h5py \\\\/h5py=2.8.0 \\\\/" "$KERAS_TEMPDIR/docker/Dockerfile"
	sed -i "s/matplotlib \\\\/matplotlib=2.2.2 \\\\/" "$KERAS_TEMPDIR/docker/Dockerfile"
	sed -i "s/mkl \\\\/mkl=2018.0.3 \\\\/" "$KERAS_TEMPDIR/docker/Dockerfile"
	sed -i "s/nose \\\\/nose=1.3.7 \\\\/" "$KERAS_TEMPDIR/docker/Dockerfile"
	sed -i "s/notebook \\\\/notebook=5.6.0 \\\\/" "$KERAS_TEMPDIR/docker/Dockerfile"
	sed -i "s/Pillow \\\\/Pillow=5.2.0 \\\\/" "$KERAS_TEMPDIR/docker/Dockerfile"
	sed -i "s/pandas \\\\/pandas=0.22.0 numpy=1.14.5 \\\\/" "$KERAS_TEMPDIR/docker/Dockerfile"
	sed -i "s/pygpu \\\\/pygpu=0.7.6 \\\\/" "$KERAS_TEMPDIR/docker/Dockerfile"
	sed -i "s/pyyaml \\\\/pyyaml=3.12 \\\\/" "$KERAS_TEMPDIR/docker/Dockerfile"
	sed -i "s/scikit-learn \\\\/scikit-learn=0.19.2 \\\\/" "$KERAS_TEMPDIR/docker/Dockerfile"
	sed -i "s/six \\\\/six=1.11.0 \\\\/" "$KERAS_TEMPDIR/docker/Dockerfile"
	sed -i "s/theano \\&\\&/theano=1.0.2 \\&\\&/" "$KERAS_TEMPDIR/docker/Dockerfile"
	
	# Patch the Dockerfile to comment out the CMD directive
	sed -i "s/CMD jupyter/#CMD jupyter/" "$KERAS_TEMPDIR/docker/Dockerfile"
	
	# Build the Docker image for Keras and remove the temporary directory
	docker build -t keras:$KERAS_VERSION "$KERAS_TEMPDIR/docker"
	rm -f -R "$KERAS_TEMPDIR"
	
fi

# Build the Docker image for our project
DOCKER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
docker build --build-arg KERAS_VERSION=$KERAS_VERSION -t adamrehn/charcoal-morphotypes:latest "$DOCKER_DIR"
