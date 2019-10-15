#!/usr/bin/env bash

# Determine if an instance of the Docker image is already running
EXISTING=`docker ps --filter "name=charcoal-morphotypes-running" --format "{{.ID}}"`
if [ "$EXISTING" != "" ]; then
	
	# Attach an additional interactive shell to the running container
	docker exec -ti "charcoal-morphotypes-running" bash
	
else
	
	# Spin up an instance of the Docker image and launch an interactive shell
	DOCKER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
	docker run --name "charcoal-morphotypes-running" --rm -ti --gpus=all "-v$DOCKER_DIR/..:/hostdir" -w /hostdir -p 6006:6006 adamrehn/charcoal-morphotypes:latest bash
	
fi
