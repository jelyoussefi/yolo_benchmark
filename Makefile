#----------------------------------------------------------------------------------------------------------------------
# Flags
#----------------------------------------------------------------------------------------------------------------------
SHELL:=/bin/bash
CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))


#----------------------------------------------------------------------------------------------------------------------
# Docker Settings
#----------------------------------------------------------------------------------------------------------------------
DOCKER_IMAGE_NAME=advantech_whitepaper_image
export DOCKER_BUILDKIT=1

MODELS ?= "yolo11n yolo11s yolo11m yolo11n-seg yolo11s-seg yolo11m-seg"

MODEL ?= yolo11n-seg
PRECISION ?= FP16
DEVICE ?= GPU
INPUT ?= images/person_horse_dog.jpg
#INPUT ?= images/person_horse.jpg


DOCKER_RUN_PARAMS= \
	-it --rm -a stdout -a stderr  \
	--privileged  \
	-v ${CURRENT_DIR}:/workspace \
	-e HTTP_PROXY=$(HTTP_PROXY) \
	-e HTTPS_PROXY=$(HTTPS_PROXY) \
	-e NO_PROXY=$(NO_PROXY) \
	${DOCKER_IMAGE_NAME}

DOCKER_BUILD_PARAMS := \
	--rm \
	--network=host \
	--build-arg MODELS=$(MODELS) \
	--build-arg http_proxy=$(HTTP_PROXY) \
	--build-arg https_proxy=$(HTTPS_PROXY) \
	--build-arg no_proxy=$(NO_PROXY) \
	-t $(DOCKER_IMAGE_NAME) . 
	
	
#----------------------------------------------------------------------------------------------------------------------
# Targets
#----------------------------------------------------------------------------------------------------------------------
default: benchmark
.PHONY:  

build:
	@$(call msg, Building Docker image ${DOCKER_IMAGE_NAME} ...)
	@docker build ${DOCKER_BUILD_PARAMS}
	
benchmark: build
	@$(call msg, Running the Safety Zone Monitoring ...)
	@docker run ${DOCKER_RUN_PARAMS} bash -c ' \
		python3 ./benchmark.py  \
				--input=${INPUT}	 \
				--model=${MODEL} \
				--precision ${PRECISION} \
				--device ${DEVICE} \
		'
bash: build
	@docker run ${DOCKER_RUN_PARAMS} bash
#----------------------------------------------------------------------------------------------------------------------
# helper functions
#----------------------------------------------------------------------------------------------------------------------
define msg
	tput setaf 2 && \
	for i in $(shell seq 1 120 ); do echo -n "-"; done; echo  "" && \
	echo "         "$1 && \
	for i in $(shell seq 1 120 ); do echo -n "-"; done; echo "" && \
	tput sgr0
endef

