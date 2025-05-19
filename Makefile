.PHONY: preprocess filelists all clean train validate download-dataset

ROOT_DIR ?= $(shell pwd)
DATASET_PATH ?= $(ROOT_DIR)/downloaded_datasets/LJSpeech-1.1
CONFIG_PATH ?= datasets/ljs_base/config.yaml
MODEL_NAME ?= ljs_base
SYMLINK_NAME ?= DUMMY1

all: download-dataset preprocess filelists train

download-dataset:
	@echo "Downloading LJSpeech dataset..."
	python3 download_dataset.py

preprocess:
	@echo "Processing mel spectrograms..."
	python3 preprocess/mel_transform.py

filelists:
	@echo "Generating filelists..."
	DATASET_PATH=$(DATASET_PATH) python3 datasets/ljs_base/prepare/filelists.py

train:
	@echo "Initializing training..."
	python train.py -c $(CONFIG_PATH) -m ljs_base

docker-build:
	docker build -t vits2 .


