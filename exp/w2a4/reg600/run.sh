#!/bin/bash
PYTHONPATH=../../../:$PYTHONPATH \
python ../../../qdrop/solver/main_imagenet.py --config config.yaml
