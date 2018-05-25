#!/bin/bash

DATE=`date '+%Y%m%d_%H%M%S'`

export TRAIN_FILE=gs://intelligent-candy-image-classifier/bottlenecks/training.csv
export EVAL_FILE=gs://intelligent-candy-image-classifier/bottlenecks/validation.csv
export TRAIN_STEPS=1000
export OUTPUT_DIR=census_$DATE

python -m trainer.task --train-files="gs://intelligent-candy-image-classifier/bottlenecks/training.csv" --eval-files="gs://intelligent-candy-image-classifier/bottlenecks/validation.csv" \
  --job-dir census_$DATE \
  --train-steps 1000 \
  --eval-steps 100