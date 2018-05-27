#!/bin/bash

DATE=`date '+%Y%m%d_%H%M%S'`

TRAIN_FILE=gs://intelligent-candy-image-classifier/bottlenecks/training.csv
EVAL_FILE=gs://intelligent-candy-image-classifier/bottlenecks/validation.csv
TRAIN_STEPS=1000
OUTPUT_DIR=census_$DATE
JOB_NAME=census_$DATE
GCS_JOB_DIR=gs://intelligent-candy-image-classifier/cloudml-training/$JOB_NAME

gcloud ml-engine jobs submit training $JOB_NAME \
  --stream-logs \
  --runtime-version 1.4 \
  --job-dir $GCS_JOB_DIR \
  --module-name trainer.task \
  --package-path trainer/ \
  --region us-central1 \
  -- \
  --train-files $TRAIN_FILE \
  --eval-files $EVAL_FILE \
  --train-steps $TRAIN_STEPS \
  --eval-steps 100