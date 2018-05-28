#!/bin/bash

DATE=`date '+%Y%m%d_%H%M%S'`

TRAIN_FILE=gs://intelligent-candy-image-classifier/bottlenecks/training.csv
EVAL_FILE=gs://intelligent-candy-image-classifier/bottlenecks/validation.csv
TRAIN_STEPS=1000
OUTPUT_DIR=census_$DATE

python -m trainer.task --train-files ${TRAIN_FILE} \
                       --eval-files ${EVAL_FILE} \
                       --job-dir $OUTPUT_DIR \
                       --train-steps $TRAIN_STEPS \
                       --eval-steps 100
