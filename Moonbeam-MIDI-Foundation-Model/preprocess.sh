#!/bin/bash

export PYTHONPATH=$(pwd)/src:$PYTHONPATH

DATASET_NAME="maestro"
DATASET_FOLDER="data/maestro"
OUTPUT_FOLDER="preprocessed/839M"
TRAIN_TEST_SPLIT="data/maestro/maestro-v3.0.0.csv"

python data_preprocess.py \
  --dataset_name "$DATASET_NAME" \
  --dataset_folder "$DATASET_FOLDER" \
  --output_folder "$OUTPUT_FOLDER" \
  --model_config src/llama_recipes/configs/model_config.json \
  --train_test_split_file "$TRAIN_TEST_SPLIT" \
  --ts_threshold None
  # --train_ratio 0.9 \ 
  