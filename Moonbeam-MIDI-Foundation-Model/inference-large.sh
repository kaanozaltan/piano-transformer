#!/bin/bash

export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Define variables
CSV_FILE="preprocessed/839M/train_test_split.csv"
TOP_P=0.95
TEMPERATURE=0.95
MODEL_CONFIG="src/llama_recipes/configs/model_config.json"
CKPT_DIR="checkpoints/pre-trained/moonbeam_839M.pt"
PEFT_WEIGHT="checkpoints/fine-tuned/fine-tuned_3_epoch_839M/0-370/"
TOKENIZER_PATH="tokenizer.model"
MAX_SEQ_LEN=1024
MAX_GEN_LEN=1024
MAX_BATCH_SIZE=6
NUM_TEST_DATA=20
PROMPT_LEN=200

# Run the inference script
torchrun --nproc_per_node=1 recipes/inference/custom_music_generation/unconditional_music_generation.py \
  --csv_file "$CSV_FILE" \
  --top_p "$TOP_P" \
  --temperature "$TEMPERATURE" \
  --model_config_path "$MODEL_CONFIG" \
  --ckpt_dir "$CKPT_DIR" \
  --finetuned_PEFT_weight_path "$PEFT_WEIGHT" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --max_gen_len "$MAX_GEN_LEN" \
  --max_batch_size "$MAX_BATCH_SIZE" \
  --num_test_data "$NUM_TEST_DATA" \
  --prompt_len "$PROMPT_LEN"