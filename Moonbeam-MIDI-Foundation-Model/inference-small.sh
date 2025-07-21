#!/bin/bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Define variables
CSV_FILE="preprocessed/309M/train_test_split.csv"
TOP_P=0.95
TEMPERATURE=1
MODEL_CONFIG="src/llama_recipes/configs/model_config_small.json"
CKPT_DIR="/hpcwork/yh522379/moonbeam/checkpoints/pre-trained/moonbeam_309M.pt"
TOKENIZER_PATH="tokenizer.model"
PEFT_WEIGHT="/hpcwork/yh522379/moonbeam/checkpoints/fine-tuned/fine-tune_309M_context_256_batch_64_lr_5e-5_gamma_0.98_epoch_100/99-10.safetensors"

MAX_SEQ_LEN=256
MAX_GEN_LEN=256
MAX_BATCH_SIZE=4
NUM_SAMPLES=400
PROMPT_LEN=256
FROM_SCRATCH=False

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
  --num_samples "$NUM_SAMPLES" \
  --prompt_len "$PROMPT_LEN" \
  --from-scratch "$FROM_SCRATCH" \
  --folder "$FOLDER"