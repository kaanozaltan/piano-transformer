#!/bin/bash

module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.3
export PYTHONPATH=$(pwd)/src:$(pwd)/Moonbeam-MIDI-Foundation-Model/src:$PYTHONPATH
source .venv-moonbeam/bin/activate

# Configurable variables for comparison experiments
CSV_FILE="preprocessed/839M/train_test_split.csv"
TOP_P=0.95
TEMPERATURE=1.1
MODEL_CONFIG="src/llama_recipes/configs/model_config.json"
CKPT_DIR="/hpcwork/yh522379/moonbeam/checkpoints/pre-trained/moonbeam_839M.pt"
TOKENIZER_PATH="tokenizer.model"
PEFT_WEIGHT="/hpcwork/yh522379/moonbeam/checkpoints/fine-tuned/ft_839M_peft_ctx512_bs32_lr1e-4_cosine_gamma0.99_temp1.1_ep150_20250803_045343/137-20.safetensors"

MAX_SEQ_LEN=1024
MAX_GEN_LEN=610
MAX_BATCH_SIZE=4
NUM_SAMPLES=600  # Increased for high-fidelity comparison
PROMPT_LEN=610
GENERATION_MODE="all_test_files"  # "from_scratch", "random_files", or "all_test_files"

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
  --generation_mode "$GENERATION_MODE"