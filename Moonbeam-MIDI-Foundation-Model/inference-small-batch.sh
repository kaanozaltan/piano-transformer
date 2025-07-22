#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=moonbeam-ft
#SBATCH --output=logs/moonbeam_inference_%j.out
#SBATCH --error=logs/moonbeam_inference_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=24
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --partition=c23g
#SBATCH --account=lect0148

module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.3
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
source .venv-moonbeam/bin/activate

# Define variables
CSV_FILE="preprocessed/309M/train_test_split.csv"
TOP_P=0.95
TEMPERATURE=1.05
MODEL_CONFIG="src/llama_recipes/configs/model_config_small.json"
CKPT_DIR="/hpcwork/yh522379/moonbeam/checkpoints/pre-trained/moonbeam_309M.pt"
TOKENIZER_PATH="tokenizer.model"
PEFT_WEIGHT="/hpcwork/yh522379/moonbeam/checkpoints/fine-tuned/309M-50epoch/49-20.safetensors"

MAX_SEQ_LEN=1024
MAX_GEN_LEN=1024
MAX_BATCH_SIZE=4
NUM_SAMPLES=400
PROMPT_LEN=100
FROM_SCRATCH=True

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