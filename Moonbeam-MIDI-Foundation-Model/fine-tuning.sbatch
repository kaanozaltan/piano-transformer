#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=moonbeam-ft
#SBATCH --output=logs/moonbeam_ft_%j.out
#SBATCH --error=logs/moonbeam_ft_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:15:00
#SBATCH --partition=c23g
#SBATCH --account=lect0148

# module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.3

# Activate virtual environment
source .venv-moonbeam/bin/activate

export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Define variables
PRETRAINED_CKPT="$HPCWORK/moonbeam/checkpoints/pre-trained/moonbeam_839M.pt"
OUTPUT_DIR="$HPCWORK/moonbeam/checkpoints/fine-tuned/839M-10epoch"
MODEL_NAME="maestro"
DATASET_NAME="maestro_839M"
MODEL_CONFIG_PATH="src/llama_recipes/configs/model_config.json"

mkdir -p logs

# Run the fine-tuning script
torchrun --nnodes 1 --nproc_per_node 2 recipes/finetuning/real_finetuning_uncon_gen.py \
  --lr 3e-4 \
  --val_batch_size 64 \
  --run_validation True \
  --validation_interval 10 \
  --save_metrics True \
  --dist_checkpoint_root_folder "$OUTPUT_DIR" \
  --dist_checkpoint_folder ddp \
  --trained_checkpoint_path "$PRETRAINED_CKPT" \
  --pure_bf16 True \
  --enable_ddp True \
  --use_peft True \
  --peft_method lora \
  --quantization False \
  --model_name "$MODEL_NAME" \
  --dataset "$DATASET_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size_training 32 \
  --context_length 2048 \
  --num_epochs 10 \
  --use_wandb True \
  --gamma 0.99 \
  --model_config_path "$MODEL_CONFIG_PATH"
