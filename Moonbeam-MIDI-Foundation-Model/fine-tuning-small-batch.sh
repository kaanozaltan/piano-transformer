#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=moonbeam-ft
#SBATCH --output=logs/moonbeam_ft_%j.out
#SBATCH --error=logs/moonbeam_ft_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=24
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --partition=c23g
#SBATCH --account=lect0148

module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.3
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
source .venv-moonbeam/bin/activate

# Constant paths
PRETRAINED_CKPT="$HPCWORK/moonbeam/checkpoints/pre-trained/moonbeam_309M.pt"
MODEL_NAME="maestro"
DATASET_NAME="maestro_309M"
MODEL_CONFIG_PATH="src/llama_recipes/configs/model_config_small.json"

# Change between configs
OUTPUT_DIR="$HPCWORK/moonbeam/checkpoints/fine-tuned/fine-tune_309M_context_256_batch_64_lr_5e-5_gamma_0.98_epoch_100"
WANDB_NAME="fine-tune_309M_context_256_batch_64_lr_5e-5_gamma_0.98_epoch_100"

mkdir -p logs

# Config for small context length
torchrun --nnodes 1 --nproc_per_node 1 recipes/finetuning/real_finetuning_uncon_gen.py \
  --lr 5e-5 \
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
  --batch_size_training 64 \
  --context_length 256 \
  --num_epochs 100 \
  --use_wandb True \
  --gamma 0.98 \
  --model_config_path "$MODEL_CONFIG_PATH" \
  --wandb_name "$WANDB_NAME"

  # Original config
  # torchrun --nnodes 1 --nproc_per_node 1 recipes/finetuning/real_finetuning_uncon_gen.py \
  # --lr 3e-4 \
  # --val_batch_size 64 \
  # --run_validation True \
  # --validation_interval 10 \
  # --save_metrics True \
  # --dist_checkpoint_root_folder "$OUTPUT_DIR" \
  # --dist_checkpoint_folder ddp \
  # --trained_checkpoint_path "$PRETRAINED_CKPT" \
  # --pure_bf16 True \
  # --enable_ddp True \
  # --use_peft True \
  # --peft_method lora \
  # --quantization False \
  # --model_name "$MODEL_NAME" \
  # --dataset "$DATASET_NAME" \
  # --output_dir "$OUTPUT_DIR" \
  # --batch_size_training 32 \
  # --context_length 256 \
  # --num_epochs 50 \
  # --use_wandb True \
  # --gamma 0.99 \
  # --model_config_path "$MODEL_CONFIG_PATH" \
  # --wandb_name "$WANDB_NAME"
