#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=moonbeam-ft
#SBATCH --output=logs/moonbeam_ft_%j.out
#SBATCH --error=logs/moonbeam_ft_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=24
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --partition=c23g
#SBATCH --account=lect0148

module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.3
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
source .venv-moonbeam/bin/activate

# Define variables
PRETRAINED_CKPT="$HPCWORK/moonbeam/checkpoints/pre-trained/moonbeam_839M.pt"

MODEL_NAME="maestro"
DATASET_NAME="maestro_839M"
MODEL_CONFIG_PATH="src/llama_recipes/configs/model_config.json"

OUTPUT_DIR="$HPCWORK/moonbeam/checkpoints/fine-tuned/839M"
WANDB_NAME="fine-tune_839M"

mkdir -p logs

# Run the fine-tuning script
torchrun --nnodes 1 --nproc_per_node 2 recipes/finetuning/real_finetuning_uncon_gen.py \
  --lr 3e-4 \
  --val_batch_size 16 \
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
  --batch_size_training 8 \
  --context_length 2048 \
  --num_epochs 30 \
  --use_wandb True \
  --gamma 0.99 \
  --model_config_path "$MODEL_CONFIG_PATH" \
  --wandb_name "$WANDB_NAME"
