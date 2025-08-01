#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=moonbeam-ft
#SBATCH --output=logs/moonbeam_ft_%j.out
#SBATCH --error=logs/moonbeam_ft_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=24
#SBATCH --gres=gpu:1
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

OUTPUT_DIR="$HPCWORK/moonbeam/checkpoints/fine-tuned/fine-tune_839M_context_512_batch_16_lr_3e-5_gamma_0.98_epoch_50"
WANDB_NAME="fine-tune_839M_context_1024_batch_16_lr_3e-5_gamma_0.98_epoch_100"

mkdir -p logs

# Run the fine-tuning script
torchrun --nnodes 1 --nproc_per_node 1 recipes/finetuning/real_finetuning_uncon_gen.py \
  --lr 3e-4 \
  --val_batch_size 16 \
  --run_validation True \
  --validation_interval 20 \
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
  --batch_size_training 16 \
  --context_length 1024\
  --num_epochs 100 \
  --use_wandb True \
  --gamma 0.99 \
  --scheduler_type steplr \
  --model_config_path "$MODEL_CONFIG_PATH" \
  --enable_generation True \
  --generation_save_dir "$HPCWORK/moonbeam/generation_results" \
  --generation_temperature 1.1 \
  --generation_top_p 0.95 \
  --generation_max_gen_len 256 \
  --generation_num_samples 200 \
  --generation_mode random_files \
  --enable_evaluation True \
  --evaluation_ref_dir "$HPCWORK/experiments/mistral-155M_remi_maestro_v8/output/subset/train" \
  --wandb_name "$WANDB_NAME"
