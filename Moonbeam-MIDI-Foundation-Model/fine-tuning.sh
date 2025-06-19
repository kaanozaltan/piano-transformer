#!/bin/bash

export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Define variables
PRETRAINED_CKPT="checkpoints/pre-trained/moonbeam_839M.pt"
OUTPUT_DIR="checkpoints/fine-tuned/fine-tuned_3_epoch_839M"
MODEL_NAME="maestro_839M"
DATASET_NAME="maestro_839M"

# Run the training script with torchrun
torchrun --nnodes 1 --nproc_per_node 1 recipes/finetuning/real_finetuning_uncon_gen.py \
  --lr 3e-4 \
  --val_batch_size 2 \
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
  --batch_size_training 2 \
  --context_length 2048 \
  --num_epochs 3 \
  --use_wandb True \
  --gamma 0.99
