#!/bin/bash

export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Define variables
PRETRAINED_CKPT="$HPCWORK/moonbeam/checkpoints/pre-trained/moonbeam_309M.pt"
OUTPUT_DIR="$HPCWORK/moonbeam/checkpoints/fine-tuned/309M"
MODEL_NAME="maestro"
DATASET_NAME="maestro_309M"
MODEL_CONFIG_PATH="src/llama_recipes/configs/model_config_small.json"

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
  --num_epochs 10 \
  --use_wandb True \
  --gamma 0.99 \
  --model_config_path "$MODEL_CONFIG_PATH" \
  --enable_generation True \
  --generation_save_dir "/hpcwork/yh522379/moonbeam/generation_results" \
  --generation_temperature 1.1 \
  --generation_top_p 0.95 \
  --generation_max_gen_len 256 \
  --generation_num_samples 20 \
  --generation_mode "all_test_files" \
  --enable_evaluation False \
  --evaluation_ref_dir "/hpcwork/lect0148/experiments/mistral-155M_remi_maestro_v8/output/subset/train"
