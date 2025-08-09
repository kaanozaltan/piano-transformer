#!/bin/bash

module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.3
export PYTHONPATH=$(pwd)/src:$(pwd)/Moonbeam-MIDI-Foundation-Model/src:$PYTHONPATH
source Moonbeam-MIDI-Foundation-Model/.venv-moonbeam/bin/activate

# Paths
MODEL_SIZE="839M"  # Change to "309M" for the smaller model

if [ "$MODEL_SIZE" = "839M" ]; then
    PRETRAINED_CKPT="/hpcwork/yh522379/moonbeam/checkpoints/pre-trained/moonbeam_839M.pt"
    MODEL_NAME="maestro"
    DATASET_NAME="maestro_839M"
    MODEL_CONFIG_PATH="Moonbeam-MIDI-Foundation-Model/src/llama_recipes/configs/model_config.json"
elif [ "$MODEL_SIZE" = "309M" ]; then
    PRETRAINED_CKPT="/hpcwork/yh522379/moonbeam/checkpoints/pre-trained/moonbeam_309M.pt"
    MODEL_NAME="maestro"
    DATASET_NAME="maestro_309M"
    MODEL_CONFIG_PATH="Moonbeam-MIDI-Foundation-Model/src/llama_recipes/configs/model_config_small.json"
else
    echo "Error: MODEL_SIZE must be either '839M' or '309M'"
    exit 1
fi

# Params
NUM_EPOCHS="100"

SCHEDULER_TYPE="cosine"
LR="1e-4"
GAMMA="0.99"

GEN_MAX_LEN="512"
GEN_SAMPLES="200"
GEN_TEMPERATURE="1.1"
ENABLE_GENERATION="False"
ENABLE_EVALUATION="False"
EVALUATION_FREQUENCY_EPOCHS="1"

CONTEXT_LENGTH="512"
WEIGHT_DECAY="0"
BATCH_SIZE="32"

USE_PEFT="True"
PEFT_METHOD="lora"

GRADIENT_ACCUMULATION_STEPS="1"
GRADIENT_CLIPPING="True"

# Auto-generate names based on configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ "$USE_PEFT" = "True" ]; then
    PEFT_STRING="peft"
else
    PEFT_STRING="full"
fi
# Add weight decay to name only if it's not 0.0
if [ "$WEIGHT_DECAY" != "0.0" ] && [ "$WEIGHT_DECAY" != "0" ]; then
    WD_STRING="_wd${WEIGHT_DECAY}"
else
    WD_STRING=""
fi
CONFIG_STRING="${MODEL_SIZE}_${PEFT_STRING}_ctx${CONTEXT_LENGTH}_bs${BATCH_SIZE}_gr_acc${GRADIENT_ACCUMULATION_STEPS}_clip${GRADIENT_CLIPPING}_lr${LR}_${SCHEDULER_TYPE}_gamma${GAMMA}${WD_STRING}_temp${GEN_TEMPERATURE}_ep${NUM_EPOCHS}"
OUTPUT_DIR="$HPCWORK/moonbeam/checkpoints/fine-tuned/ft_${CONFIG_STRING}_${TIMESTAMP}"
WANDB_NAME="ft_${CONFIG_STRING}_${TIMESTAMP}"


mkdir -p logs

# Run the fine-tuning script
torchrun --nnodes 1 --nproc_per_node 1 Moonbeam-MIDI-Foundation-Model/recipes/finetuning/real_finetuning_uncon_gen.py \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --val_batch_size "$BATCH_SIZE" \
  --run_validation True \
  --validation_interval 20 \
  --save_metrics True \
  --dist_checkpoint_root_folder "$OUTPUT_DIR" \
  --dist_checkpoint_folder ddp \
  --trained_checkpoint_path "$PRETRAINED_CKPT" \
  --pure_bf16 True \
  --enable_ddp True \
  --use_peft "$USE_PEFT" \
  --peft_method "$PEFT_METHOD" \
  --quantization False \
  --model_name "$MODEL_NAME" \
  --dataset "$DATASET_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size_training "$BATCH_SIZE" \
  --context_length "$CONTEXT_LENGTH" \
  --num_epochs "$NUM_EPOCHS" \
  --use_wandb True \
  --gamma "$GAMMA" \
  --scheduler_type "$SCHEDULER_TYPE" \
  --model_config_path "$MODEL_CONFIG_PATH" \
  --enable_generation "$ENABLE_GENERATION" \
  --generation_save_dir "$OUTPUT_DIR/generation_results" \
  --generation_temperature "$GEN_TEMPERATURE" \
  --generation_top_p 0.95 \
  --generation_max_gen_len "$GEN_MAX_LEN" \
  --generation_num_samples "$GEN_SAMPLES" \
  --generation_mode from_scratch \
  --enable_evaluation "$ENABLE_EVALUATION" \
  --evaluation_ref_dir "/hpcwork/lect0148/experiments/mistral-155M_remi_maestro_v8/output/subset/train" \
  --wandb_name "$WANDB_NAME" \
  --evaluation_frequency_epochs "$EVALUATION_FREQUENCY_EPOCHS" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  -gradient_clipping "$GRADIENT_CLIPPING" \
