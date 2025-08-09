#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=moonbeam-ft
#SBATCH --output=Moonbeam-MIDI-Foundation-Model/logs/moonbeam_ft_%j.out
#SBATCH --error=Moonbeam-MIDI-Foundation-Model/logs/moonbeam_ft_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=24
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --partition=c23g
#SBATCH --account=lect0148
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ikunabel@gmail.com

module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.3
export PYTHONPATH=$(pwd)/src:$(pwd)/Moonbeam-MIDI-Foundation-Model/src:$PYTHONPATH
source Moonbeam-MIDI-Foundation-Model/.venv-moonbeam/bin/activate

# Paths
MODEL_SIZE="309M"  # Change to "309M" for the smaller model

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
NUM_EPOCHS="150"

SCHEDULER_TYPE="cosine"
LR="5e-5"
GAMMA="0.99"

GEN_MAX_LEN="500"
GEN_SAMPLES="200"
GEN_TEMPERATURE="1.1"
ENABLE_GENERATION="True"
ENABLE_EVALUATION="True"
EVALUATION_FREQUENCY_EPOCHS="4"

CONTEXT_LENGTH="1024"
WEIGHT_DECAY="0.01"
BATCH_SIZE="16"

USE_PEFT="False"
PEFT_METHOD="lora"

GRADIENT_ACCUMULATION_STEPS="4"
GRADIENT_CLIPPING="True"
GRADIENT_CLIPPING_THRESHOLD="1.0"

# Generate a random port to avoid conflicts
MASTER_PORT=$((29500 + RANDOM % 1000))

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
# Add gradient clipping threshold to name only if clipping is enabled
if [ "$GRADIENT_CLIPPING" = "True" ]; then
    CLIP_STRING="_clip${GRADIENT_CLIPPING}_clipthresh${GRADIENT_CLIPPING_THRESHOLD}"
else
    CLIP_STRING="_clip${GRADIENT_CLIPPING}"
fi
CONFIG_STRING="${MODEL_SIZE}_${PEFT_STRING}_ctx${CONTEXT_LENGTH}_bs${BATCH_SIZE}_gradacc${GRADIENT_ACCUMULATION_STEPS}${CLIP_STRING}_lr${LR}_${SCHEDULER_TYPE}_gamma${GAMMA}${WD_STRING}_temp${GEN_TEMPERATURE}_ep${NUM_EPOCHS}"
OUTPUT_DIR="$HPCWORK/moonbeam/checkpoints/fine-tuned/ft_${CONFIG_STRING}_${TIMESTAMP}"
WANDB_NAME="moonbeam_ft_${CONFIG_STRING}_${TIMESTAMP}"


mkdir -p logs

# Run the fine-tuning script with random port to avoid conflicts
echo "Using master port: $MASTER_PORT"
torchrun --nnodes 1 --nproc_per_node 1 --master_port $MASTER_PORT Moonbeam-MIDI-Foundation-Model/recipes/finetuning/real_finetuning_uncon_gen.py \
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
  --gradient_clipping "$GRADIENT_CLIPPING" \
  --gradient_clipping_threshold "$GRADIENT_CLIPPING_THRESHOLD"
