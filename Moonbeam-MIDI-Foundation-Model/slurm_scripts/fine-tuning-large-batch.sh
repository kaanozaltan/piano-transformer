#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=moonbeam-ft
#SBATCH --output=Moonbeam-MIDI-Foundation-Model/logs/moonbeam_ft_%j.out
#SBATCH --error=Moonbeam-MIDI-Foundation-Model/logs/moonbeam_ft_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=24
#SBATCH --gres=gpu:1
#SBATCH --time=07:00:00
#SBATCH --partition=c23g
#SBATCH --exclude=n23g0001,n23g0002,n23g0003
#SBATCH --account=lect0148
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ikunabel@gmail.com

module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.3
export PYTHONPATH=$(pwd)/src:$(pwd)/Moonbeam-MIDI-Foundation-Model/src:$PYTHONPATH
source Moonbeam-MIDI-Foundation-Model/.venv-moonbeam/bin/activate

# Define variables
PRETRAINED_CKPT="/hpcwork/yh522379/moonbeam/checkpoints/pre-trained/moonbeam_839M.pt"

MODEL_NAME="maestro"
DATASET_NAME="maestro_839M"
MODEL_CONFIG_PATH="Moonbeam-MIDI-Foundation-Model/src/llama_recipes/configs/model_config.json"

# Configuration parameters
LR="5e-5"
WEIGHT_DECAY="0.05"
CONTEXT_LENGTH="512"
BATCH_SIZE="32"
NUM_EPOCHS="150"
GAMMA="0.99"
SCHEDULER_TYPE="cosine"
GEN_SAMPLES="200"
USE_PEFT="False"
PEFT_METHOD="lora"
GEN_TEMPERATURE="0.9"
ENABLE_GENERATION="True"
ENABLE_EVALUATION="True"

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
CONFIG_STRING="839M_${PEFT_STRING}_ctx${CONTEXT_LENGTH}_bs${BATCH_SIZE}_lr${LR}_${SCHEDULER_TYPE}_gamma${GAMMA}${WD_STRING}_temp${GEN_TEMPERATURE}_ep${NUM_EPOCHS}"
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
  --generation_max_gen_len 256 \
  --generation_num_samples "$GEN_SAMPLES" \
  --generation_mode from_scratch \
  --enable_evaluation "$ENABLE_EVALUATION" \
  --evaluation_ref_dir "/hpcwork/lect0148/experiments/mistral-155M_remi_maestro_v8/output/subset/train" \
  --wandb_name "$WANDB_NAME"
