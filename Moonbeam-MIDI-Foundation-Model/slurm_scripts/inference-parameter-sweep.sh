#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=moonbeam-param-sweep
#SBATCH --output=logs/moonbeam_param_sweep_%j.out
#SBATCH --error=logs/moonbeam_param_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=24
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=c23g
#SBATCH --account=lect0148

module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.3
export PYTHONPATH=$(pwd)/src:$(pwd)/Moonbeam-MIDI-Foundation-Model/src:$PYTHONPATH
source .venv-moonbeam/bin/activate

# Base configuration
CSV_FILE="preprocessed/839M/train_test_split.csv"
MODEL_CONFIG="src/llama_recipes/configs/model_config.json"
CKPT_DIR="/hpcwork/yh522379/moonbeam/checkpoints/pre-trained/moonbeam_839M.pt"
TOKENIZER_PATH="tokenizer.model"
PEFT_WEIGHT="/hpcwork/yh522379/moonbeam/checkpoints/fine-tuned/ft_839M_peft_ctx512_bs32_lr1e-4_cosine_gamma0.99_temp1.1_ep150_20250803_045343/137-20.safetensors"

MAX_SEQ_LEN=1024
MAX_GEN_LEN=512
MAX_BATCH_SIZE=4
NUM_SAMPLES=15  # Small number for quick parameter exploration
PROMPT_LEN=512
GENERATION_MODE="random_files"

# Parameter arrays for grid search
TOP_P_VALUES=(0.6 0.7 0.8 0.9 1.0)
TEMPERATURE_VALUES=(0.7 0.8 0.9 1.0 1.1)

echo "Starting parameter sweep at $(date)"
echo "Total combinations: $((${#TOP_P_VALUES[@]} * ${#TEMPERATURE_VALUES[@]}))"
echo "Samples per combination: $NUM_SAMPLES"
echo "=========================================="

# Counter for progress tracking
COMBO_COUNT=0
TOTAL_COMBOS=$((${#TOP_P_VALUES[@]} * ${#TEMPERATURE_VALUES[@]}))

# Loop through all combinations
for TOP_P in "${TOP_P_VALUES[@]}"; do
    for TEMPERATURE in "${TEMPERATURE_VALUES[@]}"; do
        COMBO_COUNT=$((COMBO_COUNT + 1))
        
        echo ""
        echo "[$COMBO_COUNT/$TOTAL_COMBOS] Processing combination: top_p=$TOP_P, temperature=$TEMPERATURE"
        echo "Started at: $(date)"
        
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
          --generation_mode "$GENERATION_MODE"
        
        if [ $? -eq 0 ]; then
            echo "✅ Combination $COMBO_COUNT completed successfully"
        else
            echo "❌ Combination $COMBO_COUNT failed"
        fi
        
        echo "Finished at: $(date)"
        echo "----------------------------------------"
    done
done

echo ""
echo "🎵 Parameter sweep completed at $(date)"
echo "Generated files are organized by parameter values in the checkpoint directory"
echo "Use the post-process script to validate and analyze the results!"