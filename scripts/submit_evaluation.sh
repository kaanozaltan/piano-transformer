#!/bin/bash
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --cpus-per-gpu=24
#SBATCH --export=ALL
#SBATCH --job-name=evaluation
#SBATCH --output=logs/job_%j.out
#SBATCH --mail-user=ikunabel@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=lect0148

source .venv/bin/activate
python scripts/evaluate.py "/hpcwork/lect0148/experiments/mistral-162M_remi_maestro_v3/output/train" "/hpcwork/lect0148/data/moonbeam/generations/from_scratch/309M/temperature_1_top_p_0.95_genlen_256" "/hpcwork/lect0148/data/moonbeam/generations/from_scratch/309M/temperature_1_top_p_0.95_genlen_256/metrics"
