#!/usr/bin/zsh

### Job Parameters
#SBATCH --account=lect0148
#SBATCH --gres=gpu:2
#SBATCH --time=00:80:00
#SBATCH --cpus-per-gpu=12
#SBATCH --export=ALL
#SBATCH --job-name=piano-transformer_$1
#SBATCH --partition=c23g
#SBATCH --output=$2
#SBATCH --mail-user=$3
#SBATCH --mail-type=END,FAIL

### Program Code
source .venv/bin/activate
python3 $4.py
