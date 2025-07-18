#!/bin/bash
#SBATCH --job-name=debug_import
#SBATCH --output=logs/debug_import_%j.log
#SBATCH --error=logs/debug_import_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --chdir=/rwthfs/rz/cluster/home/yh522379/lab/piano-transformer/Moonbeam-MIDI-Foundation-Model

export PYTHONPATH=$(pwd)/src:$PYTHONPATH

module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.3

# Activate your virtual environment
source .venv-moonbeam/bin/activate


echo "=== ENVIRONMENT DEBUG ==="
echo "Current working dir: $(pwd)"
echo "Python path: $PYTHONPATH"
which python
python --version
echo "========================="

echo "=== Module Path Check ==="
python -c "import calendar; print('calendar:', calendar.__file__)"
python -c "import pandas; print('pandas:', pandas.__file__)"
python -c "import torch; print('torch:', torch.__file__)"
echo "========================="

echo "=== Running script ==="
python recipes/finetuning/real_finetuning_uncon_gen.py

