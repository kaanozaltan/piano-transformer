sbatch \
    --export=ALL \
    --partition=c23g \
    --cpus-per-gpu=12 \
    --gres=gpu:2 \
    --time=08:00:00 \
    --mail-user=firstname.lastname@rwth-aachen.de \
    --mail-type=END,FAIL \
    --account=lect0148 \
    piano-transformer/scripts/run_generate.sh