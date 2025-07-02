sbatch \
    --export=ALL \
    --partition=c23g \
    --cpus-per-gpu=24 \
    --gres=gpu:1 \
    --time=10:00:00 \
    --begin=now \
    --signal=TERM@120 \
    --mail-user=mail@rwth-aachen.de \
    --mail-type=END,FAIL \
    --account=lect0148 \
    scripts/run.sh
