sbatch \
    --export=ALL \
    --partition=c23g \
    --cpus-per-gpu=24 \
    --gres=gpu:1 \
    --time=4:00:00 \
    --begin=now \
    --signal=TERM@120 \
    --mail-user=kaan.oezaltan@rwth-aachen.de \
    --mail-type=END,FAIL \
    scripts/run.sh