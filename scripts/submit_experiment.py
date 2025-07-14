import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

from piano_transformer import config

slurm_template = """#!/bin/bash
#SBATCH --account=lect0148
#SBATCH --gres=gpu:{gpus}
#SBATCH --time={time}
#SBATCH --cpus-per-gpu=24
#SBATCH --export=ALL
#SBATCH --job-name=piano-transformer_{script_name}_{model_name}
#SBATCH --partition=c23g
#SBATCH --output={log_path}
#SBATCH --mail-user={email}
#SBATCH --mail-type=END,FAIL,ALL

ls -l /hpcwork/lect0148/data/maestro/maestro-v3.0.0.csv
source .venv/bin/activate
torchrun --nproc_per_node={gpus} {script_path}
"""


def load_yaml_config(yaml_path):
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def submit_experiment(slurm_path, model_name, script_name, script_path, log_path, gpus, time, email):
    slurm_content = slurm_template.format(
        model_name=model_name,
        log_path=str(log_path.resolve()),
        script_name=script_name,
        script_path=str(script_path.resolve()),
        gpus=gpus,
        time=time,
        email=email,
    )

    with open(slurm_path, "w") as slurm_file:
        slurm_file.write(slurm_content)

    cmd = [
        "sbatch",
        slurm_path,
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    output = result.stdout.decode("utf-8")
    job_id = output.split(" ")[-1]

    print(output)
    print()
    print("Commands:")
    print(f"- Status: \tsqueue --job {job_id}")
    print(f"- Details: \tscontrol show job {job_id}")
    print(f"- Cancel: \tscancel {job_id}")


def main():
    # Example: python scripts/submit_experiment.py --model mistral-409M_remi_maestro --time 18:00:00
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--script", type=str)
    parser.add_argument("--email", type=str)
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--time", type=str)

    args = parser.parse_args()

    file_path = Path(__file__).resolve().parent
    yaml_config = load_yaml_config(file_path / "submit_config.yaml")

    model_name = args.model or yaml_config.get("model")
    script_name = args.script or yaml_config.get("script")
    email = args.email or yaml_config.get("email", "")
    gpus = args.gpus or yaml_config.get("gpus", 1)
    time = args.time or yaml_config.get("time", "00:20:00")

    if not model_name or not script_name:
        print("Error: 'model' and 'script' must be specified via CLI or submit_config.yaml")
        sys.exit(1)

    slurm_path = file_path / "submit_experiment.sh"
    model_path = file_path.parent / "models" / model_name
    cfg = config.load_config(model_path / "config.yaml")
    script_path = model_path / f"{script_name}.py"
    log_dir = cfg.experiment_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"log_{script_name}_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"

    submit_experiment(
        slurm_path=slurm_path,
        model_name=cfg.model_name,
        script_name=script_name,
        script_path=script_path,
        gpus=gpus,
        time=time,
        log_path=log_path,
        email=email,
    )


if __name__ == "__main__":
    main()
