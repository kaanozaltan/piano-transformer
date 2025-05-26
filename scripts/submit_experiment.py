import os
import subprocess
import sys
from pathlib import Path

from piano_transformer import config


slurm_template = """#!/bin/bash
#SBATCH --account=lect0148
#SBATCH --gres=gpu:2
#SBATCH --time=00:20:00
#SBATCH --cpus-per-gpu=12
#SBATCH --export=ALL
#SBATCH --job-name=piano-transformer_{model_name}
#SBATCH --partition=c23g
#SBATCH --output={log_path}
#SBATCH --mail-user={email}
#SBATCH --mail-type=END,FAIL

source .venv/bin/activate
python3 {script_path}
"""


def submit_experiment(slurm_path, model_name, script_path, log_path, email):
    slurm_content = slurm_template.format(
        model_name=model_name,
        email=email,
        log_path=str(log_path.resolve()),
        script_path=str(script_path.resolve())
    )

    with open(slurm_path, "w") as slurm_file:
        slurm_file.write(slurm_content)

    cmd = [
        "sbatch",
        slurm_path,
    ]
    subprocess.run(cmd)


def main():
    experiment = sys.argv[1]
    script = sys.argv[2]
    email = sys.argv[3] if len(sys.argv) >= 3 else ""

    file_path = Path(__file__).resolve().parent
    slurm_path = file_path / "submit_experiment.sh"
    experiment_path = file_path.parent / "experiments" / experiment
    cfg = config.load_config(experiment_path / "config.yaml")
    script_path = experiment_path / f"{script}.py"
    log_path = cfg.experiment_path / "log.txt"

    print(file_path)
    print(experiment_path)
    print(script_path)
    print(log_path.resolve())

    submit_experiment(
        slurm_path=slurm_path,
        model_name=cfg.model_name,
        script_path=script_path,
        log_path=log_path,
        email=email,
    )


if __name__ == "__main__":
    main()
