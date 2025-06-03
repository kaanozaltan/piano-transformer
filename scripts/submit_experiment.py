import subprocess
import sys
from datetime import datetime
from pathlib import Path

from piano_transformer import config

slurm_template = """#!/bin/bash
###SBATCH --account=lect0148
#SBATCH --gres=gpu:2
#SBATCH --time=00:15:00
#SBATCH --cpus-per-gpu=24
#SBATCH --export=ALL
#SBATCH --job-name=piano-transformer_{script_name}_{model_name}
#SBATCH --partition=c23g
#SBATCH --output={log_path}
#SBATCH --mail-user={email}
#SBATCH --mail-type=END,FAIL,ALL

source .venv/bin/activate
torchrun --nproc_per_node=2 {script_path}
"""


def submit_experiment(slurm_path, model_name, script_name, script_path, log_path, email):
    slurm_content = slurm_template.format(
        model_name=model_name,
        email=email,
        log_path=str(log_path.resolve()),
        script_name=script_name,
        script_path=str(script_path.resolve())
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
    experiment = sys.argv[1]
    script_name = sys.argv[2]
    email = sys.argv[3] if len(sys.argv) >= 4 else ""

    file_path = Path(__file__).resolve().parent
    slurm_path = file_path / "submit_experiment.sh"
    model_path = file_path.parent / "models" / experiment
    cfg = config.load_config(model_path / "config.yaml")
    script_path = model_path / f"{script_name}.py"
    log_path = cfg.experiment_path / "logs" / f"log_{script_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

    (cfg.experiment_path / "logs").mkdir(parents=True, exist_ok=True)

    submit_experiment(
        slurm_path=slurm_path,
        model_name=cfg.model_name,
        script_name=script_name,
        script_path=script_path,
        log_path=log_path,
        email=email,
    )


if __name__ == "__main__":
    main()
