import subprocess
import sys
from pathlib import Path

slurm_template = """#!/bin/bash
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-gpu=24
#SBATCH --export=ALL
#SBATCH --job-name=evaluation_{experiment}
#SBATCH --output={log_path}
#SBATCH --mail-user={email}
#SBATCH --mail-type=END,FAIL
#SBATCH --account=lect0148

source .venv/bin/activate
python scripts/eval_setup.py "{experiment}"
python scripts/evaluate.py "{experiment}"
"""


def submit_evaluation(experiment, email):
    exp_path = Path("/hpcwork/lect0148/experiments") / experiment
    eval_path = exp_path / "evaluation"
    eval_path.mkdir(parents=True, exist_ok=True)
    log_path = eval_path / "log.txt"

    slurm_script_path = Path("scripts") / "submit_evaluation.sh"

    slurm_content = slurm_template.format(
        experiment=experiment,
        log_path=log_path.resolve(),
        email=email,
    )

    with open(slurm_script_path, "w") as f:
        f.write(slurm_content)

    result = subprocess.run(["sbatch", str(slurm_script_path)], capture_output=True, check=True)
    output = result.stdout.decode().strip()
    job_id = output.split()[-1]

    print(output)
    print("\nCommands:")
    print(f"  Status: squeue --job {job_id}")
    print(f"  Cancel: scancel {job_id}")


if __name__ == "__main__":
    # Example: python scripts/submit_evaluation.py mistral-950M_remi_maestro <email>
    if len(sys.argv) != 3:
        print("Usage: python scripts/submit_evaluation.py <experiment_name> <email>")
        sys.exit(1)

    experiment = sys.argv[1]
    email = sys.argv[2]

    submit_evaluation(experiment, email)