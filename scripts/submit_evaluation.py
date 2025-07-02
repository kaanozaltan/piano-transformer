import subprocess
import sys
from pathlib import Path

slurm_template = """#!/bin/bash
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-gpu=24
#SBATCH --export=ALL
#SBATCH --job-name=evaluation
#SBATCH --output={log_path}
#SBATCH --mail-user={email}
#SBATCH --mail-type=END,FAIL
#SBATCH --account=lect0148

source .venv/bin/activate
python scripts/evaluate.py "{dataset1}" "{dataset2}" "{output_dir}"
"""


def submit_evaluation(dataset1, dataset2, output_dir, email):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log.txt"

    slurm_script_path = Path("scripts") / "submit_evaluation.sh"

    slurm_content = slurm_template.format(
        dataset1=dataset1,
        dataset2=dataset2,
        output_dir=output_dir,
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
    if len(sys.argv) != 5:
        print("Usage: python scripts/submit_evaluation.py <dataset1> <dataset2> <output_dir> <email>")
        sys.exit(1)

    dataset1 = sys.argv[1]
    dataset2 = sys.argv[2]
    output_dir = sys.argv[3]
    email = sys.argv[4]

    submit_evaluation(dataset1, dataset2, output_dir, email)
