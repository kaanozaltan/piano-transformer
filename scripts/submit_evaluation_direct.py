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
python scripts/evaluate_direct.py "{dir1}" "{dir2}" "{out_dir}"
"""


def submit_evaluation_direct(dir1, dir2, out_dir, email):
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    log_path = out_path / "log.txt"

    slurm_script_path = Path("scripts") / "submit_evaluation_direct.sh"

    slurm_content = slurm_template.format(
        dir1=Path(dir1).resolve(),
        dir2=Path(dir2).resolve(),
        out_dir=out_path,
        log_path=log_path,
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
    # Example:
    #   python scripts/submit_evaluation_direct.py \
    #       /path/to/dir1 /path/to/dir2 /path/to/out <email>
    if len(sys.argv) != 5:
        print(
            "Usage: python scripts/submit_evaluation_direct.py "
            "<dir1> <dir2> <out_dir> <email>"
        )
        sys.exit(1)

    dir1, dir2, out_dir, email = sys.argv[1:]
    submit_evaluation_direct(dir1, dir2, out_dir, email)