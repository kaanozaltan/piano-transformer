import subprocess
import sys
from pathlib import Path

from piano_transformer import config


def submit_experiment(model_name, script, log_file, email):
    cmd = [
        "sbatch",
        "submit_experiment.sh",
        model_name,
        log_file,
        email,
        script
    ]
    subprocess.run(cmd)


def main():
    experiment = sys.argv[1]
    script = sys.argv[2]
    email = sys.argv[3] if len(sys.argv) >= 3 else ""

    script_dir = Path(__file__).resolve().parent
    cfg = config.load_config(script_dir.parent / "experiments" / experiment / "config.yaml")

    submit_experiment(
        model_name=cfg.model_name,
        script=script,
        log_file=cfg.model_base_path / "log.txt",
        email=email,
    )


if __name__ == "__main__":
    main()
