import sys
import time
from pathlib import Path

from piano_transformer.utils.metrics import (
    evaluate_mgeval_combined,
    fmd,
)

BASE_DIR = Path("/hpcwork/lect0148/experiments")

def evaluate(dir1, dir2, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluate_mgeval_combined(dir1, dir2, out_dir)
    print(f"\nFMD:", fmd(dir1, dir2))


if __name__ == "__main__":
    # Example: python scripts/evaluate.py mistral-950M_remi_maestro
    if len(sys.argv) != 2:
        print("Usage: python scripts/evaluate.py <experiment_name>")
        sys.exit(1)

    experiment_name = sys.argv[1]
    exp_path = BASE_DIR / experiment_name
    subset_path = exp_path / "output" / "subset"
    eval_path = exp_path / "evaluation"

    unconditional_out = eval_path / "unconditional"
    conditional_out = eval_path / "conditional"

    start_time = time.time()

    print("\nEvaluating unconditional generations")
    evaluate(subset_path / "train", subset_path / "generations", unconditional_out)

    print("\nEvaluating conditional continuations")
    evaluate(subset_path / "test", subset_path / "continuations", conditional_out)

    end_time = time.time()
    mins, secs = divmod(end_time - start_time, 60)
    print(f"\nTotal execution time: {int(mins)} min {secs:.2f} sec")