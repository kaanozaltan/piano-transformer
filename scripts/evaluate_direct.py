import sys
import time
from pathlib import Path

from piano_transformer.utils.metrics import (
    evaluate_mgeval_combined,
    fmd,
)


def evaluate(dir1, dir2, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluate_mgeval_combined(dir1, dir2, out_dir)
    print("\nFMD:", fmd(dir1, dir2))


if __name__ == "__main__":
    # Example:
    #   python scripts/evaluate_direct.py /path/real /path/gen /path/out
    if len(sys.argv) != 4:
        print(
            "Usage: python scripts/evaluate_direct.py "
            "<dir1> <dir2> <out_dir>"
        )
        sys.exit(1)

    dir1 = Path(sys.argv[1]).resolve()
    dir2 = Path(sys.argv[2]).resolve()
    out_dir = Path(sys.argv[3]).resolve()

    start_time = time.time()
    print("\nEvaluating:")
    print(f"  dir1: {dir1}")
    print(f"  dir2: {dir2}")
    evaluate(dir1, dir2, out_dir)
    mins, secs = divmod(time.time() - start_time, 60)
    print(f"\nTotal execution time: {int(mins)} min {secs:.2f} sec")
