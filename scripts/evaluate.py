import sys
import time
from pathlib import Path

from piano_transformer.utils.metrics import (
    create_subset,
    evaluate_mgeval_combined,
    fmd,
)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python evaluate.py <dataset1_path> <dataset2_path> <output_path>")
        sys.exit(1)

    dataset1_path = sys.argv[1].rstrip("/")
    dataset2_path = sys.argv[2].rstrip("/")
    output_path = sys.argv[3]

    start_time = time.time()

    train_dir_subset = create_subset(dataset1_path, 1000)
    generated_dir_subset = create_subset(dataset2_path, 1000)

    evaluate_mgeval_combined(train_dir_subset, generated_dir_subset, output_path)
    print("\nFMD:", fmd(train_dir_subset, generated_dir_subset))

    end_time = time.time()
    mins, secs = divmod(end_time - start_time, 60)
    print(f"\nTotal execution time: {int(mins)} min {secs:.2f} sec")
    