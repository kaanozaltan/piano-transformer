import time
from piano_transformer.utils.metrics import (
    create_subset,
    evaluate_mgeval_combined,
    fmd,
)

TRAIN_DIR = "experiments/mistral-162M_remi_maestro_v1/output/train/"
GENERATED_DIR = "experiments/mistral-162M_remi_maestro_v1/output/generated/"
GRAPHICS_DIR = "experiments/mistral-162M_remi_maestro_v1/graphics_subset_1000/"


if __name__ == "__main__":
    start_time = time.time()

    create_subset(TRAIN_DIR, 1000)
    create_subset(GENERATED_DIR, 1000)

    train_dir_subset = TRAIN_DIR.rstrip("/") + "_subset"
    generated_dir_subset = GENERATED_DIR.rstrip("/") + "_subset"

    evaluate_mgeval_combined(train_dir_subset, generated_dir_subset, GRAPHICS_DIR)
    print("\nFMD:", fmd(train_dir_subset, generated_dir_subset))

    end_time = time.time()
    elapsed = end_time - start_time
    mins, secs = divmod(elapsed, 60)
    print(f"\nTotal execution time: {int(mins)} min {secs:.2f} sec")
