from piano_transformer.utils.metrics import (
    evaluate_mgeval_combined,
    comparing_pairwise_distances_mgeval,
    fmd,
)

TRAIN_DIR = "experiments/mistral-162M_remi_maestro_v1/output/train/"
GENERATED_DIR = "experiments/mistral-162M_remi_maestro_v1/output/generated/"
GRAPHICS_DIR = "experiments/mistral-162M_remi_maestro_v1/graphics_new/"


if __name__ == "__main__":
    evaluate_mgeval_combined(TRAIN_DIR, GENERATED_DIR, GRAPHICS_DIR)
    # comparing_pairwise_distances_mgeval(TRAIN_DIR, GENERATED_DIR, GRAPHICS_DIR)
    print("\nFMD:", fmd(TRAIN_DIR, GENERATED_DIR))
