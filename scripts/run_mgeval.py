from piano_transformer.utils.metrics import (
    analyze_dataset_mgeval, 
    comparing_pairwise_distances_mgeval,
    fmd,
)

TRAIN_DIR = "experiments/mistral-162M_remi_maestro_v1/output/train/"
GENERATED_DIR = "experiments/mistral-162M_remi_maestro_v1/output/generated/"
GRAPHICS_DIR = "experiments/mistral-162M_remi_maestro_v1/graphics_162M/"


def full_evaluation_pipeline(train_dir, generated_dir, graphics_dir, max_samples=None):
    print("Absolute measurements:\n")

    print("Training set:")
    analyze_dataset_mgeval(train_dir, max_samples=max_samples)

    print("\nGenerated set:")
    analyze_dataset_mgeval(generated_dir, max_samples=max_samples)

    print("\nRelative measurements (MGEval):\n")
    comparing_pairwise_distances_mgeval(
        train_dir, 
        generated_dir, 
        graphics_dir,
        max_samples=max_samples
    )

    print("\nFrechet Music Distance (FMD):\n")
    print(fmd(train_dir, generated_dir))


if __name__ == "__main__":
    full_evaluation_pipeline(TRAIN_DIR, GENERATED_DIR, GRAPHICS_DIR)
