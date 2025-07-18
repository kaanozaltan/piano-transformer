import os
import shutil
import glob
import random
from pathlib import Path
import argparse


def create_basic_subset(input_paths, output_dir, subset_size, seed):
    output_dir.mkdir(parents=True, exist_ok=True)
    assert len(input_paths) >= subset_size, f"Not enough files in {output_dir}"
    subset = random.Random(seed).sample(sorted(input_paths), subset_size)

    for src in subset:
        shutil.copy2(src, output_dir / os.path.basename(src))


def create_matched_cont_test_subsets(cont_dir, output_base, subset_size, seed):
    prompts = glob.glob(str(cont_dir / "*_prompt.midi"))
    generations = glob.glob(str(cont_dir / "*_generated.midi"))

    prompt_ids = {Path(p).stem.split("_")[0] for p in prompts}
    gen_ids = {Path(g).stem.split("_")[0] for g in generations}
    common_ids = sorted(prompt_ids & gen_ids)

    assert len(common_ids) >= subset_size, f"Not enough matching prompt-generation pairs: {len(common_ids)}"
    chosen_ids = random.Random(seed).sample(common_ids, subset_size)

    test_dir = output_base / "test"
    cont_out_dir = output_base / "continuations"
    test_dir.mkdir(parents=True, exist_ok=True)
    cont_out_dir.mkdir(parents=True, exist_ok=True)

    for id_ in chosen_ids:
        prompt = next(p for p in prompts if Path(p).stem.startswith(id_ + "_"))
        gen = next(g for g in generations if Path(g).stem.startswith(id_ + "_"))
        shutil.copy2(prompt, test_dir / os.path.basename(prompt))
        shutil.copy2(gen, cont_out_dir / os.path.basename(gen))


def run_eval_setup(model_name, base_dir="/hpcwork/lect0148/experiments", subset_size=1000):
    model_path = Path(base_dir) / model_name
    assert model_path.exists(), f"Model path not found: {model_path}"

    subset_base = model_path / "output" / "subset"
    subset_base.mkdir(parents=True, exist_ok=True)

    # 1. Subset: train
    train_files = glob.glob(str(model_path / "data_processed" / "maestro_train" / "**" / "*.midi"), recursive=True)
    create_basic_subset(train_files, subset_base / "train", subset_size, seed=123)

    # 2. Subset: unconditional generations
    gen_files = glob.glob(str(model_path / "output" / "generations" / "*.midi"))
    create_basic_subset(gen_files, subset_base / "generations", subset_size, seed=None)

    # 3–4. Subsets: matched prompt/continuation pairs
    cont_dir = model_path / "output" / "continuations"
    create_matched_cont_test_subsets(cont_dir, subset_base, subset_size, seed=789)

    print(f"Subsets created at: {subset_base}")


if __name__ == "__main__":
    # Example: python scripts/eval_setup.py mistral-950M_remi_maestro
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", help="Model directory name under experiments/")
    args = parser.parse_args()

    run_eval_setup(args.experiment_name)