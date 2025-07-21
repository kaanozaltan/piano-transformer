import os
import numpy as np

from dotenv import load_dotenv
import wandb
from pathlib import Path
from transformers.trainer_utils import set_seed

from piano_transformer.config import load_config
from piano_transformer.datasets.dataset import build_collator, build_datasets
from piano_transformer.datasets.preprocessing import split_datasets_into_chunks
from piano_transformer.model import build_mistral_model
from piano_transformer.tokenizer import create_remi_tokenizer
from piano_transformer.trainer import make_trainer
from piano_transformer.utils.midi import (
    get_midi_file_lists_by_random,
    get_midi_file_lists_by_csv,
)
from piano_transformer.utils.evaluation import EvalCallback

## SETUP

cfg = load_config(Path(__file__).resolve().parent / "config.yaml")

print(f"Model:\n{cfg.model_name}")

load_dotenv()
os.environ["WANDB_ENTITY"] = "jonathanlehmkuhl-rwth-aachen-university"
os.environ["WANDB_PROJECT"] = "piano-transformer"
wandb.login()

set_seed(cfg.seed)

## DATASET PREPARATION

midi_lists_maestro = get_midi_file_lists_by_csv(
    cfg.data_raw_path / "maestro" / "maestro-v3.0.0.csv", cfg.data_raw_path / "maestro"
)

midi_lists = get_midi_file_lists_by_random(
    cfg.data_raw_path / "aria-midi-genre-balanced", "*.mid", cfg.seed
)

rng = np.random.default_rng(cfg.seed)

# Use 10.000 files from aria for moderate-scale pre-training
for split in ["train", "validation", "test"]:
    if split == "train":
        midi_lists[split] = rng.permutation(midi_lists[split]).tolist()[:100000]
    else:
        midi_lists[split] = rng.permutation(midi_lists[split]).tolist()[:100000]
    print(f"Number of {split} files: {len(midi_lists[split])}")

# TOKENIZATION
# Use 5.000 files from aria and replicate maestro 5 times for tokenization
maestro_tokenization_set = rng.permutation(midi_lists_maestro["train"]).tolist()
aria_tokenization_set = rng.permutation(midi_lists["train"]).tolist()[:5000]
tokenization_set = rng.permutation(
    maestro_tokenization_set * 5 + aria_tokenization_set
).tolist()
tokenizer = create_remi_tokenizer(
    tokenization_set, cfg.experiment_path / "tokenizer.json", genre=True
)

MAX_SEQ_LEN = 1024
NUM_OVERLAP_BARS = 10

chunks_lists = split_datasets_into_chunks(
    midi_lists,
    tokenizer,
    cfg.data_processed_path,
    "aria",
    MAX_SEQ_LEN,
    NUM_OVERLAP_BARS,
)

augmentation_cfg = {
    "pitch_offsets": list(range(-6, 6)),
    "velocity_offsets": list(range(-20, 21)),
    "duration_offsets": [-0.5, -0.375, -0.25, -0.125, 0, 0.125, 0.25, 0.375, 0.5],
    "tempo_factors": [0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1],
}


train_ds, valid_ds, test_ds = build_datasets(
    chunks_lists, tokenizer, MAX_SEQ_LEN, augmentation_cfg
)
collator = build_collator(tokenizer)

## TRAINING

model_cfg = {
    "num_hidden_layers": 12,
    "hidden_size": 768,
}

model = build_mistral_model(model_cfg, tokenizer, MAX_SEQ_LEN)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(
    f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)

trainer_cfg = {
    "output_dir": cfg.runs_path,
    "gradient_accumulation_steps": 2,
    "per_device_train_batch_size": 64,
    "per_device_eval_batch_size": 64,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "max_grad_norm": 3.0,
    "lr_scheduler_type": "cosine_with_min_lr",
    "min_lr_rate": 0.1,
    "warmup_ratio": 0.03,
    "logging_steps": 20,
    "eval_steps": 68,
    "save_steps": 1020,
    "num_train_epochs": 150,
    "seed": cfg.seed,
    "data_seed": cfg.seed,
    "run_name": cfg.model_name,
    "optim": "adamw_torch",
    "max_steps": 40800,
}

trainer = make_trainer(trainer_cfg, model, collator, train_ds, valid_ds)

val_callback = EvalCallback(
    tokenizer=tokenizer,
    ref_dir=cfg.data_processed_path / "aria_train",
    gen_dir=cfg.experiment_path / "output" / "validation",
    num_samples=200,
    every_n_steps=2040,
)
trainer.add_callback(val_callback)

result = trainer.train()
trainer.save_model(cfg.model_path)
trainer.log_metrics("train", result.metrics)
trainer.save_metrics("train", result.metrics)
trainer.save_state()
