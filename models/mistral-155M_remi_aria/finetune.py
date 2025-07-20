import os
from pathlib import Path

from dotenv import load_dotenv
from transformers.trainer_utils import set_seed

import wandb
from piano_transformer.config import load_config
from piano_transformer.datasets.dataset import build_collator, build_datasets
from piano_transformer.datasets.preprocessing import split_datasets_into_chunks
from piano_transformer.model import load_model
from piano_transformer.tokenizer import load_remi_tokenizer
from piano_transformer.trainer import make_trainer
from piano_transformer.utils.evaluation import EvalCallback
from piano_transformer.utils.midi import get_midi_file_lists_by_csv

## SETUP

cfg = load_config(Path(__file__).resolve().parent / "config.yaml")

print(f"Model:\n{cfg.model_name}")

load_dotenv()
os.environ["WANDB_ENTITY"] = "jonathanlehmkuhl-rwth-aachen-university"
os.environ["WANDB_PROJECT"] = "piano-transformer"
wandb.login()

set_seed(cfg.seed)

## DATASET PREPARATION

midi_lists = get_midi_file_lists_by_csv(
    cfg.data_raw_path / "maestro" / "maestro-v3.0.0.csv", cfg.data_raw_path / "maestro"
)

tokenizer = load_remi_tokenizer(cfg.experiment_path / "tokenizer.json")

MAX_SEQ_LEN = 1024
NUM_OVERLAP_BARS = 10

chunks_lists = split_datasets_into_chunks(
    midi_lists,
    tokenizer,
    cfg.data_processed_path,
    "maestro",
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

model = load_model(cfg.model_path)

# TODO: Maybe freezing?
N_FREEZE = 8
FREEZE_EMBEDDINGS = False  # freeze if using only classical music
KEEP_NORMS_TRAINABLE = True

def freeze_layers(model, n_freeze, freeze_embeddings=True, keep_norms_trainable=True):
    if freeze_embeddings:
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            for p in model.model.embed_tokens.parameters():
                p.requires_grad = False
    layers = model.model.layers
    for i, layer in enumerate(layers):
        for name, p in layer.named_parameters():
            if i < n_freeze:
                if keep_norms_trainable and ("norm" in name.lower() or "ln" in name.lower()):
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            else:
                p.requires_grad = True


# freeze_layers(model, N_FREEZE, FREEZE_EMBEDDINGS, KEEP_NORMS_TRAINABLE)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

trainer_cfg = {
    "output_dir": cfg.runs_path.parent / f"{cfg.runs_path}_finetune_no-freezing",
    "gradient_accumulation_steps": 2,
    "per_device_train_batch_size": 64,
    "per_device_eval_batch_size": 64,
    "learning_rate": 2e-5,
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
    "run_name": cfg.model_name + "_finetune_no-freezing",
    "optim": "adamw_torch",
    "max_steps": 10200,
}

print(f"Trainer config: {trainer_cfg}")

trainer = make_trainer(trainer_cfg, model, collator, train_ds, valid_ds)

val_callback = EvalCallback(
    tokenizer=tokenizer,
    ref_dir=cfg.data_processed_path / "maestro_train",
    gen_dir=cfg.experiment_path / "output" / "validation_no-freezing",
    num_samples=200,
    every_n_steps=2040,
)
trainer.add_callback(val_callback)

result = trainer.train()
trainer.save_model(cfg.model_path.parent / f"{cfg.model_path}_finetune_no-freezing")
trainer.log_metrics("train", result.metrics)
trainer.save_metrics("train", result.metrics)
trainer.save_state()
