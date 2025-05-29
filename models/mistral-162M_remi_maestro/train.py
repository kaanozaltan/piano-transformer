import os

from dotenv import load_dotenv
import wandb
from transformers.trainer_utils import set_seed

from piano_transformer.config import load_config
from piano_transformer.datasets.dataset import build_collator, build_datasets
from piano_transformer.datasets.preprocessing import split_datasets_into_chunks
from piano_transformer.model.mistral_model import build_mistral_model
from piano_transformer.tokenization.tokenizer import create_remi_tokenizer
from piano_transformer.training.trainer import make_trainer
from piano_transformer.utils.midi import get_midi_file_lists

## SETUP

cfg = load_config("config.yaml")

print(f"Model:\n{cfg.model_name}")

load_dotenv()
os.environ["WANDB_ENTITY"] = "jonathanlehmkuhl-rwth-aachen-university"
os.environ["WANDB_PROJECT"] = "piano-transformer"
wandb.login()

set_seed(cfg.seed)

## DATASET PREPARATION

midi_lists = get_midi_file_lists(
    cfg.data_raw_path / "maestro" / "maestro-v3.0.0.csv", cfg.data_raw_path / "maestro"
)

for split in ["train", "validation", "test"]:
    print(f"Number of {split} files: {len(midi_lists[split])}")

tokenizer = create_remi_tokenizer(
    midi_lists["train"], cfg.experiment_path / "tokenizer.json"
)

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

model_cfg = {
    "num_hidden_layers": 18,
    "hidden_size": 512,
    "intermediate_size": 512 * 8,
    "num_attention_heads": 8,
    "attention_dropout": 0.1,
}

model = build_mistral_model(model_cfg, tokenizer, MAX_SEQ_LEN)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(
    f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)

trainer_cfg = {
    "output_dir": cfg.runs_path,
    "per_device_train_batch_size": 96,
    "per_device_eval_batch_size": 96,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "max_grad_norm": 3.0,
    "lr_scheduler_type": "cosine_with_min_lr",
    "min_lr_rate": 0.1,
    "warmup_ratio": 0.03,
    "logging_steps": 20,
    "num_train_epochs": 150,
    "seed": cfg.seed,
    "data_seed": cfg.seed,
    "run_name": cfg.model_name,
    "optim": "adamw_torch",
    "early_stopping_patience": 10,
}

trainer = make_trainer(trainer_cfg, model, collator, train_ds, valid_ds)

result = trainer.train()
trainer.save_model(cfg.model_path)
trainer.log_metrics("train", result.metrics)
trainer.save_metrics("train", result.metrics)
trainer.save_state()
