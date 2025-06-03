from pathlib import Path

import torch
from torch.utils.data import DataLoader
import time

from transformers import set_seed

from piano_transformer.config import load_config
from piano_transformer.datasets.dataset import build_collator, build_datasets
from piano_transformer.datasets.preprocessing import split_datasets_into_chunks
from piano_transformer.tokenization.tokenizer import create_remi_tokenizer
from piano_transformer.utils.midi import get_midi_file_lists

cfg = load_config(Path(__file__).resolve().parent / "config.yaml")

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


## Profile data loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

loader = DataLoader(valid_ds, batch_size=128, num_workers=0, pin_memory=False, collate_fn=collator)

start = time.time()
for epoch in range(3):
    for i, batch in enumerate(loader):
        print(i)
        # Send data to GPU (simulate training input pipeline)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

print(f"Loaded in {time.time() - start:.2f} seconds")
