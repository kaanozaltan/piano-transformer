import os
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import wandb
from evaluate import load as load_metric
from miditok import REMI, TokenizerConfig
from miditok.data_augmentation import augment_dataset
from miditok.pytorch_data import DataCollator, DatasetMIDI
from miditok.utils import split_files_for_training
from sklearn.model_selection import train_test_split
from torch import argmax
from torch.cuda import is_available as cuda_available
from torch.cuda import is_bf16_supported
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    MistralConfig,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed

TRANSFORMER_NAME = "mistral-309M"
TOKENIZER_NAME = "remi"
DATASET_NAME = "maestro"
MODEL_VERSION = "1"

MODEL_NAME = f"{TRANSFORMER_NAME}_{TOKENIZER_NAME}_{DATASET_NAME}_v{MODEL_VERSION}"

print(f"Model:\n{MODEL_NAME}")

BASE_PATH = Path("/hpcwork/gd010186/piano-transformer")

DATA_RAW_PATH = BASE_PATH / "data"

MODEL_BASE_PATH = BASE_PATH / "models" / MODEL_NAME
DATA_PROCESSED_PATH = MODEL_BASE_PATH / "data_processed"
MODEL_PATH = MODEL_BASE_PATH / "model"
RUNS_PATH = MODEL_BASE_PATH / "runs"
OUTPUT_PATH = MODEL_BASE_PATH / "output"

os.environ["WANDB_ENTITY"] = "jonathanlehmkuhl-rwth-aachen-university"
os.environ["WANDB_PROJECT"] = "piano-transformer"
wandb.login()

SEED = 222
set_seed(SEED)

maestro_files = list((DATA_RAW_PATH / "maestro").resolve().glob("**/*.midi"))

tokenizer_config = TokenizerConfig(
    pitch_range=(21, 109),
    beat_res={(0, 1): 12, (1, 4): 8, (4, 12): 4},
    special_tokens=["PAD", "BOS", "EOS"],
    use_chords=True,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=True,
    use_sustain_pedals=True,
    num_velocities=16,
    num_tempos=32,
    tempo_range=(50, 200),
)
tokenizer = REMI(tokenizer_config)

# TODO: train/test split already here?
tokenizer.train(vocab_size=30000, files_paths=maestro_files)
tokenizer.save(MODEL_BASE_PATH / "tokenizer.json")

tokenizer = REMI(params=MODEL_BASE_PATH / "tokenizer.json")

# Split data into train/validation/test datasets
midi_files_train, midi_files_temp = train_test_split(
    maestro_files, test_size=0.3, random_state=SEED
)
midi_files_valid, midi_files_test = train_test_split(
    midi_files_temp, test_size=0.5, random_state=SEED
)

# Split MIDIs into smaller chunks that approximately matches the token sequence length for training
for midi_files, subset_name in (
    (midi_files_train, "train"),
    (midi_files_valid, "valid"),
    (midi_files_test, "test"),
):
    subset_chunks_dir = Path(DATA_PROCESSED_PATH / f"maestro_{subset_name}")
    split_files_for_training(
        files_paths=midi_files,
        tokenizer=tokenizer,
        save_dir=subset_chunks_dir,
        max_seq_len=2048,
        num_overlap_bars=2,
    )

# Augment training data set
augment_dataset(
    Path(DATA_PROCESSED_PATH / "maestro_train"),
    pitch_offsets=[-12, 12],
    velocity_offsets=[-4, 4],
    duration_offsets=[-0.5, 0.5],
)

# Create pytorch datasets
midi_files_train = list((DATA_PROCESSED_PATH / "maestro_train").glob("**/*.midi"))
midi_files_valid = list((DATA_PROCESSED_PATH / "maestro_valid").glob("**/*.midi"))
midi_files_test = list((DATA_PROCESSED_PATH / "maestro_train").glob("**/*.midi"))

dataset_kwargs = {
    "max_seq_len": 2048,
    "tokenizer": tokenizer,
    "bos_token_id": tokenizer["BOS_None"],
    "eos_token_id": tokenizer["EOS_None"],
}
dataset_train = DatasetMIDI(midi_files_train, **dataset_kwargs)
dataset_valid = DatasetMIDI(midi_files_valid, **dataset_kwargs)
dataset_test = DatasetMIDI(midi_files_test, **dataset_kwargs)

collator = DataCollator(tokenizer["PAD_None"], copy_inputs_as_labels=True)

print(len(midi_files_train))
