import time
from copy import deepcopy
from pathlib import Path

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GenerationConfig

from piano_transformer.config import load_config
from piano_transformer.datasets.dataset import build_datasets, build_collator
from piano_transformer.datasets.preprocessing import split_datasets_into_chunks
from piano_transformer.tokenizer import load_remi_tokenizer
from piano_transformer.utils.midi import get_midi_file_lists, midi2wav

## SETUP

cfg = load_config(Path(__file__).resolve().parent / "config.yaml")

print(f"Model:\n{cfg.model_name}")

## DATASET PREPARATION
# TODO: most of this is not needed for generate, but right now included for simplicity

midi_lists = get_midi_file_lists(
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

train_ds, _, _ = build_datasets(
    chunks_lists, tokenizer, MAX_SEQ_LEN, augmentation_cfg
)

start = time.time()
print("[INFO] Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(cfg.model_path)
model.to("cuda")
print(f"[INFO] Model loaded in {time.time() - start:.2f} seconds.", flush=True)

generation_config = GenerationConfig(
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.9,
    top_k=50,
    pad_token_id=tokenizer.pad_token_id,
)

# Here the sequences are padded to the left, so that the last token along the time dimension
# is always the last token of each seq, allowing to efficiently generate by batch

model.eval()

BATCH_SIZE = 256


def generate_from_scratch(output, num_samples):
    (output_path := Path(output)).mkdir(parents=True, exist_ok=True)

    count = 0
    for batch_start in tqdm(range(0, num_samples, BATCH_SIZE), desc="Generating from scratch"):
        current_batch_size = min(BATCH_SIZE, num_samples - count)
        # Create empty generation inputs with just the BOS token
        BOS_TOKEN_ID = 1
        input_ids = torch.full(
            (current_batch_size, 1),
            BOS_TOKEN_ID,
            dtype=torch.long,
            device=model.device
        )
        
        # Generate sequence from scratch
        res = model.generate(
            inputs=input_ids,
            generation_config=generation_config,
        )
        for generated in res:
            tokens = generated.tolist()
            
            midi_generated = tokenizer.decode([deepcopy(tokens)])
            
            if midi_generated.tracks:
                midi_generated.tracks[0].name = (
                    f"Generated from scratch ({len(tokens)} tokens)"
                )

            midi_generated.dump_midi(output_path / f"{count}_generated.midi")
            tokenizer.save_tokens([tokens], output_path / f"{count}.json")
            
            count += 1

generate_from_scratch(cfg.output_path / "test_jonathan_from_scratch", len(train_ds))
