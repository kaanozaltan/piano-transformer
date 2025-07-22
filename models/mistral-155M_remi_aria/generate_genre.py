from copy import deepcopy
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import GenerationConfig

from piano_transformer.config import load_config
from piano_transformer.model import load_model
from piano_transformer.tokenizer import load_remi_tokenizer

## SETUP

cfg = load_config(Path(__file__).resolve().parent / "config.yaml")

print(f"Model:\n{cfg.model_name}")

tokenizer = load_remi_tokenizer(cfg.experiment_path / "tokenizer.json")

MAX_SEQ_LEN = 1024
NUM_OVERLAP_BARS = 10

model = load_model(cfg.runs_path / "checkpoint-40800")

generation_config = GenerationConfig(
    max_new_tokens=1024,
    do_sample=True,
    temperature=1.0,
    pad_token_id=tokenizer.pad_token_id,
)

# Here the sequences are padded to the left, so that the last token along the time dimension
# is always the last token of each seq, allowing to efficiently generate by batch
# collator.pad_on_left = True
# collator.eos_token = None

model.eval()

BATCH_SIZE = 64


# possible genres: "ambient", "atonal", "blues", "classical", "folk", "jazz", "pop", "ragtime", "rock", "soundtrack"
def get_genre_token_id(genre: str) -> int:
    genre_token = f"GENRE_{genre.upper()}"
    return tokenizer.vocab[genre_token]


def generate_from_scratch(output, num_samples):
    (output_path := Path(output)).mkdir(parents=True, exist_ok=True)

    count = 0
    for batch_start in tqdm(
        range(0, num_samples, BATCH_SIZE), desc="Generating from scratch"
    ):
        current_batch_size = min(BATCH_SIZE, num_samples - count)
        # Create empty generation inputs with just the BOS token
        BOS_TOKEN_ID = 1
        input_ids = torch.full(
            (current_batch_size, 1), BOS_TOKEN_ID, dtype=torch.long, device=model.device
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


def generate_conditioned_on_genre(output, num_samples, genre: str):
    (output_path := Path(output)).mkdir(parents=True, exist_ok=True)

    genre_token_id = get_genre_token_id(genre)
    if genre_token_id is None:
        raise ValueError(
            f"Genre token for '{genre}' not found in tokenizer vocabulary."
        )

    BOS_TOKEN_ID = 1
    count = 0

    for batch_start in tqdm(
        range(0, num_samples, BATCH_SIZE), desc=f"Generating for genre {genre}"
    ):
        current_batch_size = min(BATCH_SIZE, num_samples - count)

        # Create input with genre token + BOS token
        input_ids = torch.tensor(
            [[genre_token_id, BOS_TOKEN_ID]] * current_batch_size,
            dtype=torch.long,
            device=model.device,
        )

        # Generate sequences
        res = model.generate(
            inputs=input_ids,
            generation_config=generation_config,
        )

        for generated in res:
            tokens = generated.tolist()

            midi_generated = tokenizer.decode([deepcopy(tokens)])
            if midi_generated.tracks:
                midi_generated.tracks[0].name = (
                    f"Generated {genre.capitalize()} ({len(tokens)} tokens)"
                )

            midi_generated.dump_midi(output_path / f"{count}_generated.midi")
            tokenizer.save_tokens([tokens], output_path / f"{count}.json")

            count += 1


for genre in [
    "ambient",
    "atonal",
    "blues",
    "classical",
    "folk",
    "jazz",
    "pop",
    "ragtime",
    "rock",
    "soundtrack",
]:
    print(f"Generating {genre} music...")
    generate_conditioned_on_genre(cfg.output_path / genre, 30, genre)
