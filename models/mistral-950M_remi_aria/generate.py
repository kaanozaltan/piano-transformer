import time
from copy import deepcopy
from pathlib import Path
import math

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, GenerationConfig

from piano_transformer.config import load_config
from piano_transformer.datasets.dataset import build_datasets, build_collator
from piano_transformer.datasets.preprocessing import split_datasets_into_chunks
from piano_transformer.tokenizer import load_remi_tokenizer
from piano_transformer.utils.midi import get_midi_file_lists_by_random, midi2wav

## SETUP

cfg = load_config(Path(__file__).resolve().parent / "config.yaml")

print(f"Model:\n{cfg.model_name}")

## DATASET PREPARATION
# TODO: most of this is not needed for generate, but right now included for simplicity

midi_lists = get_midi_file_lists_by_random(
    cfg.data_raw_path / "aria-midi-v1-deduped-ext", "*.mid", cfg.seed
)

tokenizer = load_remi_tokenizer(cfg.experiment_path / "tokenizer.json")

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

train_ds, _, test_ds = build_datasets(
    chunks_lists, tokenizer, MAX_SEQ_LEN, augmentation_cfg
)
collator = build_collator(tokenizer)


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
collator.pad_on_left = True
collator.eos_token = None

model.eval()

BATCH_SIZE = 256


def generate(dataset, output, max_samples=None):
    (output_path := Path(output)).mkdir(parents=True, exist_ok=True)
    dataloader = DataLoader(dataset, BATCH_SIZE, collate_fn=collator)

    count = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating outputs")):
        print(
            f"[INFO] Processing batch {batch_idx} (generated so far: {count}/{max_samples})"
        )
        print(f"[INFO] Batch size: {BATCH_SIZE}")
        res = model.generate(
            inputs=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            generation_config=generation_config,
        )
        # Saves the generated music, as MIDI files and tokens (json)
        for prompt, continuation in zip(batch["input_ids"], res):
            generated = continuation[len(prompt) :]
            tokens = [generated, prompt, continuation]
            tokens = [seq.tolist() for seq in tokens]

            midi_generated = tokenizer.decode([deepcopy(tokens[0])])
            midi_prompt = tokenizer.decode([deepcopy(tokens[1])])
            midi_full = tokenizer.decode([deepcopy(tokens[2])])

            # Name the tracks
            if midi_generated.tracks:
                midi_generated.tracks[0].name = (
                    f"Generated continuation ({len(tokens[0])} tokens)"
                )
            if midi_prompt.tracks:
                midi_prompt.tracks[0].name = (
                    f"Original prompt ({len(tokens[1])} tokens)"
                )
            if midi_full.tracks:
                midi_full.tracks[0].name = f"Full sequence ({len(tokens[2])} tokens)"

            # Save each as a separate MIDI file
            midi_generated.dump_midi(output_path / f"{count}_generated.midi")
            midi_prompt.dump_midi(output_path / f"{count}_prompt.midi")
            midi_full.dump_midi(output_path / f"{count}_full.midi")
            tokenizer.save_tokens(tokens, output_path / f"{count}.json")

            count += 1

            if max_samples and count >= max_samples:
                break
        if max_samples and count >= max_samples:
            break


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


def compute_perplexity(model, dataset, collator, batch_size=32):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating perplexity"):
            input_ids = batch["input_ids"].to(model.device)  # [B, T]
            attention_mask = batch["attention_mask"].to(model.device)

            # Shift inputs and targets for teacher forcing
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            mask = attention_mask[:, 1:]

            outputs = model(inputs)
            logits = outputs.logits  # [B, T-1, vocab_size]

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            )

            # Mask out the padding positions
            loss = loss * mask.reshape(-1)
            total_loss += loss.sum().item()
            total_tokens += mask.sum().item()

    avg_nll = total_loss / total_tokens
    perplexity = math.exp(avg_nll)
    return perplexity


# generate(test_ds, cfg.output_path / "test_jonathan_2")

generate_from_scratch(cfg.output_path / "generations", 1000)

# perplexity = compute_perplexity(model, test_ds, collator, batch_size=128)
# print(f"Perplexity on test set: {perplexity:.2f}")

# midi2wav(cfg.output_path / "test_jonathan", cfg.output_path / "test_jonathan_wav", "SalC5Light2.sf2")
