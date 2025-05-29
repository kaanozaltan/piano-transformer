import time
from copy import deepcopy
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GenerationConfig

from piano_transformer.config import load_config
from piano_transformer.datasets.dataset import build_datasets, build_collator
from piano_transformer.datasets.preprocessing import split_datasets_into_chunks
from piano_transformer.tokenization.tokenizer import load_remi_tokenizer
from piano_transformer.utils.midi import get_midi_file_lists, convert_midi_to_wav

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

_, _, test_ds = build_datasets(
    chunks_lists, tokenizer, MAX_SEQ_LEN, augmentation_cfg
)
collator = build_collator(tokenizer)


start = time.time()
print("[INFO] Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(cfg.model_path)
print(f"[INFO] Model loaded in {time.time() - start:.2f} seconds.", flush=True)

generation_config = GenerationConfig(
    max_new_tokens=200,
    num_beams=1,
    do_sample=True,
    temperature=0.9,
    top_k=15,
    top_p=0.95,
    epsilon_cutoff=3e-4,
    eta_cutoff=1e-3,
    pad_token_id=tokenizer.pad_token_id,
)

# Here the sequences are padded to the left, so that the last token along the time dimension
# is always the last token of each seq, allowing to efficiently generate by batch
collator.pad_on_left = True
collator.eos_token = None

model.eval()


def generate(dataset, output, max_samples=None):
    (output_path := Path(output)).mkdir(parents=True, exist_ok=True)
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collator)

    count = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating outputs")):
        print(f"[INFO] Processing batch {batch_idx} (generated so far: {count}/{max_samples})")
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
        if max_samples is not None and count >= max_samples:
            break


#generate(test_ds, cfg.output_path / "test", 20)
convert_midi_to_wav(cfg.output_path / "test", cfg.output_path / "test_wav", "SalC5Light2.sf2")
