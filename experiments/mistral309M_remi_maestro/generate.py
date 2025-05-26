import time
from copy import deepcopy
from pathlib import Path

from miditok import REMI
from miditok.pytorch_data import DataCollator, DatasetMIDI
from torch import ones_like, tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GenerationConfig

TRANSFORMER_NAME = "mistral-309M"
TOKENIZER_NAME = "remi"
DATASET_NAME = "maestro"
MODEL_VERSION = "1"

MODEL_NAME = f"{TRANSFORMER_NAME}_{TOKENIZER_NAME}_{DATASET_NAME}_v{MODEL_VERSION}"

print(f"Model:\n{MODEL_NAME}")

BASE_PATH = Path(".")
# BASE_PATH = Path("/hpcwork/lect0148")

DATA_RAW_PATH = BASE_PATH / "data"

MODEL_BASE_PATH = BASE_PATH / "models" / MODEL_NAME
DATA_PROCESSED_PATH = MODEL_BASE_PATH / "data_processed"
MODEL_PATH = MODEL_BASE_PATH / "model"
RUNS_PATH = MODEL_BASE_PATH / "runs"
OUTPUT_PATH = MODEL_BASE_PATH / "output"

tokenizer = REMI(params=MODEL_BASE_PATH / "tokenizer.json")

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

start = time.time()
print("[INFO] Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
print(f"[INFO] Model loaded in {time.time() - start:.2f} seconds.", flush=True)

generation_config = GenerationConfig(
    max_new_tokens=2048,  # extends samples by 200 tokens
    num_beams=1,
    do_sample=True,
    temperature=1.0,
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


# def generate(dataset, output, max_samples=None):
#     (output_path := Path(output)).mkdir(parents=True, exist_ok=True)
#     dataloader = DataLoader(dataset, batch_size=16, collate_fn=collator)

#     count = 0
#     for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating outputs")):
#         print(
#             f"[INFO] Processing batch {batch_idx} (generated so far: {count}/{max_samples})"
#         )
#         res = model.generate(
#             inputs=batch["input_ids"].to(model.device),
#             attention_mask=batch["attention_mask"].to(model.device),
#             generation_config=generation_config,
#         )

#         # Saves the generated music, as MIDI files and tokens (json)
#         for prompt, continuation in zip(batch["input_ids"], res):
#             generated = continuation[len(prompt) :]
#             tokens = [generated, prompt, continuation]
#             tokens = [seq.tolist() for seq in tokens]

#             midi_generated = tokenizer.decode([deepcopy(tokens[0])])
#             midi_prompt = tokenizer.decode([deepcopy(tokens[1])])
#             midi_full = tokenizer.decode([deepcopy(tokens[2])])

#             # Name the tracks
#             if midi_generated.tracks:
#                 midi_generated.tracks[0].name = (
#                     f"Generated continuation ({len(tokens[0])} tokens)"
#                 )
#             if midi_prompt.tracks:
#                 midi_prompt.tracks[0].name = (
#                     f"Original prompt ({len(tokens[1])} tokens)"
#                 )
#             if midi_full.tracks:
#                 midi_full.tracks[0].name = f"Full sequence ({len(tokens[2])} tokens)"

#             # Save each as a separate MIDI file
#             midi_generated.dump_midi(output_path / f"{count}_generated.midi")
#             midi_prompt.dump_midi(output_path / f"{count}_prompt.midi")
#             midi_full.dump_midi(output_path / f"{count}_full.midi")
#             tokenizer.save_tokens(tokens, output_path / f"{count}.json")

#             count += 1

#             if max_samples and count >= max_samples:
#                 break
#         if max_samples is not None and count >= max_samples:
#             break


def generate_from_scratch(num_samples: int, output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)
    model.eval()

    input_ids = tensor([[tokenizer["BOS_None"]]] * num_samples).to(model.device)
    attention_mask = ones_like(input_ids)

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
    )

    for i, sequence in enumerate(generated):
        print(f"[INFO] Generated sequence {i + 1}/{num_samples}")
        tokens = sequence.tolist()

        midi = tokenizer.decode([deepcopy(tokens)])
        if midi.tracks:
            midi.tracks[0].name = f"Unconditioned ({len(tokens)} tokens)"

        midi.dump_midi(output_path / f"{i}_scratch.midi")
        tokenizer.save_tokens([tokens], output_path / f"{i}_scratch.json")


# generate(dataset_test, OUTPUT_PATH / "test", 20)

generate_from_scratch(20, OUTPUT_PATH / "scratch")
