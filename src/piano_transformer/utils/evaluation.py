from piano_transformer.utils import metrics

from pathlib import Path
from tqdm import tqdm
import torch
import wandb
import json
from copy import deepcopy


def generate_and_save(model, tokenizer, dataset, save_dir, mode="eval"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for i, batch in enumerate(tqdm(loader, desc="Generating")):
        input_ids = batch["input_ids"].cuda()  # shape: (1, seq_len)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                max_length=2048,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=1.0,
                pad_token_id=tokenizer["PAD_None"],
                bos_token_id=tokenizer["BOS_None"],
                eos_token_id=tokenizer["EOS_None"],
            )

        prompt = input_ids[0]
        generated = generated_ids[0][len(prompt):]
        full = generated_ids[0]

        if mode == "eval":
            midi_generated = tokenizer.decode([deepcopy(generated.tolist())])
            midi_generated.dump_midi(save_dir / f"{i}.midi")
        elif mode == "all":
            midi_prompt = tokenizer.decode([deepcopy(prompt.tolist())])
            midi_generated = tokenizer.decode([deepcopy(generated.tolist())])
            midi_full = tokenizer.decode([deepcopy(full.tolist())])

            midi_prompt.dump_midi(save_dir / f"{i}_prompt.midi")
            midi_generated.dump_midi(save_dir / f"{i}_generated.midi")
            midi_full.dump_midi(save_dir / f"{i}_full.midi")
        else:
            raise ValueError(f"Unknown mode: {mode}")


def evaluate(ref_dir, gen_dir):
    ref_dir = Path(ref_dir)
    gen_dir = Path(gen_dir)

    return {
        "kld": metrics.compare(ref_dir, gen_dir, metrics.kld),
        "oa": metrics.compare(ref_dir, gen_dir, metrics.oa),
        "fmd": metrics.compare(ref_dir, gen_dir, metrics.fmd),
    }
