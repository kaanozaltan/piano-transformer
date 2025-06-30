from piano_transformer.utils import metrics

import shutil

from pathlib import Path
from tqdm import tqdm
import torch
import wandb
from copy import deepcopy
from transformers import TrainerCallback


# use 1 gpu with this callback, fails on 2 gpus
class EvalCallback(TrainerCallback):
    def __init__(self, tokenizer, dataset, ref_dir, gen_dir):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.ref_dir = ref_dir
        self.gen_dir = gen_dir

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if self.gen_dir.exists():
            shutil.rmtree(self.gen_dir)
        self.gen_dir.mkdir(parents=True)

        model = model.cuda().eval()
        generate_and_save(model, self.tokenizer, self.dataset, self.gen_dir)
        # generate_and_save(model, self.tokenizer, self.dataset, self.gen_dir, max_samples=50)
        metrics = evaluate(self.ref_dir, self.gen_dir)
        print(f"Evaluation metrics: {metrics}")
        wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=state.global_step)


def generate_and_save(model, tokenizer, dataset, save_dir, max_samples=50, mode="eval"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    count = 0
    for i, batch in enumerate(tqdm(loader, desc="Generating")):
        if max_samples is not None and count >= max_samples:
            break

        input_ids = batch["input_ids"].cuda()  # shape: (1, seq_len)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                # max_length=8192,
                max_new_tokens=200,
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

        count += 1


# def generate_and_save(model, tokenizer, dataset, output, max_samples=50):
#     (output_path := Path(output)).mkdir(parents=True, exist_ok=True)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, collate_fn=collator)

#     count = 0
#     for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating outputs")):
#         print(f"[INFO] Processing batch {batch_idx} (generated so far: {count}/{max_samples})")
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
#             midi_generated.dump_midi(output_path / f"{count}.midi")
#             count += 1

#             if max_samples and count >= max_samples:
#                 break
#         if max_samples is not None and count >= max_samples:
#             break


def evaluate(ref_dir, gen_dir):
    ref_dir = Path(ref_dir)
    gen_dir = Path(gen_dir)

    kld = metrics.compare(ref_dir, gen_dir, metrics.kld)
    oa = metrics.compare(ref_dir, gen_dir, metrics.oa)
    fmd = metrics.compare(ref_dir, gen_dir, metrics.fmd)

    return {
        "kld": kld,
        "oa": oa,
        "fmd": fmd,
    }
