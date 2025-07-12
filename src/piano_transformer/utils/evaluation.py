from piano_transformer.utils import metrics

import shutil
import glob
import os

from pathlib import Path
from tqdm import tqdm
import torch
import wandb
from copy import deepcopy
from transformers import TrainerCallback, GenerationConfig
from piano_transformer.utils.metrics import fmd, create_subset, evaluate_mgeval_combined
from transformers.trainer_utils import is_main_process


class EvalCallback(TrainerCallback):
    def __init__(self, tokenizer, ref_dir, gen_dir, num_samples, every_n_steps):
        self.tokenizer = tokenizer
        self.ref_dir = ref_dir
        self.gen_dir = gen_dir
        self.num_samples = num_samples
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every_n_steps != 0:
            return
        if not is_main_process(args.local_rank):
            return
        print(f"[EvalCallback] Running custom eval at step {state.global_step} ...")
        gen_dir_step = self.gen_dir / f"step_{state.global_step}"
        if gen_dir_step.exists():
            shutil.rmtree(gen_dir_step)
        gen_dir_step.mkdir(parents=True)

        generate_from_scratch(
            model,
            self.tokenizer,
            gen_dir_step,
            num_samples=self.num_samples,
            max_new_tokens=1024,
            batch_size=64,
        )

        train_dir_subset = create_subset(self.ref_dir, self.num_samples)

        metrics = {}

        fmd_value = fmd(train_dir_subset, gen_dir_step)

        metrics["FMD"] = fmd_value

        absolute_summary, relative_summary = evaluate_mgeval_combined(
            dataset1_path=train_dir_subset,
            dataset2_path=gen_dir_step,
            compute_kld=False,
        )

        for item in absolute_summary:
            feature_name = item["Feature"]
            metrics[f"{feature_name}_Rel_Diff_Mean"] = item["Rel_Diff_Mean"]
            metrics[f"{feature_name}_Rel_Diff_Std"] = item["Rel_Diff_Std"]

        # kld_sum = 0
        oa_sum = 0
        for item in relative_summary:
            feature_name = item["Feature"]
            # kld_sum += item["KLD"]
            # metrics[f"{feature_name}_KLD"] = item["KLD"]
            oa_sum += item["OA"]
            metrics[f"{feature_name}_OA"] = item["OA"]
        # metrics["KLD_average"] = kld_sum / len(relative_summary)
        metrics["OA_average"] = oa_sum / len(relative_summary)

        print(f"Evaluation metrics: {metrics}")

        wandb.log(
            {f"custom_eval/{k}": v for k, v in metrics.items()}, step=state.global_step
        )


def generate_from_scratch(
    model, tokenizer, output, num_samples, max_new_tokens, batch_size
):
    (output_path := Path(output)).mkdir(parents=True, exist_ok=True)

    count = 0
    for batch_start in tqdm(
        range(0, num_samples, batch_size), desc="Generating from scratch"
    ):
        current_batch_size = min(batch_size, num_samples - count)
        # Create empty generation inputs with just the BOS token
        BOS_TOKEN_ID = 1
        input_ids = torch.full(
            (current_batch_size, 1), BOS_TOKEN_ID, dtype=torch.long, device=model.device
        )

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1,
            pad_token_id=tokenizer.pad_token_id,
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


# def generate_and_save(model, tokenizer, dataset, save_dir, max_samples=50, mode="eval"):
#     save_dir = Path(save_dir)
#     save_dir.mkdir(parents=True, exist_ok=True)

#     loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

#     count = 0
#     for i, batch in enumerate(tqdm(loader, desc="Generating")):
#         if max_samples is not None and count >= max_samples:
#             break

#         input_ids = batch["input_ids"].cuda()  # shape: (1, seq_len)

#         with torch.no_grad():
#             generated_ids = model.generate(
#                 input_ids=input_ids,
#                 # max_length=8192,
#                 max_new_tokens=200,
#                 do_sample=True,
#                 top_k=50,
#                 top_p=0.95,
#                 temperature=1.0,
#                 pad_token_id=tokenizer["PAD_None"],
#                 bos_token_id=tokenizer["BOS_None"],
#                 eos_token_id=tokenizer["EOS_None"],
#             )

#         prompt = input_ids[0]
#         generated = generated_ids[0][len(prompt) :]
#         full = generated_ids[0]

#         if mode == "eval":
#             midi_generated = tokenizer.decode([deepcopy(generated.tolist())])
#             midi_generated.dump_midi(save_dir / f"{i}.midi")
#         elif mode == "all":
#             midi_prompt = tokenizer.decode([deepcopy(prompt.tolist())])
#             midi_generated = tokenizer.decode([deepcopy(generated.tolist())])
#             midi_full = tokenizer.decode([deepcopy(full.tolist())])

#             midi_prompt.dump_midi(save_dir / f"{i}_prompt.midi")
#             midi_generated.dump_midi(save_dir / f"{i}_generated.midi")
#             midi_full.dump_midi(save_dir / f"{i}_full.midi")
#         else:
#             raise ValueError(f"Unknown mode: {mode}")

#         count += 1


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


# def evaluate(ref_dir, gen_dir):
#     ref_dir = Path(ref_dir)
#     gen_dir = Path(gen_dir)

#     kld = metrics.compare(ref_dir, gen_dir, metrics.kld)
#     oa = metrics.compare(ref_dir, gen_dir, metrics.oa)
#     fmd = metrics.compare(ref_dir, gen_dir, metrics.fmd)

#     return {
#         "kld": kld,
#         "oa": oa,
#         "fmd": fmd,
#     }
