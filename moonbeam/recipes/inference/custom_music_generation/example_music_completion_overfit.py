# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional

import fire

# from llama import Dialog, Llama
from generation import MusicLlama

def main(
    ckpt_dir: str,
    overfitting_ckpt_sample_path: str,
    tokenizer_path: str,
    model_config_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    # Set the random seed for CPU and GPU
    seed = 42
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.    

    overfitting_sample = torch.load(f"{overfitting_ckpt_sample_path}/batch_data_train.pth")['input_ids']
    overfitting_label = torch.load(f"{overfitting_ckpt_sample_path}/batch_data_train.pth")['labels']
    generator = MusicLlama.build(
        ckpt_dir=ckpt_dir,
        model_config_path = model_config_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    ) # this is a model actually

    prompts = overfitting_sample[:, :5].tolist()
    print(f"overfitting prompts during training:{prompts}")
    print(f"overfitting labels during training:{overfitting_label}")
    results = generator.music_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for i, (dialog, result) in enumerate(zip(prompts, results)):
        for msg in dialog:
            print(f"msg: {msg}")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        result['generation']['content'].save(f'{overfitting_ckpt_sample_path}/{str(i)}.mid')
        print(f"midi saved at {overfitting_ckpt_sample_path}/{str(i)}.mid")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
