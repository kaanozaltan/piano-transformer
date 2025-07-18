# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional

import fire
import pandas as pd
import numpy as np
import os
import re
# from llama import Dialog, Llama
from generation import MusicLlama
import random

def main(
    ckpt_dir: str,
    csv_file: str,
    tokenizer_path: str,
    model_config_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    prompt_len: int = 5,
    num_test_data: int = 50,
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
    random.seed(seed)  # You can choose any seed value, 42 is commonly used
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.    


    generator = MusicLlama.build(
        ckpt_dir=ckpt_dir,
        model_config_path = model_config_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    ) # this is a model actually

    #1. read the csv file; 2. read the header with split = test and get the npy file path; 3. load the npy file; 4. get the first 10 tokens;
    
    prompts = []
    df = pd.read_csv(csv_file)

    # 2. Find the row where 'split' is 'test' and get the corresponding .npy file path
    test_filenames = df[df['split'] == 'train']['file_base_name'].tolist() #change from test to train, since there's really no test data in LLM
    test_filenames_sampled = random.sample(test_filenames, num_test_data)

    for filename in test_filenames_sampled:
        test_data = np.load(os.path.join(os.path.dirname(csv_file), filename))
        test_data_with_sos = generator.tokenizer.encode_series(test_data, if_add_sos = True, if_add_eos = False)
        prompts.append(test_data_with_sos[:prompt_len])
        
    results = generator.music_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    save_folder = os.path.join(os.path.dirname(ckpt_dir), f"temperature_{temperature}_top_p_{top_p}")
    os.makedirs(save_folder, exist_ok=True)
    for i, (dialog, result) in enumerate(zip(prompts, results)):
        for msg in dialog:
            print(f"msg: {msg}")

        epoch_step = re.search(r'(\d+-\d+)\.pt$', ckpt_dir).group(1)
        for j, sub_mid in enumerate(result['generation']['content']):
            sub_mid.save(f'{save_folder}/{epoch_step}_{str(i)}_{str(j)}.mid')
        print(f"midis saved at {save_folder}/{epoch_step}_{str(i)}_x.mid")
        #Also save the prompt 
        generator.tokenizer.compound_to_midi(dialog[1:]).save(f'{save_folder}/{epoch_step}_{str(i)}_prompt.mid')
        print(f"midi prompt saved at {save_folder}/{epoch_step}_{str(i)}_prompt.mid")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
