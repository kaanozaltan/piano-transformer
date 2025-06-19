from typing import List, Optional

import fire
import pandas as pd
import numpy as np
import os
import re
from generation import MusicLlama
import random
import ast
import json

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
    finetuned_PEFT_weight_path: Optional[str] = None,
):

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
        finetuned_PEFT_weight_path = finetuned_PEFT_weight_path) 
    
    df = pd.read_csv(csv_file)
    split = "test"
    test_filenames = df[df['split'] == split]['file_base_name'].tolist()
    test_files_sampled = random.sample(test_filenames, num_test_data)
    prompts = []

    for filename in test_files_sampled:
        test_data = np.load(os.path.join(os.path.dirname(csv_file), 'processed', filename))
        test_data_with_sos = generator.tokenizer.encode_series(test_data, if_add_sos = True, if_add_eos = False)
        prompts.append(test_data_with_sos[:prompt_len])
    
    results = generator.music_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    
    save_folder = os.path.join(finetuned_PEFT_weight_path, os.path.basename(ckpt_dir), f"temperature_{temperature}_top_p_{top_p}")
    os.makedirs(save_folder, exist_ok=True)

    for i, (dialog, result) in enumerate(zip(prompts, results)):
        # epoch_step = re.search(r'(\d+-\d+)\.pt$', ckpt_dir).group(1)
        epoch_step = os.path.splitext(os.path.basename(ckpt_dir))[0]
        save_path = f'{save_folder}/{epoch_step}_{str(i)}.mid'
        result['generation']['content'].save(save_path)
        result['generation']['prompt'].save(save_path.replace(".mid", "_prompt.mid"))
        print(f"Midi and prompt saved to {save_path} and {save_path.replace('.mid', '_prompt.mid')}")
        print("\n==================================\n")
if __name__ == "__main__":
    fire.Fire(main)
