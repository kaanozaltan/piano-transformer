from typing import List, Optional
from pathlib import Path

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
    num_samples: int = 50,
    max_gen_len: Optional[int] = None,
    finetuned_PEFT_weight_path: Optional[str] = None,
    generation_mode: str = "all_test_files",  # "from_scratch", "random_files", or "all_test_files"
    folder: str = "first"
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
    
    prompts = []
    
    if generation_mode == "from_scratch":
        # Only use SOS token as prompt
        prompts = [[generator.tokenizer.sos_token_compound] for _ in range(num_samples)]
    
    elif generation_mode == "random_files":
        # Randomly sample test files and use as prompts
        df = pd.read_csv(csv_file)
        split = "test"
        test_filenames = df[df['split'] == split]['file_base_name'].tolist()
        test_files_sampled = random.sample(test_filenames, num_samples)

        for filename in test_files_sampled:
            test_data = np.load(os.path.join(os.path.dirname(csv_file), 'processed', filename))
            test_data_with_sos = generator.tokenizer.encode_series(test_data, if_add_sos = True, if_add_eos = False)
            prompts.append(test_data_with_sos[:prompt_len])

    elif generation_mode == "all_test_files":
        # Use all test files for continuation 
        df = pd.read_csv(csv_file)
        split = "test"
        test_files_sampled = df[df['split'] == split]['file_base_name'].tolist() * 2
        # test_files_sampled = random.sample(test_filenames, num_samples)

        for filename in test_files_sampled:
            test_data = np.load(os.path.join(os.path.dirname(csv_file), 'processed', filename))
            test_data_with_sos = generator.tokenizer.encode_series(test_data, if_add_sos = True, if_add_eos = False)
            prompts.append(test_data_with_sos[:prompt_len])
    
    else:
        raise ValueError(f"Invalid generation_mode: {generation_mode}. Must be one of: 'from_scratch', 'random_files', 'all_test_files'")

        # # Load chunked test data
        # df = pd.read_csv(csv_file)
        # split = "test"
        # test_files = df[df['split'] == split]['file_base_name'].tolist()

        # for filename in test_files:
        #     test_data = np.load(os.path.join(os.path.dirname(csv_file), 'processed', filename))
        #     test_data_tokenized = generator.tokenizer.encode_series(test_data, if_add_sos = False, if_add_eos = False)
            
        #     # Calculate number of full prompt_len-sized chunks
        #     num_chunks = len(test_data_tokenized) // prompt_len
        #     for i in range(num_chunks):
        #         chunk = test_data_tokenized[i * prompt_len : (i + 1) * prompt_len]
        #         # Convert all values to plain Python ints
        #         chunk = [[int(value) for value in token] for token in chunk]
        #         sos_token = [[int(value) for value in token] for token in [generator.tokenizer.sos_token_compound]]
        #         chunk_with_sos = sos_token + chunk
        #         prompts.append(chunk_with_sos)


    # if from_scratch:
    results = generator.music_completion(
    prompts,
    max_gen_len=max_gen_len,
    temperature=temperature,
    top_p=top_p,
    )   
    # else:
    #     results = []
    #     BATCH_SIZE = 1000
    #     for i in range(0, len(prompts), BATCH_SIZE):
    #         prompt_batch = prompts[i:i + BATCH_SIZE]
            
    #         batch_results = generator.music_completion(
    #             prompt_batch,
    #             max_gen_len=max_gen_len,
    #             temperature=temperature,
    #             top_p=top_p,
    #         )

    #         results.extend(batch_results)

    # Build generation settings folder name
    gen_settings_folder = f"temperature_{temperature}_top_p_{top_p}_genlen_{max_gen_len}"

    # Add prompt length for continuation modes
    if generation_mode != "from_scratch":
        gen_settings_folder += f"_promptlen_{prompt_len}"

    # Build final save path based on generation mode
    save_folder = os.path.join(
        finetuned_PEFT_weight_path,
        Path(ckpt_dir).stem,
        generation_mode,  # "from_scratch", "random_files", or "all_test_files"
        gen_settings_folder
    )

    os.makedirs(save_folder, exist_ok=True)


    def get_next_start_index(save_folder, epoch_step):
        """Find max index used in existing files and return next starting index."""
        if not os.path.exists(save_folder):
            return 0  # folder doesn't exist yet, start from zero

        existing_files = os.listdir(save_folder)
        pattern = re.compile(rf"{re.escape(epoch_step)}_(\d+)\.mid$")
        indices = []

        for filename in existing_files:
            match = pattern.search(filename)
            if match:
                indices.append(int(match.group(1)))

        if not indices:
            return 0
        return max(indices) + 1
    
    epoch_step = os.path.splitext(os.path.basename(ckpt_dir))[0]
    start_index = get_next_start_index(save_folder, epoch_step)
    
    for i, (dialog, result) in enumerate(zip(prompts, results), start=start_index):
        save_path = f'{save_folder}/{epoch_step}_{i}.mid'
        try:
            result['generation']['content'].save(save_path)
            print(f"Midi saved to {save_path}")
        except Exception as e:
            print("Error saving MIDI file:", e)

        if generation_mode != "from_scratch":  # Save prompt for random_files and all_test_files modes (both use real data prompts)
            prompt_save_path = save_path.replace(".mid", "_prompt.mid")
            try:
                result['generation']['prompt'].save(prompt_save_path)
                print(f"Prompt MIDI saved to {prompt_save_path}")
            except Exception as e:
                print("Error saving prompt MIDI file:", e)

        print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)
