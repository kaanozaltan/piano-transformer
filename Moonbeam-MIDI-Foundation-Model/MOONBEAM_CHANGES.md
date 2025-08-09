# Changes (Original Moonbeam Repository → Our Additions)
_Generated: 2025-08-09 17:16:14_

```diff
diff --git a/data_preprocess.py b/data_preprocess.py
index ffe8848..8cf8c3e 100644
--- a/data_preprocess.py
+++ b/data_preprocess.py
@@ -61,6 +61,25 @@ def find_midi_files_from_file(dataset_name, split_file, dataset_folder):
         midi_files = train_files + test_files
         splits = ['train']*len(train_files) + ['test']*len(test_files)
 
+    elif dataset_name == "maestro":
+        df = pd.read_csv(split_file)
+        train_files = df[df['split'] == 'train']['midi_filename'].tolist()
+        val_files = df[df['split'] == 'validation']['midi_filename'].tolist()
+        test_files = df[df['split'] == 'test']['midi_filename'].tolist()
+
+        midi_files = train_files + val_files + test_files
+        midi_files = [os.path.join(dataset_folder, f) for f in midi_files]
+
+        splits = (
+            ['train'] * len(train_files) +
+            ['validation'] * len(val_files) +
+            ['test'] * len(test_files)
+        )
+
+    else:
+        raise ValueError(f"Unsupported dataset name: {dataset_name}")
+
+
     assert len(midi_files) == len(splits)
     return midi_files, splits
 

diff --git a/post-process/compare_tokenizers.py b/post-process/compare_tokenizers.py
new file mode 100644
index 0000000..1beec32
--- /dev/null
+++ b/post-process/compare_tokenizers.py
@@ -0,0 +1,99 @@
+import os
+import re
+import json
+import numpy as np
+from collections import defaultdict
+
+# Count lengths of npy files in 309M and 839M
+def collect_npy_lengths(base_dir):
+    lengths_by_year = defaultdict(dict)
+    for file in os.listdir(base_dir):
+        if file.endswith(".npy"):
+            year_match = re.search(r"maestro_(\d{4})", file)
+            if not year_match:
+                continue
+            year = year_match.group(1)
+            # Remove "data_maestro_" and the year + underscore from filename
+            name = file.replace("data_maestro_", "")
+            # Remove year prefix like "2004_" or "2006_" from the name if present
+            name = re.sub(r"^\d{4}_", "", name)
+            name = name.replace(".npy", "")
+            file_path = os.path.join(base_dir, file)
+            try:
+                data = np.load(file_path, allow_pickle=True)
+                lengths_by_year[year][name] = len(data)
+            except Exception as e:
+                print(f"Failed to load {file_path}: {e}")
+    return lengths_by_year
+
+lengths_309M = collect_npy_lengths("preprocessed/309M/processed")
+lengths_839M = collect_npy_lengths("preprocessed/839M/processed")
+
+# Count MIDI chunks per base file across train/val/test
+midi_dirs = [
+    "/hpcwork/lect0148/experiments/mistral-155M_remi_maestro_v8/data_processed/maestro_train",
+    "/hpcwork/lect0148/experiments/mistral-155M_remi_maestro_v8/data_processed/maestro_validation",
+    "/hpcwork/lect0148/experiments/mistral-155M_remi_maestro_v8/data_processed/maestro_test"
+]
+
+midi_chunks = defaultdict(lambda: defaultdict(int))
+
+for base_dir in midi_dirs:
+    for year in os.listdir(base_dir):
+        year_path = os.path.join(base_dir, year)
+        if not os.path.isdir(year_path):
+            continue
+        for file in os.listdir(year_path):
+            if file.endswith(".midi"):
+                base_name = re.sub(r"_\d+\.midi$", "", file)
+                midi_chunks[year][base_name] += 1
+
+# Multiply chunk counts by 1024
+for year in midi_chunks:
+    for base_name in midi_chunks[year]:
+        midi_chunks[year][base_name] *= 1024
+
+
+# Compute averages per year
+def average_dicts(dict_of_dicts):
+    averages = {}
+    for year, files in dict_of_dicts.items():
+        if len(files) > 0:
+            averages[year] = sum(files.values()) / len(files)
+    return averages
+
+avg_309M = average_dicts(lengths_309M)
+avg_839M = average_dicts(lengths_839M)
+avg_midi_chunks = average_dicts(midi_chunks)
+
+# Calculate scaling factors
+scaling_factors_309M = {}
+scaling_factors_839M = {}
+
+for year in avg_midi_chunks:
+    if year in avg_309M:
+        scaling_factors_309M[year] = avg_midi_chunks[year] / avg_309M[year]
+    if year in avg_839M:
+        scaling_factors_839M[year] = avg_midi_chunks[year] / avg_839M[year]
+
+# Save to JSON
+output = {
+    "npy_lengths": {
+        "309M": lengths_309M,
+        "839M": lengths_839M
+    },
+    "midi_chunks": midi_chunks,
+    "averages": {
+        "309M": avg_309M,
+        "839M": avg_839M,
+        "midi_chunks": avg_midi_chunks
+    },
+    "scaling_factors": {
+        "309M": scaling_factors_309M,
+        "839M": scaling_factors_839M
+    }
+}
+
+with open("analysis_results.json", "w") as f:
+    json.dump(output, f, indent=2)
+
diff --git a/post-process/post-process.py b/post-process/post-process.py
new file mode 100644
index 0000000..707fbe7
--- /dev/null
+++ b/post-process/post-process.py
@@ -0,0 +1,192 @@
+import os
+import time
+from mido import MidiFile
+
+FOLDER_PATH = "/hpcwork/yh522379/moonbeam/checkpoints/fine-tuned/relevant/ft_309M_peft_ctx1024_bs16_gradacc4_clipTrue_clipthresh1.0_lr1e-4_cosine_gamma0.99_temp1.1_ep150_20250807_155620/generation_results"
+
+def is_valid_midi(file_path, verbose=True):
+    """
+    Check if a MIDI file is valid by attempting to load it.
+    
+    Args:
+        file_path: Path to the MIDI file
+        verbose: Whether to print detailed validation info
+    
+    Returns:
+        tuple: (is_valid, error_message, midi_info)
+    """
+    try:
+        start_time = time.time()
+        midi_file = MidiFile(file_path)
+        load_time = time.time() - start_time
+        
+        # Get MIDI file information
+        num_tracks = len(midi_file.tracks)
+        total_messages = sum(len(track) for track in midi_file.tracks)
+        duration = midi_file.length
+        
+        midi_info = {
+            'tracks': num_tracks,
+            'messages': total_messages,
+            'duration': round(duration, 2),
+            'load_time': round(load_time * 1000, 2)  # in milliseconds
+        }
+        
+        return True, None, midi_info
+        
+    except Exception as e:
+        error_msg = str(e)
+        if verbose:
+            relative_path = os.path.relpath(file_path, FOLDER_PATH)
+            print(f"INVALID: {relative_path} | Error: {error_msg}")
+        return False, error_msg, None
+
+def find_midi_files(folder_path):
+    """
+    Recursively find all MIDI files in folder and subdirectories.
+    
+    Args:
+        folder_path: Root folder path to search
+        
+    Returns:
+        list: List of full paths to MIDI files
+    """
+    midi_files = []
+    
+    for root, dirs, files in os.walk(folder_path):
+        for file in files:
+            if file.lower().endswith(('.mid', '.midi')):
+                full_path = os.path.join(root, file)
+                midi_files.append(full_path)
+    
+    return midi_files
+
+def get_directory_stats(folder_path):
+    """
+    Get statistics about directory structure.
+    
+    Args:
+        folder_path: Root folder path
+        
+    Returns:
+        dict: Directory statistics
+    """
+    total_dirs = 0
+    total_files = 0
+    
+    for root, dirs, files in os.walk(folder_path):
+        total_dirs += len(dirs)
+        total_files += len(files)
+    
+    return {
+        'directories': total_dirs,
+        'total_files': total_files
+    }
+
+def delete_invalid_midis(folder_path, verbose=True):
+    """
+    Process all MIDI files in a folder and subdirectories, validate them, and delete invalid ones.
+    
+    Args:
+        folder_path: Path to the root folder containing MIDI files
+        verbose: Whether to print detailed processing info
+    """
+    global FOLDER_PATH
+    FOLDER_PATH = folder_path
+    
+    print("=" * 80)
+    print("RECURSIVE MIDI FILE VALIDATION AND CLEANUP")
+    print("=" * 80)
+    
+    # Check if folder exists
+    if not os.path.isdir(folder_path):
+        print(f"ERROR: Invalid folder path: {folder_path}")
+        return
+    
+    print(f"Processing folder (recursive): {folder_path}")
+    
+    # Get directory statistics
+    dir_stats = get_directory_stats(folder_path)
+    print(f"Directory structure: {dir_stats['directories']} subdirectories, {dir_stats['total_files']} total files")
+    
+    # Find all MIDI files recursively
+    print("Scanning for MIDI files in all subdirectories...")
+    midi_files = find_midi_files(folder_path)
+    
+    print(f"Found {len(midi_files)} MIDI files across all directories")
+    
+    if not midi_files:
+        print("No MIDI files found to process.")
+        return
+    
+    # Show directory distribution
+    if verbose:
+        dir_counts = {}
+        for midi_file in midi_files:
+            dir_name = os.path.dirname(os.path.relpath(midi_file, folder_path))
+            if dir_name == "":
+                dir_name = "." # root directory
+            dir_counts[dir_name] = dir_counts.get(dir_name, 0) + 1
+        
+        print(f"\nMIDI files distribution:")
+        for dir_name, count in sorted(dir_counts.items()):
+            print(f"   {dir_name}: {count} files")
+    
+    # Initialize counters
+    valid_count = 0
+    invalid_count = 0
+    deleted_count = 0
+    error_count = 0
+    
+    valid_files = []
+    invalid_files = []
+    
+    print(f"\nStarting validation of {len(midi_files)} MIDI files...")
+    print("-" * 80)
+    
+    # Process each MIDI file
+    for i, file_path in enumerate(midi_files, 1):
+        is_valid, error_msg, midi_info = is_valid_midi(file_path, verbose)
+        
+        if is_valid:
+            valid_count += 1
+            valid_files.append((file_path, midi_info))
+        else:
+            invalid_count += 1
+            invalid_files.append((file_path, error_msg))
+            
+            # Try to delete invalid file
+            try:
+                os.remove(file_path)
+                deleted_count += 1
+                if verbose:
+                    relative_path = os.path.relpath(file_path, folder_path)
+                    print(f"    Successfully deleted: {relative_path}")
+            except Exception as delete_error:
+                error_count += 1
+                relative_path = os.path.relpath(file_path, folder_path)
+                print(f"    Could not delete {relative_path}: {delete_error}")
+    
+    # Print summary
+    print("\n" + "=" * 80)
+    print("PROCESSING SUMMARY")
+    print("=" * 80)
+    print(f"Root folder: {folder_path}")
+    print(f"Total MIDI files processed: {len(midi_files)}")
+    print(f"Valid MIDI files: {valid_count}")
+    print(f"Invalid MIDI files: {invalid_count}")
+    print(f"Files deleted: {deleted_count}")
+    print(f"Deletion errors: {error_count}")
+    print(f"Success rate: {valid_count/len(midi_files)*100:.1f}%")
+    
+    if invalid_files:
+        print(f"\nINVALID FILES REMOVED:")
+        for file_path, error in invalid_files:
+            relative_path = os.path.relpath(file_path, folder_path)
+            print(f"   {relative_path}: {error}")
+    
+    print("=" * 80)
+    print("Processing complete!")
+
+if __name__ == "__main__":
+    delete_invalid_midis(FOLDER_PATH, verbose=True)
\ No newline at end of file
diff --git a/recipes/inference/custom_music_generation/generation.py b/recipes/inference/custom_music_generation/generation.py
index 8b1ae03..cab92f8 100644
--- a/recipes/inference/custom_music_generation/generation.py
+++ b/recipes/inference/custom_music_generation/generation.py
@@ -161,6 +161,13 @@ class MusicLlama:
         past_key_values = None
         for cur_pos in range(min_prompt_len, total_len): #recursively generate new tokens in parallel
             print(f"{cur_pos}/{total_len} generated")
+
+            # GPU memory usage
+            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
+            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
+            print(f"    CUDA Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
+            print("Grad enabled before forward:", torch.is_grad_enabled())
+
             output = self.model.forward(input_ids = tokens[:, prev_pos:cur_pos], past_key_values = past_key_values, use_cache = True, attention_mask = None) #output logtis: (batch, len, dim 
             next_decoder_token = torch.tensor(self.tokenizer.sos_out).to(tokens).expand(tokens.shape[0]*(cur_pos - prev_pos), 1) #batch*len_x, len_y = 1
             next_decoder_token_out = next_decoder_token
@@ -276,7 +283,7 @@ class MusicLlama:
             temperature=temperature,
             top_p=top_p,
             logprobs=logprobs,
-            echo = True
+            echo = False,  # Do not echo prompt tokens in the output
         )
 
         prompt_tokens = [t[1:] for t in prompt_tokens] #remove SOS token
@@ -293,18 +300,37 @@ class MusicLlama:
                 }
                 for t, logprobs_i in zip(generation_tokens, generation_logprobs)
             ]
-        return [
-            {
+        # return [
+        #     {
+        #         "generation": {
+        #             "role": "assistant",
+        #             "content": self.tokenizer.compound_to_midi(t), 
+        #             "prompt": self.tokenizer.compound_to_midi(p),
+        #             "prompt_tokens": p,
+        #             "tokens": t,
+        #         },
+        #     }
+        #     for t, p in zip(generation_tokens, prompt_tokens) 
+        # ]
+        results = []
+        for t, p in zip(generation_tokens, prompt_tokens):
+            try:
+                content = self.tokenizer.compound_to_midi(t)
+                prompt = self.tokenizer.compound_to_midi(p)
+                results.append({
                     "generation": {
                         "role": "assistant",
-                    "content": self.tokenizer.compound_to_midi(t), 
-                    "prompt": self.tokenizer.compound_to_midi(p),
+                        "content": content,
+                        "prompt": prompt,
                         "prompt_tokens": p,
                         "tokens": t,
                     },
-            }
-            for t, p in zip(generation_tokens, prompt_tokens) 
-        ]
+                })
+            except Exception as e:
+                print(f"[Warning] Skipped a generation due to error in compound_to_midi: {e}")
+                continue
+
+        return results
 
 def sample_top_p(probs, p):
     """
diff --git a/recipes/inference/custom_music_generation/unconditional_music_generation.py b/recipes/inference/custom_music_generation/unconditional_music_generation.py
index fe3ea26..5f40227 100644
--- a/recipes/inference/custom_music_generation/unconditional_music_generation.py
+++ b/recipes/inference/custom_music_generation/unconditional_music_generation.py
@@ -1,4 +1,5 @@
 from typing import List, Optional
+from pathlib import Path
 
 import fire
 import pandas as pd
@@ -20,9 +21,11 @@ def main(
     max_seq_len: int = 512,
     max_batch_size: int = 4,
     prompt_len: int = 5,
-    num_test_data: int = 50,
+    num_samples: int = 50,
     max_gen_len: Optional[int] = None,
     finetuned_PEFT_weight_path: Optional[str] = None,
+    generation_mode: str = "all_test_files",  # "from_scratch", "random_files", or "all_test_files"
+    folder: str = "first"
 ):
 
     # Set the random seed for CPU and GPU
@@ -42,33 +45,137 @@ def main(
         max_batch_size=max_batch_size,
         finetuned_PEFT_weight_path = finetuned_PEFT_weight_path) 
     
+    prompts = []
+    
+    if generation_mode == "from_scratch":
+        # Only use SOS token as prompt
+        prompts = [[generator.tokenizer.sos_token_compound] for _ in range(num_samples)]
+    
+    elif generation_mode == "random_files":
+        # Randomly sample test files and use as prompts
         df = pd.read_csv(csv_file)
         split = "test"
         test_filenames = df[df['split'] == split]['file_base_name'].tolist()
-    test_files_sampled = random.sample(test_filenames, num_test_data)
-    prompts = []
+        test_files_sampled = random.sample(test_filenames, num_samples)
+
+        for filename in test_files_sampled:
+            test_data = np.load(os.path.join(os.path.dirname(csv_file), 'processed', filename))
+            test_data_with_sos = generator.tokenizer.encode_series(test_data, if_add_sos = True, if_add_eos = False)
+            prompts.append(test_data_with_sos[:prompt_len])
+
+    elif generation_mode == "all_test_files":
+        # Use all test files for continuation 
+        df = pd.read_csv(csv_file)
+        split = "test"
+        test_files_sampled = df[df['split'] == split]['file_base_name'].tolist() * 2
+        # test_files_sampled = random.sample(test_filenames, num_samples)
 
         for filename in test_files_sampled:
             test_data = np.load(os.path.join(os.path.dirname(csv_file), 'processed', filename))
             test_data_with_sos = generator.tokenizer.encode_series(test_data, if_add_sos = True, if_add_eos = False)
             prompts.append(test_data_with_sos[:prompt_len])
     
+    else:
+        raise ValueError(f"Invalid generation_mode: {generation_mode}. Must be one of: 'from_scratch', 'random_files', 'all_test_files'")
+
+        # # Load chunked test data
+        # df = pd.read_csv(csv_file)
+        # split = "test"
+        # test_files = df[df['split'] == split]['file_base_name'].tolist()
+
+        # for filename in test_files:
+        #     test_data = np.load(os.path.join(os.path.dirname(csv_file), 'processed', filename))
+        #     test_data_tokenized = generator.tokenizer.encode_series(test_data, if_add_sos = False, if_add_eos = False)
+            
+        #     # Calculate number of full prompt_len-sized chunks
+        #     num_chunks = len(test_data_tokenized) // prompt_len
+        #     for i in range(num_chunks):
+        #         chunk = test_data_tokenized[i * prompt_len : (i + 1) * prompt_len]
+        #         # Convert all values to plain Python ints
+        #         chunk = [[int(value) for value in token] for token in chunk]
+        #         sos_token = [[int(value) for value in token] for token in [generator.tokenizer.sos_token_compound]]
+        #         chunk_with_sos = sos_token + chunk
+        #         prompts.append(chunk_with_sos)
+
+
+    # if from_scratch:
     results = generator.music_completion(
     prompts,
     max_gen_len=max_gen_len,
     temperature=temperature,
     top_p=top_p,
     )   
+    # else:
+    #     results = []
+    #     BATCH_SIZE = 1000
+    #     for i in range(0, len(prompts), BATCH_SIZE):
+    #         prompt_batch = prompts[i:i + BATCH_SIZE]
+            
+    #         batch_results = generator.music_completion(
+    #             prompt_batch,
+    #             max_gen_len=max_gen_len,
+    #             temperature=temperature,
+    #             top_p=top_p,
+    #         )
+
+    #         results.extend(batch_results)
+
+    # Build generation settings folder name
+    gen_settings_folder = f"temperature_{temperature}_top_p_{top_p}_genlen_{max_gen_len}"
+
+    # Add prompt length for continuation modes
+    if generation_mode != "from_scratch":
+        gen_settings_folder += f"_promptlen_{prompt_len}"
+
+    # Build final save path based on generation mode
+    save_folder = os.path.join(
+        finetuned_PEFT_weight_path,
+        Path(ckpt_dir).stem,
+        generation_mode,  # "from_scratch", "random_files", or "all_test_files"
+        gen_settings_folder
+    )
 
-    save_folder = os.path.join(finetuned_PEFT_weight_path, os.path.basename(ckpt_dir), f"temperature_{temperature}_top_p_{top_p}")
     os.makedirs(save_folder, exist_ok=True)
 
-    for i, (dialog, result) in enumerate(zip(prompts, results)):
-        epoch_step = re.search(r'(\d+-\d+)\.pt$', ckpt_dir).group(1)
-        save_path = f'{save_folder}/{epoch_step}_{str(i)}.mid'
+
+    def get_next_start_index(save_folder, epoch_step):
+        """Find max index used in existing files and return next starting index."""
+        if not os.path.exists(save_folder):
+            return 0  # folder doesn't exist yet, start from zero
+
+        existing_files = os.listdir(save_folder)
+        pattern = re.compile(rf"{re.escape(epoch_step)}_(\d+)\.mid$")
+        indices = []
+
+        for filename in existing_files:
+            match = pattern.search(filename)
+            if match:
+                indices.append(int(match.group(1)))
+
+        if not indices:
+            return 0
+        return max(indices) + 1
+    
+    epoch_step = os.path.splitext(os.path.basename(ckpt_dir))[0]
+    start_index = get_next_start_index(save_folder, epoch_step)
+    
+    for i, (dialog, result) in enumerate(zip(prompts, results), start=start_index):
+        save_path = f'{save_folder}/{epoch_step}_{i}.mid'
+        try:
             result['generation']['content'].save(save_path)
-        result['generation']['prompt'].save(save_path.replace(".mid", "_prompt.mid"))
-        print(f"Midi and prompt saved to {save_path} and {save_path.replace('.mid', '_prompt.mid')}")
+            print(f"Midi saved to {save_path}")
+        except Exception as e:
+            print("Error saving MIDI file:", e)
+
+        if generation_mode != "from_scratch":  # Save prompt for random_files and all_test_files modes (both use real data prompts)
+            prompt_save_path = save_path.replace(".mid", "_prompt.mid")
+            try:
+                result['generation']['prompt'].save(prompt_save_path)
+                print(f"Prompt MIDI saved to {prompt_save_path}")
+            except Exception as e:
+                print("Error saving prompt MIDI file:", e)
+
         print("\n==================================\n")
+
 if __name__ == "__main__":
     fire.Fire(main)

diff --git a/requirements.txt b/requirements-original.txt
similarity index 100%
rename from requirements.txt
rename to requirements-original.txt
diff --git a/slurm_scripts/fine-tuning-batch.sh b/slurm_scripts/fine-tuning-batch.sh
new file mode 100755
index 0000000..5fcc682
--- /dev/null
+++ b/slurm_scripts/fine-tuning-batch.sh
@@ -0,0 +1,132 @@
+#!/bin/bash
+#SBATCH --export=ALL
+#SBATCH --job-name=moonbeam-ft
+#SBATCH --output=Moonbeam-MIDI-Foundation-Model/logs/moonbeam_ft_%j.out
+#SBATCH --error=Moonbeam-MIDI-Foundation-Model/logs/moonbeam_ft_%j.err
+#SBATCH --nodes=1
+#SBATCH --ntasks-per-node=1
+#SBATCH --cpus-per-gpu=24
+#SBATCH --gres=gpu:1
+#SBATCH --time=06:00:00
+#SBATCH --partition=c23g
+#SBATCH --account=lect0148
+#SBATCH --mail-type=ALL
+#SBATCH --mail-user=ikunabel@gmail.com
+
+module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.3
+export PYTHONPATH=$(pwd)/src:$(pwd)/Moonbeam-MIDI-Foundation-Model/src:$PYTHONPATH
+source Moonbeam-MIDI-Foundation-Model/.venv-moonbeam/bin/activate
+
+# Paths
+MODEL_SIZE="309M"  # Change to "309M" for the smaller model
+
+if [ "$MODEL_SIZE" = "839M" ]; then
+    PRETRAINED_CKPT="/hpcwork/yh522379/moonbeam/checkpoints/pre-trained/moonbeam_839M.pt"
+    MODEL_NAME="maestro"
+    DATASET_NAME="maestro_839M"
+    MODEL_CONFIG_PATH="Moonbeam-MIDI-Foundation-Model/src/llama_recipes/configs/model_config.json"
+elif [ "$MODEL_SIZE" = "309M" ]; then
+    PRETRAINED_CKPT="/hpcwork/yh522379/moonbeam/checkpoints/pre-trained/moonbeam_309M.pt"
+    MODEL_NAME="maestro"
+    DATASET_NAME="maestro_309M"
+    MODEL_CONFIG_PATH="Moonbeam-MIDI-Foundation-Model/src/llama_recipes/configs/model_config_small.json"
+else
+    echo "Error: MODEL_SIZE must be either '839M' or '309M'"
+    exit 1
+fi
+
+# Params
+NUM_EPOCHS="150"
+
+SCHEDULER_TYPE="cosine"
+LR="5e-5"
+GAMMA="0.99"
+
+GEN_MAX_LEN="500"
+GEN_SAMPLES="200"
+GEN_TEMPERATURE="1.1"
+ENABLE_GENERATION="True"
+ENABLE_EVALUATION="True"
+EVALUATION_FREQUENCY_EPOCHS="4"
+
+CONTEXT_LENGTH="1024"
+WEIGHT_DECAY="0.01"
+BATCH_SIZE="16"
+
+USE_PEFT="False"
+PEFT_METHOD="lora"
+
+GRADIENT_ACCUMULATION_STEPS="4"
+GRADIENT_CLIPPING="True"
+GRADIENT_CLIPPING_THRESHOLD="1.0"
+
+# Generate a random port to avoid conflicts
+MASTER_PORT=$((29500 + RANDOM % 1000))
+
+# Auto-generate names based on configuration
+TIMESTAMP=$(date +%Y%m%d_%H%M%S)
+if [ "$USE_PEFT" = "True" ]; then
+    PEFT_STRING="peft"
+else
+    PEFT_STRING="full"
+fi
+# Add weight decay to name only if it's not 0.0
+if [ "$WEIGHT_DECAY" != "0.0" ] && [ "$WEIGHT_DECAY" != "0" ]; then
+    WD_STRING="_wd${WEIGHT_DECAY}"
+else
+    WD_STRING=""
+fi
+# Add gradient clipping threshold to name only if clipping is enabled
+if [ "$GRADIENT_CLIPPING" = "True" ]; then
+    CLIP_STRING="_clip${GRADIENT_CLIPPING}_clipthresh${GRADIENT_CLIPPING_THRESHOLD}"
+else
+    CLIP_STRING="_clip${GRADIENT_CLIPPING}"
+fi
+CONFIG_STRING="${MODEL_SIZE}_${PEFT_STRING}_ctx${CONTEXT_LENGTH}_bs${BATCH_SIZE}_gradacc${GRADIENT_ACCUMULATION_STEPS}${CLIP_STRING}_lr${LR}_${SCHEDULER_TYPE}_gamma${GAMMA}${WD_STRING}_temp${GEN_TEMPERATURE}_ep${NUM_EPOCHS}"
+OUTPUT_DIR="$HPCWORK/moonbeam/checkpoints/fine-tuned/ft_${CONFIG_STRING}_${TIMESTAMP}"
+WANDB_NAME="moonbeam_ft_${CONFIG_STRING}_${TIMESTAMP}"
+
+
+mkdir -p logs
+
+# Run the fine-tuning script with random port to avoid conflicts
+echo "Using master port: $MASTER_PORT"
+torchrun --nnodes 1 --nproc_per_node 1 --master_port $MASTER_PORT Moonbeam-MIDI-Foundation-Model/recipes/finetuning/real_finetuning_uncon_gen.py \
+  --lr "$LR" \
+  --weight_decay "$WEIGHT_DECAY" \
+  --val_batch_size "$BATCH_SIZE" \
+  --run_validation True \
+  --validation_interval 20 \
+  --save_metrics True \
+  --dist_checkpoint_root_folder "$OUTPUT_DIR" \
+  --dist_checkpoint_folder ddp \
+  --trained_checkpoint_path "$PRETRAINED_CKPT" \
+  --pure_bf16 True \
+  --enable_ddp True \
+  --use_peft "$USE_PEFT" \
+  --peft_method "$PEFT_METHOD" \
+  --quantization False \
+  --model_name "$MODEL_NAME" \
+  --dataset "$DATASET_NAME" \
+  --output_dir "$OUTPUT_DIR" \
+  --batch_size_training "$BATCH_SIZE" \
+  --context_length "$CONTEXT_LENGTH" \
+  --num_epochs "$NUM_EPOCHS" \
+  --use_wandb True \
+  --gamma "$GAMMA" \
+  --scheduler_type "$SCHEDULER_TYPE" \
+  --model_config_path "$MODEL_CONFIG_PATH" \
+  --enable_generation "$ENABLE_GENERATION" \
+  --generation_save_dir "$OUTPUT_DIR/generation_results" \
+  --generation_temperature "$GEN_TEMPERATURE" \
+  --generation_top_p 0.95 \
+  --generation_max_gen_len "$GEN_MAX_LEN" \
+  --generation_num_samples "$GEN_SAMPLES" \
+  --generation_mode from_scratch \
+  --enable_evaluation "$ENABLE_EVALUATION" \
+  --evaluation_ref_dir "/hpcwork/lect0148/experiments/mistral-155M_remi_maestro_v8/output/subset/train" \
+  --wandb_name "$WANDB_NAME" \
+  --evaluation_frequency_epochs "$EVALUATION_FREQUENCY_EPOCHS" \
+  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
+  --gradient_clipping "$GRADIENT_CLIPPING" \
+  --gradient_clipping_threshold "$GRADIENT_CLIPPING_THRESHOLD"
diff --git a/slurm_scripts/fine-tuning.sh b/slurm_scripts/fine-tuning.sh
new file mode 100755
index 0000000..7fad838
--- /dev/null
+++ b/slurm_scripts/fine-tuning.sh
@@ -0,0 +1,107 @@
+#!/bin/bash
+
+module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.3
+export PYTHONPATH=$(pwd)/src:$(pwd)/Moonbeam-MIDI-Foundation-Model/src:$PYTHONPATH
+source Moonbeam-MIDI-Foundation-Model/.venv-moonbeam/bin/activate
+
+# Paths
+MODEL_SIZE="839M"  # Change to "309M" for the smaller model
+
+if [ "$MODEL_SIZE" = "839M" ]; then
+    PRETRAINED_CKPT="/hpcwork/yh522379/moonbeam/checkpoints/pre-trained/moonbeam_839M.pt"
+    MODEL_NAME="maestro"
+    DATASET_NAME="maestro_839M"
+    MODEL_CONFIG_PATH="Moonbeam-MIDI-Foundation-Model/src/llama_recipes/configs/model_config.json"
+elif [ "$MODEL_SIZE" = "309M" ]; then
+    PRETRAINED_CKPT="/hpcwork/yh522379/moonbeam/checkpoints/pre-trained/moonbeam_309M.pt"
+    MODEL_NAME="maestro"
+    DATASET_NAME="maestro_309M"
+    MODEL_CONFIG_PATH="Moonbeam-MIDI-Foundation-Model/src/llama_recipes/configs/model_config_small.json"
+else
+    echo "Error: MODEL_SIZE must be either '839M' or '309M'"
+    exit 1
+fi
+
+# Params
+NUM_EPOCHS="100"
+
+SCHEDULER_TYPE="cosine"
+LR="1e-4"
+GAMMA="0.99"
+
+GEN_MAX_LEN="512"
+GEN_SAMPLES="200"
+GEN_TEMPERATURE="1.1"
+ENABLE_GENERATION="False"
+ENABLE_EVALUATION="False"
+EVALUATION_FREQUENCY_EPOCHS="1"
+
+CONTEXT_LENGTH="512"
+WEIGHT_DECAY="0"
+BATCH_SIZE="32"
+
+USE_PEFT="True"
+PEFT_METHOD="lora"
+
+GRADIENT_ACCUMULATION_STEPS="1"
+GRADIENT_CLIPPING="True"
+
+# Auto-generate names based on configuration
+TIMESTAMP=$(date +%Y%m%d_%H%M%S)
+if [ "$USE_PEFT" = "True" ]; then
+    PEFT_STRING="peft"
+else
+    PEFT_STRING="full"
+fi
+# Add weight decay to name only if it's not 0.0
+if [ "$WEIGHT_DECAY" != "0.0" ] && [ "$WEIGHT_DECAY" != "0" ]; then
+    WD_STRING="_wd${WEIGHT_DECAY}"
+else
+    WD_STRING=""
+fi
+CONFIG_STRING="${MODEL_SIZE}_${PEFT_STRING}_ctx${CONTEXT_LENGTH}_bs${BATCH_SIZE}_gr_acc${GRADIENT_ACCUMULATION_STEPS}_clip${GRADIENT_CLIPPING}_lr${LR}_${SCHEDULER_TYPE}_gamma${GAMMA}${WD_STRING}_temp${GEN_TEMPERATURE}_ep${NUM_EPOCHS}"
+OUTPUT_DIR="$HPCWORK/moonbeam/checkpoints/fine-tuned/ft_${CONFIG_STRING}_${TIMESTAMP}"
+WANDB_NAME="ft_${CONFIG_STRING}_${TIMESTAMP}"
+
+
+mkdir -p logs
+
+# Run the fine-tuning script
+torchrun --nnodes 1 --nproc_per_node 1 Moonbeam-MIDI-Foundation-Model/recipes/finetuning/real_finetuning_uncon_gen.py \
+  --lr "$LR" \
+  --weight_decay "$WEIGHT_DECAY" \
+  --val_batch_size "$BATCH_SIZE" \
+  --run_validation True \
+  --validation_interval 20 \
+  --save_metrics True \
+  --dist_checkpoint_root_folder "$OUTPUT_DIR" \
+  --dist_checkpoint_folder ddp \
+  --trained_checkpoint_path "$PRETRAINED_CKPT" \
+  --pure_bf16 True \
+  --enable_ddp True \
+  --use_peft "$USE_PEFT" \
+  --peft_method "$PEFT_METHOD" \
+  --quantization False \
+  --model_name "$MODEL_NAME" \
+  --dataset "$DATASET_NAME" \
+  --output_dir "$OUTPUT_DIR" \
+  --batch_size_training "$BATCH_SIZE" \
+  --context_length "$CONTEXT_LENGTH" \
+  --num_epochs "$NUM_EPOCHS" \
+  --use_wandb True \
+  --gamma "$GAMMA" \
+  --scheduler_type "$SCHEDULER_TYPE" \
+  --model_config_path "$MODEL_CONFIG_PATH" \
+  --enable_generation "$ENABLE_GENERATION" \
+  --generation_save_dir "$OUTPUT_DIR/generation_results" \
+  --generation_temperature "$GEN_TEMPERATURE" \
+  --generation_top_p 0.95 \
+  --generation_max_gen_len "$GEN_MAX_LEN" \
+  --generation_num_samples "$GEN_SAMPLES" \
+  --generation_mode from_scratch \
+  --enable_evaluation "$ENABLE_EVALUATION" \
+  --evaluation_ref_dir "/hpcwork/lect0148/experiments/mistral-155M_remi_maestro_v8/output/subset/train" \
+  --wandb_name "$WANDB_NAME" \
+  --evaluation_frequency_epochs "$EVALUATION_FREQUENCY_EPOCHS" \
+  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
+  -gradient_clipping "$GRADIENT_CLIPPING" \
diff --git a/slurm_scripts/inference-large-batch.sh b/slurm_scripts/inference-large-batch.sh
new file mode 100755
index 0000000..f1b3473
--- /dev/null
+++ b/slurm_scripts/inference-large-batch.sh
@@ -0,0 +1,48 @@
+#!/bin/bash
+#SBATCH --export=ALL
+#SBATCH --job-name=moonbeam-ft
+#SBATCH --output=logs/moonbeam_inference_%j.out
+#SBATCH --error=logs/moonbeam_inference_%j.err
+#SBATCH --nodes=1
+#SBATCH --ntasks-per-node=1
+#SBATCH --cpus-per-gpu=24
+#SBATCH --gres=gpu:1
+#SBATCH --time=00:05:00
+#SBATCH --partition=c23g
+#SBATCH --account=lect0148
+
+module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.3
+export PYTHONPATH=$(pwd)/src:$(pwd)/Moonbeam-MIDI-Foundation-Model/src:$PYTHONPATH
+source .venv-moonbeam/bin/activate
+
+# Configurable variables for comparison experiments
+CSV_FILE="preprocessed/839M/train_test_split.csv"
+TOP_P=0.95
+TEMPERATURE=1.1
+MODEL_CONFIG="src/llama_recipes/configs/model_config.json"
+CKPT_DIR="/hpcwork/yh522379/moonbeam/checkpoints/pre-trained/moonbeam_839M.pt"
+TOKENIZER_PATH="tokenizer.model"
+PEFT_WEIGHT="/hpcwork/yh522379/moonbeam/checkpoints/fine-tuned/ft_839M_peft_ctx512_bs32_lr1e-4_cosine_gamma0.99_temp1.1_ep150_20250803_045343/137-20.safetensors"
+
+MAX_SEQ_LEN=1024
+MAX_GEN_LEN=512
+MAX_BATCH_SIZE=4
+NUM_SAMPLES=20  # Increased for high-fidelity comparison
+PROMPT_LEN=512
+GENERATION_MODE="random_files"  # "from_scratch", "random_files", or "all_test_files"
+
+# Run the inference script
+torchrun --nproc_per_node=1 recipes/inference/custom_music_generation/unconditional_music_generation.py \
+  --csv_file "$CSV_FILE" \
+  --top_p "$TOP_P" \
+  --temperature "$TEMPERATURE" \
+  --model_config_path "$MODEL_CONFIG" \
+  --ckpt_dir "$CKPT_DIR" \
+  --finetuned_PEFT_weight_path "$PEFT_WEIGHT" \
+  --tokenizer_path "$TOKENIZER_PATH" \
+  --max_seq_len "$MAX_SEQ_LEN" \
+  --max_gen_len "$MAX_GEN_LEN" \
+  --max_batch_size "$MAX_BATCH_SIZE" \
+  --num_samples "$NUM_SAMPLES" \
+  --prompt_len "$PROMPT_LEN" \
+  --generation_mode "$GENERATION_MODE"
\ No newline at end of file
diff --git a/slurm_scripts/inference-large.sh b/slurm_scripts/inference-large.sh
new file mode 100755
index 0000000..01c46b9
--- /dev/null
+++ b/slurm_scripts/inference-large.sh
@@ -0,0 +1,37 @@
+#!/bin/bash
+
+module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.3
+export PYTHONPATH=$(pwd)/src:$(pwd)/Moonbeam-MIDI-Foundation-Model/src:$PYTHONPATH
+source .venv-moonbeam/bin/activate
+
+# Configurable variables for comparison experiments
+CSV_FILE="preprocessed/839M/train_test_split.csv"
+TOP_P=0.95
+TEMPERATURE=1.1
+MODEL_CONFIG="src/llama_recipes/configs/model_config.json"
+CKPT_DIR="/hpcwork/yh522379/moonbeam/checkpoints/pre-trained/moonbeam_839M.pt"
+TOKENIZER_PATH="tokenizer.model"
+PEFT_WEIGHT="/hpcwork/yh522379/moonbeam/checkpoints/fine-tuned/ft_839M_peft_ctx1024_bs16_gradacc4_clipTrue_clipthresh1.0_lr1e-4_cosine_gamma0.99_temp1.1_ep100_20250807_113308/18-20.safetensors"
+
+MAX_SEQ_LEN=1024
+MAX_GEN_LEN=512
+MAX_BATCH_SIZE=4
+NUM_SAMPLES=20  # Increased for high-fidelity comparison
+PROMPT_LEN=512
+GENERATION_MODE="random_files"  # "from_scratch", "random_files", or "all_test_files"
+
+# Run the inference script
+torchrun --nproc_per_node=1 recipes/inference/custom_music_generation/unconditional_music_generation.py \
+  --csv_file "$CSV_FILE" \
+  --top_p "$TOP_P" \
+  --temperature "$TEMPERATURE" \
+  --model_config_path "$MODEL_CONFIG" \
+  --ckpt_dir "$CKPT_DIR" \
+  --finetuned_PEFT_weight_path "$PEFT_WEIGHT" \
+  --tokenizer_path "$TOKENIZER_PATH" \
+  --max_seq_len "$MAX_SEQ_LEN" \
+  --max_gen_len "$MAX_GEN_LEN" \
+  --max_batch_size "$MAX_BATCH_SIZE" \
+  --num_samples "$NUM_SAMPLES" \
+  --prompt_len "$PROMPT_LEN" \
+  --generation_mode "$GENERATION_MODE"
\ No newline at end of file
diff --git a/slurm_scripts/inference-parameter-sweep.sh b/slurm_scripts/inference-parameter-sweep.sh
new file mode 100644
index 0000000..090aca3
--- /dev/null
+++ b/slurm_scripts/inference-parameter-sweep.sh
@@ -0,0 +1,84 @@
+#!/bin/bash
+#SBATCH --export=ALL
+#SBATCH --job-name=moonbeam-param-sweep
+#SBATCH --output=logs/moonbeam_param_sweep_%j.out
+#SBATCH --error=logs/moonbeam_param_sweep_%j.err
+#SBATCH --nodes=1
+#SBATCH --ntasks-per-node=1
+#SBATCH --cpus-per-gpu=24
+#SBATCH --gres=gpu:1
+#SBATCH --time=01:00:00
+#SBATCH --partition=c23g
+#SBATCH --account=lect0148
+
+module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.3
+export PYTHONPATH=$(pwd)/src:$(pwd)/Moonbeam-MIDI-Foundation-Model/src:$PYTHONPATH
+source .venv-moonbeam/bin/activate
+
+# Base configuration
+CSV_FILE="preprocessed/839M/train_test_split.csv"
+MODEL_CONFIG="src/llama_recipes/configs/model_config.json"
+CKPT_DIR="/hpcwork/yh522379/moonbeam/checkpoints/pre-trained/moonbeam_839M.pt"
+TOKENIZER_PATH="tokenizer.model"
+PEFT_WEIGHT="/hpcwork/yh522379/moonbeam/checkpoints/fine-tuned/ft_839M_peft_ctx512_bs32_lr1e-4_cosine_gamma0.99_temp1.1_ep150_20250803_045343/137-20.safetensors"
+
+MAX_SEQ_LEN=1024
+MAX_GEN_LEN=512
+MAX_BATCH_SIZE=4
+NUM_SAMPLES=15  # Small number for quick parameter exploration
+PROMPT_LEN=512
+GENERATION_MODE="random_files"
+
+# Parameter arrays for grid search
+TOP_P_VALUES=(0.6 0.7 0.8 0.9 1.0)
+TEMPERATURE_VALUES=(0.7 0.8 0.9 1.0 1.1)
+
+echo "Starting parameter sweep at $(date)"
+echo "Total combinations: $((${#TOP_P_VALUES[@]} * ${#TEMPERATURE_VALUES[@]}))"
+echo "Samples per combination: $NUM_SAMPLES"
+echo "=========================================="
+
+# Counter for progress tracking
+COMBO_COUNT=0
+TOTAL_COMBOS=$((${#TOP_P_VALUES[@]} * ${#TEMPERATURE_VALUES[@]}))
+
+# Loop through all combinations
+for TOP_P in "${TOP_P_VALUES[@]}"; do
+    for TEMPERATURE in "${TEMPERATURE_VALUES[@]}"; do
+        COMBO_COUNT=$((COMBO_COUNT + 1))
+        
+        echo ""
+        echo "[$COMBO_COUNT/$TOTAL_COMBOS] Processing combination: top_p=$TOP_P, temperature=$TEMPERATURE"
+        echo "Started at: $(date)"
+        
+        # Run the inference script
+        torchrun --nproc_per_node=1 recipes/inference/custom_music_generation/unconditional_music_generation.py \
+          --csv_file "$CSV_FILE" \
+          --top_p "$TOP_P" \
+          --temperature "$TEMPERATURE" \
+          --model_config_path "$MODEL_CONFIG" \
+          --ckpt_dir "$CKPT_DIR" \
+          --finetuned_PEFT_weight_path "$PEFT_WEIGHT" \
+          --tokenizer_path "$TOKENIZER_PATH" \
+          --max_seq_len "$MAX_SEQ_LEN" \
+          --max_gen_len "$MAX_GEN_LEN" \
+          --max_batch_size "$MAX_BATCH_SIZE" \
+          --num_samples "$NUM_SAMPLES" \
+          --prompt_len "$PROMPT_LEN" \
+          --generation_mode "$GENERATION_MODE"
+        
+        if [ $? -eq 0 ]; then
+            echo "✅ Combination $COMBO_COUNT completed successfully"
+        else
+            echo "❌ Combination $COMBO_COUNT failed"
+        fi
+        
+        echo "Finished at: $(date)"
+        echo "----------------------------------------"
+    done
+done
+
+echo ""
+echo "🎵 Parameter sweep completed at $(date)"
+echo "Generated files are organized by parameter values in the checkpoint directory"
+echo "Use the post-process script to validate and analyze the results!"
\ No newline at end of file
diff --git a/slurm_scripts/inference-small-batch.sh b/slurm_scripts/inference-small-batch.sh
new file mode 100755
index 0000000..713996d
--- /dev/null
+++ b/slurm_scripts/inference-small-batch.sh
@@ -0,0 +1,49 @@
+#!/bin/bash
+#SBATCH --export=ALL
+#SBATCH --job-name=moonbeam-ft
+#SBATCH --output=logs/moonbeam_inference_%j.out
+#SBATCH --error=logs/moonbeam_inference_%j.err
+#SBATCH --nodes=1
+#SBATCH --ntasks-per-node=1
+#SBATCH --cpus-per-gpu=24
+#SBATCH --gres=gpu:1
+#SBATCH --time=00:30:00
+#SBATCH --partition=c23g
+#SBATCH --account=lect0148
+
+module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.3
+export PYTHONPATH=$(pwd)/src:$PYTHONPATH
+source .venv-moonbeam/bin/activate
+
+# Define variables
+CSV_FILE="preprocessed/309M/train_test_split.csv"
+TOP_P=0.95
+TEMPERATURE=1
+MODEL_CONFIG="src/llama_recipes/configs/model_config_small.json"
+CKPT_DIR="/hpcwork/yh522379/moonbeam/checkpoints/pre-trained/moonbeam_309M.pt"
+TOKENIZER_PATH="tokenizer.model"
+PEFT_WEIGHT="/hpcwork/yh522379/moonbeam/checkpoints/fine-tuned/309M-50epoch/49-20.safetensors"
+
+MAX_SEQ_LEN=1024
+MAX_GEN_LEN=256
+MAX_BATCH_SIZE=4
+NUM_SAMPLES=1300
+PROMPT_LEN=100
+FROM_SCRATCH=True
+
+# Run the inference script
+torchrun --nproc_per_node=1 recipes/inference/custom_music_generation/unconditional_music_generation.py \
+  --csv_file "$CSV_FILE" \
+  --top_p "$TOP_P" \
+  --temperature "$TEMPERATURE" \
+  --model_config_path "$MODEL_CONFIG" \
+  --ckpt_dir "$CKPT_DIR" \
+  --finetuned_PEFT_weight_path "$PEFT_WEIGHT" \
+  --tokenizer_path "$TOKENIZER_PATH" \
+  --max_seq_len "$MAX_SEQ_LEN" \
+  --max_gen_len "$MAX_GEN_LEN" \
+  --max_batch_size "$MAX_BATCH_SIZE" \
+  --num_samples "$NUM_SAMPLES" \
+  --prompt_len "$PROMPT_LEN" \
+  --from-scratch "$FROM_SCRATCH" \
+  --folder "$FOLDER"
\ No newline at end of file
diff --git a/slurm_scripts/inference-small.sh b/slurm_scripts/inference-small.sh
new file mode 100755
index 0000000..a087eba
--- /dev/null
+++ b/slurm_scripts/inference-small.sh
@@ -0,0 +1,35 @@
+#!/bin/bash
+export PYTHONPATH=$(pwd)/src:$PYTHONPATH
+
+# Define variables
+CSV_FILE="preprocessed/309M/train_test_split.csv"
+TOP_P=0.95
+TEMPERATURE=1.1
+MODEL_CONFIG="src/llama_recipes/configs/model_config_small.json"
+CKPT_DIR="/hpcwork/yh522379/moonbeam/checkpoints/pre-trained/moonbeam_309M.pt"
+TOKENIZER_PATH="tokenizer.model"
+PEFT_WEIGHT="/hpcwork/yh522379/moonbeam/checkpoints/fine-tuned/309M-50epoch/49-20.safetensors"
+
+MAX_SEQ_LEN=1024
+MAX_GEN_LEN=256
+MAX_BATCH_SIZE=4
+NUM_SAMPLES=600
+PROMPT_LEN=256
+FROM_SCRATCH=False
+
+# Run the inference script
+torchrun --nproc_per_node=1 recipes/inference/custom_music_generation/unconditional_music_generation.py \
+  --csv_file "$CSV_FILE" \
+  --top_p "$TOP_P" \
+  --temperature "$TEMPERATURE" \
+  --model_config_path "$MODEL_CONFIG" \
+  --ckpt_dir "$CKPT_DIR" \
+  --finetuned_PEFT_weight_path "$PEFT_WEIGHT" \
+  --tokenizer_path "$TOKENIZER_PATH" \
+  --max_seq_len "$MAX_SEQ_LEN" \
+  --max_gen_len "$MAX_GEN_LEN" \
+  --max_batch_size "$MAX_BATCH_SIZE" \
+  --num_samples "$NUM_SAMPLES" \
+  --prompt_len "$PROMPT_LEN" \
+  --from-scratch "$FROM_SCRATCH" \
+  --folder "$FOLDER"
\ No newline at end of file
diff --git a/slurm_scripts/preprocess.sh b/slurm_scripts/preprocess.sh
new file mode 100755
index 0000000..3e967de
--- /dev/null
+++ b/slurm_scripts/preprocess.sh
@@ -0,0 +1,18 @@
+#!/bin/bash
+
+export PYTHONPATH=$(pwd)/src:$PYTHONPATH
+
+DATASET_NAME="maestro"
+DATASET_FOLDER="data/maestro"
+OUTPUT_FOLDER="preprocessed/839M"
+TRAIN_TEST_SPLIT="data/maestro/maestro-v3.0.0.csv"
+
+python data_preprocess.py \
+  --dataset_name "$DATASET_NAME" \
+  --dataset_folder "$DATASET_FOLDER" \
+  --output_folder "$OUTPUT_FOLDER" \
+  --model_config src/llama_recipes/configs/model_config.json \
+  --train_test_split_file "$TRAIN_TEST_SPLIT" \
+  --ts_threshold None
+  # --train_ratio 0.9 \ 
+  
\ No newline at end of file
diff --git a/src/llama_recipes/configs/datasets.py b/src/llama_recipes/configs/datasets.py
index 59ed26c..041f26e 100644
--- a/src/llama_recipes/configs/datasets.py
+++ b/src/llama_recipes/configs/datasets.py
@@ -35,11 +35,27 @@ class custom_dataset:
 
 @dataclass
 class lakhmidi_dataset:
-    dataset: str = "lakhmidi_dataset"
+    dataset: str = "maestro"
     train_split: str = "train"
     test_split: str = "test"
-    data_dir: str = "/PATH/TO/DATA/DIR"
-    csv_file: str = "/PATH/TO/CSV"
+    data_dir: str = "Moonbeam-MIDI-Foundation-Model/preprocessed/maestro/"
+    csv_file: str = "Moonbeam-MIDI-Foundation-Model/preprocessed/maestro/train_test_split.csv"
+
+@dataclass
+class maestro_309M:
+    dataset: str = "maestro_309M"
+    train_split: str = "train"
+    test_split: str = "test"
+    data_dir: str = "Moonbeam-MIDI-Foundation-Model/preprocessed/309M/"
+    csv_file: str = "Moonbeam-MIDI-Foundation-Model/preprocessed/309M/train_test_split.csv"
+
+@dataclass
+class maestro_839M:
+    dataset: str = "maestro_839M"
+    train_split: str = "train"
+    test_split: str = "test"
+    data_dir: str = "Moonbeam-MIDI-Foundation-Model/preprocessed/839M/"
+    csv_file: str = "Moonbeam-MIDI-Foundation-Model/preprocessed/839M/train_test_split.csv"
 
 @dataclass
 class merge_dataset:
diff --git a/src/llama_recipes/configs/training.py b/src/llama_recipes/configs/training.py
index 71e7869..a97bcc9 100644
--- a/src/llama_recipes/configs/training.py
+++ b/src/llama_recipes/configs/training.py
@@ -22,10 +22,11 @@ class train_config:
     num_epochs: int=3
     max_train_step: int=0
     max_eval_step: int=0
-    num_workers_dataloader: int=1
+    num_workers_dataloader: int=4
     lr: float=1e-4
     weight_decay: float=0.0
     gamma: float= 0.85
+    scheduler_type: str = "steplr" # Learning rate scheduler: "steplr" or "cosine"
     seed: int=42
     use_fp16: bool=False
     mixed_precision: bool=True
@@ -50,3 +51,19 @@ class train_config:
     flop_counter_start: int = 3 # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
     use_profiler: bool = False # Enable pytorch profiler, can not be used with flop counter at the same time.
     profiler_dir: str = "PATH/to/save/profiler/results" # will be used if using profiler
+
+    # Generation parameters
+    enable_generation: bool = False # Enable music generation during evaluation
+    generation_temperature: float = 1 # Temperature for generation sampling
+    generation_top_p: float = 0.95 # Top-p value for nucleus sampling
+    generation_max_gen_len: int = 256 # Maximum generation length
+    generation_prompt_len: int = 512 # Length of prompt for generation (when using data for prompts)
+    generation_num_samples: int = 20 # Number of samples to generate during evaluation
+    generation_mode: str = "random_files" # Generation mode: "from_scratch", "random_files", or "all_test_files"
+    generation_max_prompt_samples: int = 200 # Max number of prompts to sample for prompt-based generation (will duplicate if fewer test files)
+    generation_save_dir: str = "PATH/to/save/generation/results" # Directory to save generated music
+    
+    # Evaluation parameters
+    enable_evaluation: bool = False # Enable evaluation against training set after generation
+    evaluation_frequency_epochs: int = 5 # Frequency of evaluation in epochs (1 = every epoch, 5 = every 5 epochs)
+    evaluation_ref_dir: str = "PATH/to/training/data" # Reference directory containing training MIDI files
diff --git a/src/llama_recipes/configs/wandb.py b/src/llama_recipes/configs/wandb.py
index fb828fc..c8db897 100644
--- a/src/llama_recipes/configs/wandb.py
+++ b/src/llama_recipes/configs/wandb.py
@@ -6,8 +6,8 @@ from dataclasses import dataclass, field
 
 @dataclass
 class wandb_config:
-    project: str = 'Unconditional_Generation' # wandb project name
-    entity: Optional[str] = None # wandb entity name
+    project: str = 'piano-transformer' # wandb project name
+    entity: Optional[str] = 'jonathanlehmkuhl-rwth-aachen-university' # wandb entity name
     job_type: Optional[str] = None
     tags: Optional[List[str]] = None
     group: Optional[str] = None
diff --git a/src/llama_recipes/finetuning.py b/src/llama_recipes/finetuning.py
index 09899fa..0c3fc70 100644
--- a/src/llama_recipes/finetuning.py
+++ b/src/llama_recipes/finetuning.py
@@ -16,7 +16,7 @@ from torch.distributed.fsdp import (
 
 from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
 from torch.nn.parallel import DistributedDataParallel as DDP
-from torch.optim.lr_scheduler import StepLR
+from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
 from transformers import (
     AutoTokenizer,
     LlamaForCausalLM,
@@ -303,7 +303,19 @@ def main(**kwargs):
         starting_epoch, starting_step = 0, 0
 
     
+    # Create learning rate scheduler based on config
+    if train_config.scheduler_type.lower() == "cosine":
+        scheduler = CosineAnnealingLR(
+            optimizer, 
+            T_max=train_config.num_epochs,
+            eta_min=train_config.lr * 0.01  # End at 1% of initial LR
+        )
+        print(f"Using CosineAnnealingLR scheduler (T_max={train_config.num_epochs}, eta_min={train_config.lr * 0.01:.2e})")
+    elif train_config.scheduler_type.lower() == "steplr":
         scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
+        print(f"Using StepLR scheduler (step_size=1, gamma={train_config.gamma})")
+    else:
+        raise ValueError(f"Unsupported scheduler_type: {train_config.scheduler_type}. Must be 'steplr' or 'cosine'")
 
     # Start the training process
     results = train(
diff --git a/src/llama_recipes/overfitting_test.py b/src/llama_recipes/overfitting_test.py
index 469ac93..a3a67b0 100644
--- a/src/llama_recipes/overfitting_test.py
+++ b/src/llama_recipes/overfitting_test.py
@@ -16,7 +16,7 @@ from torch.distributed.fsdp import (
 
 from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
 from torch.nn.parallel import DistributedDataParallel as DDP
-from torch.optim.lr_scheduler import StepLR
+from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
 from transformers import (
     AutoTokenizer,
     LlamaForCausalLM, 
@@ -324,7 +324,19 @@ def main(**kwargs):
             lr=train_config.lr,
             weight_decay=train_config.weight_decay,
         )
+    # Create learning rate scheduler based on config
+    if train_config.scheduler_type.lower() == "cosine":
+        scheduler = CosineAnnealingLR(
+            optimizer, 
+            T_max=train_config.num_epochs,
+            eta_min=train_config.lr * 0.01  # End at 1% of initial LR
+        )
+        print(f"Using CosineAnnealingLR scheduler (T_max={train_config.num_epochs}, eta_min={train_config.lr * 0.01:.2e})")
+    elif train_config.scheduler_type.lower() == "steplr":
         scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
+        print(f"Using StepLR scheduler (step_size=1, gamma={train_config.gamma})")
+    else:
+        raise ValueError(f"Unsupported scheduler_type: {train_config.scheduler_type}. Must be 'steplr' or 'cosine'")
 
     # Start the training process
     batch = next(iter(train_dataloader))
diff --git a/src/llama_recipes/real_finetuning_uncon_gen.py b/src/llama_recipes/real_finetuning_uncon_gen.py
index 9068a11..10dd056 100644
--- a/src/llama_recipes/real_finetuning_uncon_gen.py
+++ b/src/llama_recipes/real_finetuning_uncon_gen.py
@@ -13,7 +13,7 @@ from torch.distributed.fsdp import (
 
 from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
 from torch.nn.parallel import DistributedDataParallel as DDP
-from torch.optim.lr_scheduler import StepLR
+from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
 from transformers import (
     AutoTokenizer,
     LlamaForCausalLM,
@@ -50,6 +50,9 @@ from llama_recipes.utils.train_utils import (
 )
 from accelerate.utils import is_xpu_available
 
+def is_distributed_env():
+    return all(var in os.environ for var in ["RANK", "WORLD_SIZE", "LOCAL_RANK"])
+
 def setup_wandb(train_config, fsdp_config, llama_config, **kwargs):
     try:
         import wandb
@@ -62,6 +65,11 @@ def setup_wandb(train_config, fsdp_config, llama_config, **kwargs):
     wandb_config = WANDB_CONFIG()
     update_config(wandb_config, **kwargs)
     init_dict = dataclasses.asdict(wandb_config)
+    
+    wandb_name = kwargs.get("wandb_name")
+    if wandb_name:
+        init_dict['name'] = wandb_name
+
     run = wandb.init(**init_dict)
     run.config.update(train_config)
     run.config.update(fsdp_config, allow_val_change=True)
@@ -85,10 +93,22 @@ def setup_wandb(train_config, fsdp_config, llama_config, **kwargs):
     return run
 
 
+def print_mem(stage):
+    if torch.cuda.is_available():
+        print(f"\n[DEBUG][{stage}] CUDA memory summary:")
+        print(torch.cuda.memory_summary())
+    elif hasattr(torch, "xpu") and torch.xpu.is_available():
+        print(f"\n[DEBUG][{stage}] XPU memory summary:")
+        print(torch.xpu.memory_summary())
+    else:
+        print(f"\n[DEBUG][{stage}] No GPU available.")
+
 def main(**kwargs):
+    # Extract model config path
+    model_config_path = kwargs.pop("model_config_path", "Moonbeam-MIDI-Foundation-Model/src/llama_recipes/configs/model_config.json")
     # Update the configuration for the training and sharding process
     train_config, fsdp_config, ddp_config = TRAIN_CONFIG(), FSDP_CONFIG(), DDP_CONFIG()
-    model_config_path = "src/llama_recipes/configs/model_config.json"
+    # model_config_path = "src/llama_recipes/configs/model_config.json"
     update_config((train_config, fsdp_config, ddp_config), **kwargs)
     print("updated training config", train_config)
     # Set the seeds for reproducibility
@@ -97,12 +117,21 @@ def main(**kwargs):
     torch.manual_seed(train_config.seed)
     random.seed(train_config.seed)
 
+    if (train_config.enable_fsdp or train_config.enable_ddp) and not is_distributed_env():
+        print("[INFO] Distributed environment not detected — disabling FSDP and DDP for debugging.")
+        train_config.enable_fsdp = False
+        train_config.enable_ddp = False
+
     if train_config.enable_fsdp or train_config.enable_ddp:
         setup() #enable nccl / ccl
         # torchrun specific
         local_rank = int(os.environ["LOCAL_RANK"])
         rank = int(os.environ["RANK"])
         world_size = int(os.environ["WORLD_SIZE"])
+    else:
+        local_rank = 0
+        rank = 0
+        world_size = 1
 
     if torch.distributed.is_initialized():
         if is_xpu_available():
@@ -143,9 +172,16 @@ def main(**kwargs):
         print(f"model_config:{llama_config}")
         model = LlamaForCausalLM(llama_config)
         
-        model_checkpoint = torch.load(train_config.trained_checkpoint_path)
+        # Debug: Check if model.config has the required attributes
+        print(f"Model config after creation - onset_vocab_size: {getattr(model.config, 'onset_vocab_size', 'MISSING')}")
+        print(f"Model config type: {type(model.config)}")
+        print(f"LlamaConfig type: {type(llama_config)}")
+        print(f"Are they the same object? {model.config is llama_config}")
 
+        model_checkpoint = torch.load(train_config.trained_checkpoint_path)    
         checkpoint = model_checkpoint['model_state_dict']
+        # checkpoint = torch.load(train_config.trained_checkpoint_path, weights_only=True)
+
         new_state_dict = {}
         for k, v in checkpoint.items():
             if k.startswith('module.'): # Check if the keys have 'module.' prefix and remove it if necessary
@@ -163,6 +199,7 @@ def main(**kwargs):
 
     # Load the tokenizer and add special tokens
     tokenizer = MusicTokenizer(timeshift_vocab_size = llama_config.onset_vocab_size, dur_vocab_size = llama_config.dur_vocab_size, octave_vocab_size = llama_config.octave_vocab_size, pitch_class_vocab_size = llama_config.pitch_class_vocab_size, instrument_vocab_size = llama_config.instrument_vocab_size, velocity_vocab_size = llama_config.velocity_vocab_size, sos_token = llama_config.sos_token, eos_token = llama_config.eos_token, pad_token = llama_config.pad_token)
+    print_mem("After tokenizer load")
 
     dataset_config = generate_dataset_config(train_config, kwargs)
 
@@ -243,6 +280,7 @@ def main(**kwargs):
         dataset_config,
         split="train",
     )
+    print_mem("After train dataset load")
 
     if not train_config.enable_fsdp or rank == 0:
         print(f"--> Training Set Length = {len(dataset_train)}")
@@ -250,8 +288,9 @@ def main(**kwargs):
     dataset_val = get_preprocessed_dataset(
         tokenizer,
         dataset_config,
-        split="test",
+        split="validation",
     )
+    print_mem("After val dataset load")
     if train_config.batching_strategy == "packing":
         dataset_train = ConcatDataset_hybrid_padding_concatenating(dataset_train, chunk_size=train_config.context_length, split="train",data_dir = dataset_config.data_dir)
 
@@ -264,6 +303,7 @@ def main(**kwargs):
         pin_memory=True,
         **train_dl_kwargs,
     )
+    print_mem("After train DataLoader creation")
 
     eval_dataloader = None
     if train_config.run_validation:
@@ -278,6 +318,7 @@ def main(**kwargs):
             pin_memory=True,
             **val_dl_kwargs,
         )
+        print_mem("After val DataLoader creation")
 
     # Initialize the optimizer and learning rate scheduler
     if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
@@ -295,10 +336,23 @@ def main(**kwargs):
             lr=train_config.lr,
             weight_decay=train_config.weight_decay,
         )
+    print_mem("After optimizer creation")
 
     starting_epoch, starting_step = 0, 0
 
+    # Create learning rate scheduler based on config
+    if train_config.scheduler_type.lower() == "cosine":
+        scheduler = CosineAnnealingLR(
+            optimizer, 
+            T_max=train_config.num_epochs,
+            eta_min=1e-5
+        )
+        print(f"Using CosineAnnealingLR scheduler (T_max={train_config.num_epochs}, eta_min={train_config.lr * 0.01:.2e})")
+    elif train_config.scheduler_type.lower() == "steplr":
         scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
+        print(f"Using StepLR scheduler (step_size=1, gamma={train_config.gamma})")
+    else:
+        raise ValueError(f"Unsupported scheduler_type: {train_config.scheduler_type}. Must be 'steplr' or 'cosine'")
     print("check model trainable parameters")
     total_trainable = 0
     for name, param in model.named_parameters():
@@ -326,6 +380,7 @@ def main(**kwargs):
         rank if (train_config.enable_fsdp or train_config.enable_ddp) else None,
         wandb_run,
     )
+    print_mem("After training loop")
     if not train_config.enable_fsdp or rank==0:
         [print(f'Key: {k}, Value: {v}') for k, v in results.items()]
         if train_config.use_wandb:
diff --git a/src/llama_recipes/utils/dataset_utils.py b/src/llama_recipes/utils/dataset_utils.py
index 91c30e8..2e68687 100644
--- a/src/llama_recipes/utils/dataset_utils.py
+++ b/src/llama_recipes/utils/dataset_utils.py
@@ -61,7 +61,9 @@ DATASET_PREPROC = {
     "lakhmidi_dataset": get_lakhmidi_dataset,
     "merge_dataset": get_merge_dataset,
     "emophia_con_gen_dataset": get_emophia_con_gen_dataset,
-    "commu_con_gen_dataset": get_commu_con_gen_dataset
+    "commu_con_gen_dataset": get_commu_con_gen_dataset,
+    "maestro_839M":  get_lakhmidi_dataset,
+    "maestro_309M": get_lakhmidi_dataset
 }
 
 
diff --git a/src/llama_recipes/utils/metrics.py b/src/llama_recipes/utils/metrics.py
new file mode 100644
index 0000000..35f5a6f
--- /dev/null
+++ b/src/llama_recipes/utils/metrics.py
@@ -0,0 +1,613 @@
+import os
+import glob
+import numpy as np
+import traceback
+from scipy.special import rel_entr
+from scipy.stats import gaussian_kde
+from miditok import REMI
+from symusic import Score
+from piano_transformer.mgeval import core, utils
+from tqdm import tqdm
+from sklearn.model_selection import LeaveOneOut
+import copy
+import seaborn as sns
+import matplotlib.pyplot as plt
+import pandas as pd
+import pretty_midi
+import pickle
+import shutil
+import hashlib
+import random
+
+
+def get_mgeval_features(num_samples):
+    set_eval_init = {
+        "total_used_pitch": np.zeros((num_samples, 1)),
+        "total_pitch_class_histogram": np.zeros((num_samples, 12)),
+        "pitch_class_transition_matrix": np.zeros((num_samples, 12, 12)),
+        "pitch_range": np.zeros((num_samples, 1)),
+        "avg_pitch_shift": np.zeros((num_samples, 1)),
+        "total_used_note": np.zeros((num_samples, 1)),
+        "avg_IOI": np.zeros((num_samples, 1)),
+        "note_length_hist": np.zeros((num_samples, 12)),
+        "note_length_transition_matrix": np.zeros((num_samples, 12, 12)),
+    }
+    kwargs_init = {
+        "total_used_pitch": {},
+        "total_pitch_class_histogram": {},
+        "pitch_class_transition_matrix": {"normalize": 2},
+        "pitch_range": {},
+        "avg_pitch_shift": {"track_num": 0},
+        "total_used_note": {"track_num": 0},
+        "avg_IOI": {},
+        "note_length_hist": {"track_num": 0, "normalize": True, "pause_event": False},
+        "note_length_transition_matrix": {
+            "track_num": 0,
+            "normalize": 2,
+            "pause_event": False,
+        },
+    }
+    return set_eval_init, kwargs_init
+
+
+def analyze_dataset_mgeval(dataset_path, output_path, features=None, max_samples=None):
+    print("running full function")
+    if not features:
+        features = [
+            "total_used_pitch",
+            "total_pitch_class_histogram",
+            "pitch_range",
+            "avg_pitch_shift",
+            "total_used_note",
+            "avg_IOI",
+            "note_length_hist",
+            "note_length_transition_matrix",
+        ]
+    dataset = glob.glob(os.path.join(dataset_path, "*.midi"))
+    if max_samples and len(dataset) > max_samples:
+        dataset = dataset[:max_samples]
+
+    valid_dataset = []
+    for path in dataset:
+        try:
+            midi = pretty_midi.PrettyMIDI(path)
+            if midi.instruments:  # has at least one instrument
+                valid_dataset.append(path)
+        except Exception as e:
+            print(f"Skipping {path} due to error: {e}")
+            continue
+
+    dataset = valid_dataset
+    num_samples = len(dataset)
+
+    set_eval_init, kwargs_init = get_mgeval_features(num_samples)
+    set_eval = {key: set_eval_init[key] for key in features}
+    kwargs = [kwargs_init[key] for key in features]
+    metrics_list = features
+    for j in range(len(metrics_list)):
+        for i in tqdm(range(0, num_samples), desc=f"Evaluating {metrics_list[j]}"):
+            feature = core.extract_feature(dataset[i])
+            set_eval[metrics_list[j]][i] = getattr(core.metrics(), metrics_list[j])(
+                feature, **kwargs[j]
+            )
+
+    for i in range(0, len(metrics_list)):
+        print("------------------------")
+        print(metrics_list[i] + ":")
+        print("mean: ", np.mean(set_eval[metrics_list[i]], axis=0))
+        print("std: ", np.std(set_eval[metrics_list[i]], axis=0))
+
+    # summarize_mgeval_results(set_eval, metrics_list)
+    summarize_and_plot_mgeval_results(set_eval, metrics_list, output_path)
+
+
+def comparing_pairwise_distances_mgeval(
+    dataset1_path, dataset2_path, output_path, features=None, max_samples=None
+):
+    print("running full function")
+    if not features:
+        features = [
+            "total_used_pitch",
+            "total_pitch_class_histogram",
+            "pitch_range",
+            "avg_pitch_shift",
+            "total_used_note",
+            "avg_IOI",
+            "note_length_hist",
+            "note_length_transition_matrix",
+        ]
+    dataset1 = glob.glob(os.path.join(dataset1_path, "*.midi"))
+    dataset2 = glob.glob(os.path.join(dataset2_path, "*.midi"))
+    if max_samples and len(dataset1) > max_samples:
+        dataset1 = dataset1[:max_samples]
+    if max_samples and len(dataset2) > max_samples:
+        dataset2 = dataset2[:max_samples]
+
+    # Filter valid MIDI files for dataset1
+    valid_dataset1 = []
+    for path in dataset1:
+        try:
+            midi = pretty_midi.PrettyMIDI(path)
+            if midi.instruments:
+                valid_dataset1.append(path)
+        except Exception as e:
+            print(f"Skipping {path} from dataset1: {e}")
+            continue
+    dataset1 = valid_dataset1
+
+    # Filter valid MIDI files for dataset2
+    valid_dataset2 = []
+    for path in dataset2:
+        try:
+            midi = pretty_midi.PrettyMIDI(path)
+            if midi.instruments:
+                valid_dataset2.append(path)
+        except Exception as e:
+            print(f"Skipping {path} from dataset2: {e}")
+            continue
+    dataset2 = valid_dataset2
+
+    num_samples = min(len(dataset1), len(dataset2))
+    metrics_list = features
+
+    set_eval_init, kwargs_init = get_mgeval_features(num_samples)
+    set1_eval = {key: set_eval_init[key] for key in features}
+    set2_eval = copy.deepcopy(set1_eval)
+    kwargs = [kwargs_init[key] for key in features]
+    metrics_list = features
+    for j in range(len(metrics_list)):
+        for i in tqdm(
+            range(0, num_samples), desc=f"Evaluating {metrics_list[j]} on dataset1"
+        ):
+            feature = core.extract_feature(dataset1[i])
+            set1_eval[metrics_list[j]][i] = getattr(core.metrics(), metrics_list[j])(
+                feature, **kwargs[j]
+            )
+    for j in range(len(metrics_list)):
+        for i in tqdm(
+            range(0, num_samples), desc=f"Evaluating {metrics_list[j]} on dataset2"
+        ):
+            feature = core.extract_feature(dataset2[i])
+            set2_eval[metrics_list[j]][i] = getattr(core.metrics(), metrics_list[j])(
+                feature, **kwargs[j]
+            )
+
+    loo = LeaveOneOut()
+    loo.get_n_splits(np.arange(num_samples))
+    set1_intra = np.zeros((num_samples, len(metrics_list), num_samples - 1))
+    for i in range(len(metrics_list)):
+        for train_index, test_index in tqdm(
+            loo.split(np.arange(num_samples)),
+            desc=f"Computing intra-set distances for {metrics_list[i]} on dataset1",
+        ):
+            set1_intra[test_index[0]][i] = utils.c_dist(
+                set1_eval[metrics_list[i]][test_index],
+                set1_eval[metrics_list[i]][train_index],
+            )
+
+    loo = LeaveOneOut()
+    loo.get_n_splits(np.arange(num_samples))
+    sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))
+    for i in range(len(metrics_list)):
+        for train_index, test_index in tqdm(
+            loo.split(np.arange(num_samples)),
+            desc=f"Computing inter-set distances for {metrics_list[i]} between dataset1 and dataset2",
+        ):
+            sets_inter[test_index[0]][i] = utils.c_dist(
+                set1_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]]
+            )
+
+    plot_set1_intra = np.transpose(set1_intra, (1, 0, 2)).reshape(len(metrics_list), -1)
+    plot_sets_inter = np.transpose(sets_inter, (1, 0, 2)).reshape(len(metrics_list), -1)
+    os.makedirs(output_path, exist_ok=True)
+    for i in range(0, len(metrics_list)):
+        sns.kdeplot(plot_set1_intra[i], label="intra_set1")
+        sns.kdeplot(plot_sets_inter[i], label="inter")
+        plt.title(metrics_list[i])
+        plt.xlabel("Euclidean distance")
+        plt.legend()
+        figure_path = os.path.join(output_path, f"{metrics_list[i]}_distance_plot.png")
+        plt.savefig(figure_path)
+        plt.clf()
+
+    for i in range(0, len(metrics_list)):
+        print("------------------------")
+        print(metrics_list[i] + ":")
+        print(
+            "Kullback–Leibler divergence:",
+            utils.kl_dist(plot_set1_intra[i], plot_sets_inter[i]),
+        )
+        print(
+            "Overlap area:", utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i])
+        )
+
+
+def evaluate_mgeval_combined(
+    dataset1_path,
+    dataset2_path,
+    output_path=None,
+    features=None,
+    max_samples=None,
+):
+    print("Running streamlined MGEval evaluation (relative only)...")
+
+    if not features:
+        features = [
+            "total_used_pitch",
+            "total_pitch_class_histogram",
+            "pitch_range",
+            "avg_pitch_shift",
+            "total_used_note",
+            "avg_IOI",
+            "note_length_hist",
+            "note_length_transition_matrix",
+        ]
+
+    # load and filter
+    def load_valid_dataset(dataset_path):
+        dataset = glob.glob(os.path.join(dataset_path, "**", "*.mid*"), recursive=True)
+        if max_samples and len(dataset) > max_samples:
+            dataset = dataset[:max_samples]
+        valid_dataset = []
+        for path in dataset:
+            try:
+                midi = pretty_midi.PrettyMIDI(path)
+                if midi.instruments:
+                    valid_dataset.append(path)
+            except Exception as e:
+                print(f"Skipping {path}: {e}")
+        return valid_dataset
+
+    dataset1 = load_valid_dataset(dataset1_path)
+    dataset2 = load_valid_dataset(dataset2_path)
+
+    num_samples = min(len(dataset1), len(dataset2))
+    dataset1 = dataset1[:num_samples]
+    dataset2 = dataset2[:num_samples]
+
+    if num_samples == 0:
+        print("No valid MIDI files found in at least one dataset.")
+        return
+
+    # extract features
+    set_eval_init, kwargs_init = get_mgeval_features(num_samples)
+    set1_eval = {key: set_eval_init[key] for key in features}
+    set2_eval = copy.deepcopy(set1_eval)
+    kwargs = [kwargs_init[key] for key in features]
+
+    for j in range(len(features)):
+        print(f"Extracting {features[j]}")
+        for i in tqdm(range(num_samples), desc=f"  Dataset 1"):
+            try:
+                feature = core.extract_feature(dataset1[i])
+                value = getattr(core.metrics(), features[j])(feature, **kwargs[j])
+
+                # check for NaNs in output and skip if found
+                if np.any(np.isnan(value)):
+                    raise ValueError("NaN in extracted feature")
+
+                set1_eval[features[j]][i] = value
+
+            except Exception as e:
+                print(f"[{features[j]}] Skipping {dataset1[i]} (idx={i}): {e}")
+                continue
+        for i in tqdm(range(num_samples), desc=f"  Dataset 2"):
+            try:
+                feature = core.extract_feature(dataset2[i])
+                value = getattr(core.metrics(), features[j])(feature, **kwargs[j])
+
+                # check for NaNs in output and skip if found
+                if np.any(np.isnan(value)):
+                    raise ValueError("NaN in extracted feature")
+
+                set2_eval[features[j]][i] = value
+
+            except Exception as e:
+                print(f"[{features[j]}] Skipping {dataset2[i]} (idx={i}): {e}")
+                continue
+
+    # Skip absolute evaluation for efficiency during training
+
+    # relative evaluation
+    print("\nRelative Evaluation")
+    loo = LeaveOneOut()
+    loo.get_n_splits(np.arange(num_samples))
+    set1_intra = np.zeros((num_samples, len(features), num_samples - 1))
+    sets_inter = np.zeros((num_samples, len(features), num_samples))
+
+    for i in range(len(features)):
+        for train_index, test_index in tqdm(
+            loo.split(np.arange(num_samples)), desc=f"Intra-set: {features[i]}"
+        ):
+            set1_intra[test_index[0]][i] = utils.c_dist(
+                set1_eval[features[i]][test_index], set1_eval[features[i]][train_index]
+            )
+        for train_index, test_index in tqdm(
+            loo.split(np.arange(num_samples)), desc=f"Inter-set: {features[i]}"
+        ):
+            sets_inter[test_index[0]][i] = utils.c_dist(
+                set1_eval[features[i]][test_index], set2_eval[features[i]]
+            )
+
+    plot_set1_intra = np.transpose(set1_intra, (1, 0, 2)).reshape(len(features), -1)
+    plot_sets_inter = np.transpose(sets_inter, (1, 0, 2)).reshape(len(features), -1)
+
+    # Skip plotting for efficiency during training
+
+    relative_summary = []
+
+    for i in range(len(features)):
+        try:
+            kld = utils.kl_dist(plot_set1_intra[i], plot_sets_inter[i])
+        except Exception as e:
+            print(f"[evaluate_mgeval_combined] kl_dist failed at {features[i]}: {e}")
+            kld = np.nan
+        try:
+            oa = utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i])
+        except Exception as e:
+            print(
+                f"[evaluate_mgeval_combined] overlap_area failed at {features[i]}: {e}"
+            )
+            oa = np.nan
+
+        relative_summary.append(
+            {
+                "Feature": features[i],
+                "KLD": kld,
+                "OA": oa,
+            }
+        )
+        print("------------------------")
+        print(f"{features[i]}:")
+        print("Kullback-Leibler divergence:", kld)
+        print("Overlap area:", oa)
+
+    return relative_summary
+
+
+def summarize_mgeval_results(set_eval, metrics_list):
+    summary = []
+
+    for feature in metrics_list:
+        mean_value = np.mean(set_eval[feature], axis=0)
+        std_value = np.std(set_eval[feature], axis=0)
+
+        if mean_value.ndim == 0 or mean_value.size == 1:
+            mean_scalar = float(mean_value)
+            std_scalar = float(std_value)
+        else:
+            mean_scalar = float(np.mean(mean_value))
+            std_scalar = float(np.mean(std_value))
+
+        summary.append({"Feature": feature, "Mean": mean_scalar, "Std": std_scalar})
+
+    df = pd.DataFrame(summary)
+    print(df.to_string(index=False))
+
+    return df
+
+
+def summarize_and_plot_mgeval_results(
+    set_eval, metrics_list, dataset_name, output_path=None
+):
+    summary = []
+
+    if output_path is not None:
+        os.makedirs(output_path, exist_ok=True)
+
+    for feature in metrics_list:
+        mean_value = np.mean(set_eval[feature], axis=0)
+        std_value = np.std(set_eval[feature], axis=0)
+
+        entry = {"Feature": feature}
+
+        # Scalar
+        if mean_value.ndim == 0 or mean_value.size == 1:
+            entry["Mean"] = float(mean_value)
+            entry["Std"] = float(std_value)
+
+        # Vector
+        elif mean_value.ndim == 1:
+            entry["Mean"] = mean_value
+            entry["Std"] = std_value
+
+            if feature == "total_pitch_class_histogram":
+                labels = [
+                    "C",
+                    "C#",
+                    "D",
+                    "D#",
+                    "E",
+                    "F",
+                    "F#",
+                    "G",
+                    "G#",
+                    "A",
+                    "A#",
+                    "B",
+                ]
+                plt.figure(figsize=(8, 4))
+                plt.bar(labels, mean_value)
+                plt.title(f"Pitch Class Histogram ({feature}, {dataset_name})")
+                plt.ylabel("Proportion")
+                if output_path:
+                    plt.savefig(
+                        os.path.join(
+                            output_path, "graphics", f"{feature}_{dataset_name}.png"
+                        ),
+                        bbox_inches="tight",
+                    )
+                    plt.close()
+                else:
+                    plt.show()
+
+            elif feature == "note_length_hist":
+                labels = [
+                    "Full",
+                    "Half",
+                    "Quarter",
+                    "8th",
+                    "16th",
+                    "Dot Half",
+                    "Dot Quarter",
+                    "Dot 8th",
+                    "Dot 16th",
+                    "Half Triplet",
+                    "Quarter Triplet",
+                    "8th Triplet",
+                ]
+                plt.figure(figsize=(10, 4))
+                plt.bar(labels, mean_value)
+                plt.title(f"Note Length Histogram ({feature}, {dataset_name})")
+                plt.ylabel("Proportion")
+                plt.xticks(rotation=45)
+                if output_path:
+                    plt.savefig(
+                        os.path.join(
+                            output_path, "graphics", f"{feature}_{dataset_name}.png"
+                        ),
+                        bbox_inches="tight",
+                    )
+                    plt.close()
+                else:
+                    plt.show()
+
+        # Matrix
+        elif mean_value.ndim == 2:
+            entry["Mean"] = mean_value
+            entry["Std"] = std_value
+
+            plt.figure(figsize=(8, 6))
+            sns.heatmap(mean_value, annot=False, cmap="viridis")
+            plt.title(f"Heatmap ({feature}, {dataset_name})")
+            if output_path:
+                plt.savefig(
+                    os.path.join(
+                        output_path, "graphics", f"{feature}_{dataset_name}.png"
+                    ),
+                    bbox_inches="tight",
+                )
+                plt.close()
+            else:
+                plt.show()
+
+        else:
+            raise ValueError(f"Unhandled feature shape: {feature}")
+
+        summary.append(entry)
+
+    # Print scalar results only
+    print_rows = []
+    for item in summary:
+        if np.isscalar(item["Mean"]) or (
+            isinstance(item["Mean"], np.ndarray) and item["Mean"].ndim == 0
+        ):
+            print_rows.append(
+                {
+                    "Feature": item["Feature"],
+                    "Mean": item["Mean"],
+                    "Std": item["Std"],
+                }
+            )
+
+    if print_rows:
+        print(pd.DataFrame(print_rows).to_string(index=False))
+
+    # Save full results to file
+    if output_path:
+        summary_path = os.path.join(
+            output_path, f"absolute_eval_summary_{dataset_name}.pkl"
+        )
+        pd.to_pickle(summary, summary_path)
+        print(f"\nSaved full summary.")
+
+    return summary
+
+
+def create_subset_auto_seed(input_dir, subset_size):
+    parent_dir = os.path.dirname(os.path.abspath(input_dir))
+    base_name = os.path.basename(os.path.normpath(input_dir))
+    output_dir = os.path.join(parent_dir, base_name + "_subset_" + str(subset_size))
+
+    if os.path.exists(output_dir):
+        # print(f"Deleting existing subset.")
+        # shutil.rmtree(output_dir)
+
+        print("Subset already exists.")
+        return output_dir
+
+    all_files = sorted(
+        glob.glob(os.path.join(input_dir, "**", "*.mid*"), recursive=True),
+        key=os.path.basename,
+    )
+
+    if len(all_files) < subset_size:
+        print("Subset size exceeds available files.")
+        return
+
+    # Create seed (if filenames are identical, subset will also be identical)
+    relative_names = sorted([os.path.basename(path) for path in all_files])
+    seed_input = "".join(relative_names)
+    hash_seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16) % (10**8)
+    print(f"Seed: {hash_seed}")
+
+    rng = random.Random(hash_seed)
+    subset = rng.sample(all_files, subset_size)
+
+    os.makedirs(output_dir)
+    for src in subset:
+        dst = os.path.join(output_dir, os.path.basename(src))
+        shutil.copy2(src, dst)
+
+    print(f"Created subset at {output_dir}.")
+    return output_dir
+
+
+def create_subset(input_dir, subset_size, seed=None):
+    parent_dir = os.path.dirname(os.path.abspath(input_dir))
+    base_name = os.path.basename(os.path.normpath(input_dir))
+
+    # Determine effective seed
+    if seed is None:
+        # Use hash-based deterministic seed
+        all_files = sorted(
+            glob.glob(os.path.join(input_dir, "**", "*.mid*"), recursive=True),
+            key=os.path.basename,
+        )
+        relative_names = sorted([os.path.basename(path) for path in all_files])
+        seed_input = "".join(relative_names)
+        seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16) % (10**8)
+        print(f"Generated deterministic seed: {seed}")
+    elif seed == -1:
+        # Use true randomness
+        seed = random.randint(0, 10**8)
+        print(f"Using random seed: {seed}")
+    else:
+        print(f"Using provided seed: {seed}")
+
+    output_dir = os.path.join(parent_dir, f"{base_name}_subset_{subset_size}")
+
+    if os.path.exists(output_dir):
+        print("Subset already exists.")
+        return output_dir
+
+    all_files = sorted(
+        glob.glob(os.path.join(input_dir, "**", "*.mid*"), recursive=True),
+        key=os.path.basename,
+    )
+
+    if len(all_files) < subset_size:
+        print("Subset size exceeds available files.")
+        return
+
+    rng = random.Random(seed)
+    subset = rng.sample(all_files, subset_size)
+
+    os.makedirs(output_dir)
+    for src in subset:
+        dst = os.path.join(output_dir, os.path.basename(src))
+        shutil.copy2(src, dst)
+
+    print(f"Created subset at {output_dir}.")
+    return output_dir
+
diff --git a/src/llama_recipes/utils/train_utils.py b/src/llama_recipes/utils/train_utils.py
index 307b618..879fed2 100644
--- a/src/llama_recipes/utils/train_utils.py
+++ b/src/llama_recipes/utils/train_utils.py
@@ -19,13 +19,53 @@ from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
 from tqdm import tqdm
 from transformers import LlamaTokenizer
 import json
-
+from mido import MidiFile
 
 from llama_recipes.model_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint, save_model_checkpoint_ddp, save_peft_checkpoint
 from llama_recipes.policies import fpSixteen,bfSixteen, get_llama_wrapper
 from llama_recipes.utils.memory_utils import MemoryTrace
 from accelerate.utils import is_xpu_available, is_ccl_available
 from llama_recipes.utils.flop_utils import FlopMeasure
+
+
+# Generation imports
+import sys
+import numpy as np
+import random
+import pickle
+import tempfile
+import subprocess
+from transformers import LlamaConfig
+from llama_recipes.datasets.music_tokenizer import MusicTokenizer
+# Import existing generation functionality
+recipes_path = os.path.join(os.path.dirname(__file__), '../../..', 'recipes/inference/custom_music_generation')
+if recipes_path not in sys.path:
+    sys.path.append(recipes_path)
+from generation import MusicLlama
+
+# Imports for evaluation functionality
+import shutil
+import importlib.util
+from pathlib import Path as PathLib
+
+# Import evaluation functions from piano_transformer metrics module
+try:
+    from llama_recipes.utils.metrics import create_subset, evaluate_mgeval_combined
+    print("Successfully imported evaluation functions from Moonbeam metrics")
+    
+except ImportError as e:
+    print(f"Error: {e}")
+    create_subset = None
+    evaluate_mgeval_combined = None
+
+def is_valid_midi(file_path):
+    try:
+        MidiFile(file_path)
+        return True
+    except Exception as e:
+        print(f"[INVALID MIDI] {file_path} - {e}")
+        return False
+
 def set_tokenizer_params(tokenizer: LlamaTokenizer):
     tokenizer.pad_token_id = 0
     tokenizer.padding_side = "left"
@@ -192,6 +232,7 @@ def train(model, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_sche
                                 'train/epoch': epoch + 1,
                                 'train/step': epoch * len(train_dataloader) + step,
                                 'train/loss': loss.detach().float(),
+                                'train/learning_rate': lr_scheduler.get_last_lr()[0],
                             })
 
                     pbar.set_description(f"Training Epoch: {epoch}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")
@@ -203,7 +244,7 @@ def train(model, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_sche
                     #TODO: More frequent evaluation; Remember to switch on model.train again
                     if step%train_config.validation_interval==0 and train_config.run_validation:
                         
-                        eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
+                        eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run, epoch=epoch, step=step, music_tokenizer=tokenizer)
                         if train_config.save_metrics:
                             val_step_loss.extend(temp_val_loss)
                             val_step_perplexity.extend(temp_step_perplexity)
@@ -250,11 +291,16 @@ def train(model, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_sche
                                         print("=====================================================")
                                 elif train_config.enable_ddp: 
                                     if not train_config.use_peft:
+                                        # Only save every 10 epochs in non-PEFT mode to save storage
+                                        if epoch % 10 == 0:
                                             save_model_checkpoint_ddp(
                                                 model, optimizer, rank, train_config, epoch=epoch, step=step
                                             )
                                             print(" Saving the DDP model checkpoints and optimizer using FULL_STATE_DICT")
                                             print("=====================================================")
+                                        else:
+                                            print(f" Skipping checkpoint save (non-PEFT mode, epoch {epoch}, next save at epoch {((epoch // 10) + 1) * 10})")
+                                            print("=====================================================")
                                     else:
                                         print("Warning! Model Checkpoints are not saved properly")
                                         print("=====================================================")
@@ -301,6 +347,17 @@ def train(model, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_sche
         # Update the learning rate as needed
         lr_scheduler.step()
 
+        # Log learning rate and epoch metrics to wandb
+        if wandb_run:
+            if not train_config.enable_fsdp or rank==0:
+                wandb_run.log({
+                    'epoch/learning_rate': lr_scheduler.get_last_lr()[0],
+                    'epoch/train_loss': train_epoch_loss,
+                    'epoch/train_perplexity': train_perplexity,
+                    'epoch/epoch_time': epoch_end_time,
+                    'epoch/epoch_number': epoch,
+                })
+
         if train_config.enable_fsdp or train_config.enable_ddp:
             if rank==0:
                 print(f"Epoch {epoch}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
@@ -311,6 +368,35 @@ def train(model, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_sche
         if train_config.save_metrics:
             save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
         
+        # Perform music generation after each epoch completes (safe from DataLoader conflicts)
+        # Check evaluation frequency - only run if it's the right epoch
+        should_evaluate = (epoch + 1) % train_config.evaluation_frequency_epochs == 0
+        if train_config.enable_generation and should_evaluate:
+            print(f"\n=== Music Generation for Epoch {epoch} (Frequency: every {train_config.evaluation_frequency_epochs} epochs) ===")
+            try:
+                generation_results = perform_music_generation(
+                    model, tokenizer, train_config, local_rank, epoch=epoch, step=None
+                )
+                print(f"Generated {len(generation_results)} music sequences for epoch {epoch}")
+                
+                # Evaluate generated music against training set
+                if train_config.enable_evaluation:
+                    print(f"\n=== Evaluation Against Train Set for Epoch {epoch} ===")
+                    try:
+                        eval_metrics = evaluate_against_train_set(
+                            generation_results, train_config, local_rank, epoch=epoch, step=None, wandb_run=wandb_run
+                        )
+                        print(f"Evaluation completed for epoch {epoch}")
+                    except Exception as e:
+                        print(f"Error during evaluation for epoch {epoch}: {e}")
+                        import traceback
+                        traceback.print_exc()
+                        
+            except Exception as e:
+                print(f"Error during music generation for epoch {epoch}: {e}")
+                import traceback
+                traceback.print_exc()
+
     avg_epoch_time = sum(epoch_times)/ len(epoch_times)
     avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
     avg_train_prep = sum(train_prep)/len(train_prep)
@@ -476,6 +562,7 @@ def train_overfit(model, batch, train_dataloader,eval_dataloader, tokenizer, opt
                                 'train/epoch': epoch + 1,
                                 'train/step': epoch * len(train_dataloader) + step,
                                 'train/loss': loss.detach().float(),
+                                'train/learning_rate': lr_scheduler.get_last_lr()[0],
                             })
 
                     pbar.set_description(f"Training Epoch: {epoch}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")
@@ -590,6 +677,17 @@ def train_overfit(model, batch, train_dataloader,eval_dataloader, tokenizer, opt
         # Update the learning rate as needed
         lr_scheduler.step()
 
+        # Log learning rate and epoch metrics to wandb
+        if wandb_run:
+            if not train_config.enable_fsdp or rank==0:
+                wandb_run.log({
+                    'epoch/learning_rate': lr_scheduler.get_last_lr()[0],
+                    'epoch/train_loss': train_epoch_loss,
+                    'epoch/train_perplexity': train_perplexity,
+                    'epoch/epoch_time': epoch_end_time,
+                    'epoch/epoch_number': epoch,
+                })
+
         if train_config.enable_fsdp or train_config.enable_ddp:
             if rank==0:
                 print(f"Epoch {epoch}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
@@ -600,6 +698,35 @@ def train_overfit(model, batch, train_dataloader,eval_dataloader, tokenizer, opt
         if train_config.save_metrics:
             save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
         
+        # Perform music generation after each epoch completes (safe from DataLoader conflicts)
+        # Check evaluation frequency - only run if it's the right epoch
+        should_evaluate = (epoch + 1) % train_config.evaluation_frequency_epochs == 0
+        if train_config.enable_generation and should_evaluate:
+            print(f"\n=== Music Generation for Epoch {epoch} (Frequency: every {train_config.evaluation_frequency_epochs} epochs) ===")
+            try:
+                generation_results = perform_music_generation(
+                    model, tokenizer, train_config, local_rank, epoch=epoch, step=None
+                )
+                print(f"Generated {len(generation_results)} music sequences for epoch {epoch}")
+                
+                # Evaluate generated music against training set
+                if train_config.enable_evaluation:
+                    print(f"\n=== Evaluation Against Train Set for Epoch {epoch} ===")
+                    try:
+                        eval_metrics = evaluate_against_train_set(
+                            generation_results, train_config, local_rank, epoch=epoch, step=None, wandb_run=wandb_run
+                        )
+                        print(f"Evaluation completed for epoch {epoch}")
+                    except Exception as e:
+                        print(f"Error during evaluation for epoch {epoch}: {e}")
+                        import traceback
+                        traceback.print_exc()
+                        
+            except Exception as e:
+                print(f"Error during music generation for epoch {epoch}: {e}")
+                import traceback
+                traceback.print_exc()
+
     avg_epoch_time = sum(epoch_times)/ len(epoch_times)
     avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
     avg_train_prep = sum(train_prep)/len(train_prep)
@@ -749,6 +876,7 @@ def train_con_gen(model, train_dataloader,eval_dataloader, tokenizer, optimizer,
                                 'train/epoch': epoch + 1,
                                 'train/step': epoch * len(train_dataloader) + step,
                                 'train/loss': loss.detach().float(),
+                                'train/learning_rate': lr_scheduler.get_last_lr()[0],
                             })
 
                     pbar.set_description(f"Training Epoch: {epoch}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")
@@ -759,7 +887,7 @@ def train_con_gen(model, train_dataloader,eval_dataloader, tokenizer, optimizer,
                 
                     #TODO: More frequent evaluation; Remember to switch on model.train again
                     if step%train_config.validation_interval==0:
-                        eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
+                        eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run, epoch=epoch, step=step, music_tokenizer=tokenizer)
                         if train_config.save_metrics:
                             val_step_loss.extend(temp_val_loss)
                             val_step_perplexity.extend(temp_step_perplexity)
@@ -807,11 +935,16 @@ def train_con_gen(model, train_dataloader,eval_dataloader, tokenizer, optimizer,
                                         print("=====================================================")
                                 elif train_config.enable_ddp: 
                                     if not train_config.use_peft:
+                                        # Only save every 10 epochs in non-PEFT mode to save storage
+                                        if epoch % 10 == 0:
                                             save_model_checkpoint_ddp(
                                                 model, optimizer, rank, train_config, epoch=epoch, step=step
                                             )
                                             print(" Saving the DDP model checkpoints and optimizer using FULL_STATE_DICT")
                                             print("=====================================================")
+                                        else:
+                                            print(f" Skipping checkpoint save (non-PEFT mode, epoch {epoch}, next save at epoch {((epoch // 10) + 1) * 10})")
+                                            print("=====================================================")
                                     else:
                                         print("Warning! Model Checkpoints are not saved properly")
                                         print("=====================================================")
@@ -858,6 +991,17 @@ def train_con_gen(model, train_dataloader,eval_dataloader, tokenizer, optimizer,
         # Update the learning rate as needed
         lr_scheduler.step()
 
+        # Log learning rate and epoch metrics to wandb
+        if wandb_run:
+            if not train_config.enable_fsdp or rank==0:
+                wandb_run.log({
+                    'epoch/learning_rate': lr_scheduler.get_last_lr()[0],
+                    'epoch/train_loss': train_epoch_loss,
+                    'epoch/train_perplexity': train_perplexity,
+                    'epoch/epoch_time': epoch_end_time,
+                    'epoch/epoch_number': epoch,
+                })
+
         if train_config.enable_fsdp or train_config.enable_ddp:
             if rank==0:
                 print(f"Epoch {epoch}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
@@ -868,6 +1012,35 @@ def train_con_gen(model, train_dataloader,eval_dataloader, tokenizer, optimizer,
         if train_config.save_metrics:
             save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
         
+        # Perform music generation after each epoch completes (safe from DataLoader conflicts)
+        # Check evaluation frequency - only run if it's the right epoch
+        should_evaluate = (epoch + 1) % train_config.evaluation_frequency_epochs == 0
+        if train_config.enable_generation and should_evaluate:
+            print(f"\n=== Music Generation for Epoch {epoch} (Frequency: every {train_config.evaluation_frequency_epochs} epochs) ===")
+            try:
+                generation_results = perform_music_generation(
+                    model, tokenizer, train_config, local_rank, epoch=epoch, step=None
+                )
+                print(f"Generated {len(generation_results)} music sequences for epoch {epoch}")
+                
+                # Evaluate generated music against training set
+                if train_config.enable_evaluation:
+                    print(f"\n=== Evaluation Against Train Set for Epoch {epoch} ===")
+                    try:
+                        eval_metrics = evaluate_against_train_set(
+                            generation_results, train_config, local_rank, epoch=epoch, step=None, wandb_run=wandb_run
+                        )
+                        print(f"Evaluation completed for epoch {epoch}")
+                    except Exception as e:
+                        print(f"Error during evaluation for epoch {epoch}: {e}")
+                        import traceback
+                        traceback.print_exc()
+                        
+            except Exception as e:
+                print(f"Error during music generation for epoch {epoch}: {e}")
+                import traceback
+                traceback.print_exc()
+
     avg_epoch_time = sum(epoch_times)/ len(epoch_times)
     avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
     avg_train_prep = sum(train_prep)/len(train_prep)
@@ -894,7 +1067,7 @@ def train_con_gen(model, train_dataloader,eval_dataloader, tokenizer, optimizer,
     return results
 
 
-def evaluation(model,train_config, eval_dataloader, local_rank, tokenizer, wandb_run):
+def evaluation(model,train_config, eval_dataloader, local_rank, tokenizer, wandb_run, epoch=None, step=None, music_tokenizer=None):
     """
     Evaluates the model on the given dataloader
 
@@ -903,8 +1076,11 @@ def evaluation(model,train_config, eval_dataloader, local_rank, tokenizer, wandb
         eval_dataloader: The dataloader containing the evaluation data
         local_rank: The rank of the current node in a distributed setting
         tokenizer: The tokenizer used to decode predictions
+        epoch: Current training epoch (for logging)
+        step: Current training step (for logging)
+        music_tokenizer: Optional pre-created MusicTokenizer (unused, kept for compatibility)
 
-    Returns: eval_ppl, eval_epoch_loss
+    Returns: eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity
     """
     if train_config.enable_fsdp:
         world_size = int(os.environ["WORLD_SIZE"])
@@ -923,7 +1099,11 @@ def evaluation(model,train_config, eval_dataloader, local_rank, tokenizer, wandb
                     print("max eval steps reached, stopping evaluation, total_eval_steps: ", total_eval_steps - 1)
                 break
             for key in batch.keys():
+            # Move to correct device
                 if train_config.enable_fsdp:
+                    if is_xpu_available():
+                        batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
+                    else:
                         batch[key] = batch[key].to(local_rank)
                 else:
                     if is_xpu_available():
@@ -960,11 +1141,18 @@ def evaluation(model,train_config, eval_dataloader, local_rank, tokenizer, wandb
     else:
         print(f" {eval_ppl=} {eval_epoch_loss=}")
 
+    # Music generation moved to after epoch completion to avoid DataLoader conflicts
+    generation_results = []
+
     if wandb_run:
-        wandb_run.log({
+        wandb_log_data = {
             'eval/perplexity': eval_ppl,
             'eval/loss': eval_epoch_loss,
-                    }, commit=False)
+        }
+        # Log generation statistics if available
+        # Generation results logging moved to epoch completion
+        
+        wandb_run.log(wandb_log_data, commit=False)
 
     return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity
 
@@ -1084,6 +1273,367 @@ def check_frozen_layers_peft_model(model):
                 print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")
 
 
+def get_model_config(model):
+    """
+    Safely get model config, handling wrapped models (DDP, FSDP)
+    """
+    if hasattr(model, 'module'):
+        # Model is wrapped (DDP, FSDP, etc.)
+        return getattr(model.module, 'config', None)
+    else:
+        # Model is not wrapped
+        return getattr(model, 'config', None)
+
+
+def evaluate_against_train_set(generation_results, train_config, local_rank, epoch=None, step=None, wandb_run=None):
+    """
+    Evaluate generated sequences against training set using mgeval metrics
+    Similar to EvalCallback from piano_transformer but without FMD
+    """
+    if not train_config.enable_evaluation:
+        return {}
+    
+    if create_subset is None or evaluate_mgeval_combined is None:
+        print("Evaluation functions not available, skipping evaluation")
+        return {}
+    
+    if not generation_results:
+        print("No generation results to evaluate")
+        return {}
+    
+    # Only run evaluation on main process
+    if train_config.enable_fsdp and local_rank != 0:
+        return {}
+    
+    try:
+        print(f"[Evaluation] Starting evaluation against train set with {len(generation_results)} generated samples...")
+        
+        # Create temporary directory for generated MIDI files
+        with tempfile.TemporaryDirectory() as temp_gen_dir:
+            temp_gen_path = PathLib(temp_gen_dir)
+            
+            # Convert generation results to MIDI files
+            midi_count = 0
+            for i, result in enumerate(generation_results):
+                try:
+                    if 'generation' in result and 'content' in result['generation']:
+                        # Save generated MIDI
+                        midi_path = temp_gen_path / f'generated_{i}.mid'
+                        result['generation']['content'].save(str(midi_path))
+                        midi_count += 1
+                    else:
+                        print(f"Warning: No MIDI content found in generation result {i}")
+                except Exception as e:
+                    print(f"Error saving MIDI file {i} for evaluation: {e}")
+            
+            if midi_count == 0:
+                print("No valid MIDI files generated for evaluation")
+                return {}
+            
+            print(f"[Evaluation] Saved {midi_count} MIDI files for evaluation")
+            
+            # Create subset of training data
+            ref_dir = PathLib(train_config.evaluation_ref_dir)
+            if not ref_dir.exists():
+                print(f"Warning: Reference directory {ref_dir} does not exist")
+                return {}
+            
+            train_dir_subset = create_subset(ref_dir, train_config.generation_num_samples)
+            print(f"[Evaluation] Created training subset with {train_config.generation_num_samples} files from {ref_dir}")
+            
+            # Compute evaluation metrics
+            metrics = {}
+            
+            # Compute mgeval metrics
+            try:
+                print(f"[Evaluation] Starting mgeval evaluation between {train_dir_subset} and {temp_gen_path}")
+                relative_summary = evaluate_mgeval_combined(
+                    dataset1_path=train_dir_subset,
+                    dataset2_path=temp_gen_path,
+                )
+                
+                print(f"[Evaluation] Got results: {len(relative_summary)} relative items")
+                
+                # Process only relative summary for averages
+                kld_sum = 0
+                oa_sum = 0
+                for item in relative_summary:
+                    kld_sum += item["KLD"]
+                    oa_sum += item["OA"]
+                
+                if len(relative_summary) > 0:
+                    metrics["KLD_average"] = kld_sum / len(relative_summary)
+                    metrics["OA_average"] = oa_sum / len(relative_summary)
+                    print(f"[Evaluation] Computed averages - KLD: {metrics['KLD_average']:.4f}, OA: {metrics['OA_average']:.4f}")
+                else:
+                    metrics["KLD_average"] = 0.0
+                    metrics["OA_average"] = 0.0
+                    print(f"[Evaluation] No relative summary data, setting averages to 0.0")
+                
+                print(f"[Evaluation] Total metrics computed: {len(metrics)}")
+                
+            except Exception as e:
+                print(f"Error computing mgeval metrics: {e}")
+                import traceback
+                traceback.print_exc()
+            
+            # Log metrics to wandb
+            if wandb_run and "KLD_average" in metrics and "OA_average" in metrics:
+                try:
+                    # Log metrics if available
+                    wandb_log_data = {
+                        "custom_eval/KLD_average": float(metrics["KLD_average"]),
+                        "custom_eval/OA_average": float(metrics["OA_average"])
+                    }
+                    
+                    # Create success message with available metrics
+                    log_message = f"[Evaluation] Successfully logged KLD_average={metrics['KLD_average']:.4f}, OA_average={metrics['OA_average']:.4f}"
+                    
+                except Exception as e:
+                    print(f"Error logging to wandb: {e}")
+                    import traceback
+                    traceback.print_exc()
+            elif wandb_run is None:
+                print("[Evaluation] wandb_run is None - wandb logging disabled")
+            elif not ("KLD_average" in metrics and "OA_average" in metrics):
+                print(f"[Evaluation] Missing required metrics - KLD_average: {'KLD_average' in metrics}, OA_average: {'OA_average' in metrics}")
+            else:
+                print("[Evaluation] Unknown condition preventing wandb logging")
+            
+            print(f"[Evaluation] Evaluation metrics: {metrics}")
+            
+            # Print the averages
+            if "KLD_average" in metrics and "OA_average" in metrics:
+                key_metrics_msg = f"[Evaluation] Key metrics - KLD_average: {metrics['KLD_average']:.4f}, OA_average: {metrics['OA_average']:.4f}"
+                print(key_metrics_msg)
+            return metrics
+            
+    except Exception as e:
+        print(f"Error during evaluation: {e}")
+        import traceback
+        traceback.print_exc()
+        return {}
+
+
+def perform_music_generation(model, tokenizer, train_config, local_rank, epoch=None, step=None):
+    """
+    Perform music generation during evaluation using MusicLlama library
+    """
+    if not train_config.enable_generation:
+        return []
+    
+    print(f"Starting music generation with {train_config.generation_num_samples} samples...")
+    
+    try:
+        # Handle DDP wrapping
+        if hasattr(model, 'module'):
+            unwrapped_model = model.module
+            model_config = unwrapped_model.config
+        else:
+            # Model is not wrapped
+            unwrapped_model = model
+            model_config = model.config
+        
+        unwrapped_model.eval()
+        
+        # Get model device and dtype
+        model_device = next(unwrapped_model.parameters()).device
+        model_dtype = next(unwrapped_model.parameters()).dtype
+        print(f"Model device: {model_device}, dtype: {model_dtype}")
+        
+        
+        def run_generation_safe():
+            """Run generation after DataLoader is finished"""
+            
+            # Save current default tensor type and device
+            original_default_dtype = torch.get_default_dtype()
+            original_cuda_device = torch.cuda.current_device() if model_device.type == 'cuda' else None
+            
+            # Save the device or dtype
+            try:
+                original_tensor_type = torch.tensor([]).type()  # Get default type
+                print(f"DEBUG: Original tensor type before generation: {original_tensor_type}")
+            except:
+                original_tensor_type = "torch.FloatTensor"  # Fallback to CPU float tensor
+                print(f"DEBUG: Failed to get original tensor type, using fallback: {original_tensor_type}")
+            
+            try:
+                print(f"Running music generation with {train_config.generation_num_samples} samples...")
+                
+                # Set CUDA context for generation
+                if model_device.type == 'cuda':
+                    torch.cuda.set_device(model_device)
+                    
+                    # Set default tensor type
+                    if model_dtype == torch.bfloat16:
+                        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
+                    elif model_dtype == torch.float16:
+                        torch.set_default_tensor_type(torch.cuda.HalfTensor)
+                    else:
+                        torch.set_default_tensor_type(torch.cuda.FloatTensor)
+                
+                # Create MusicLlama wrapper
+                print(f"DEBUG: Using fine-tuned model with {sum(p.numel() for p in unwrapped_model.parameters())} parameters")
+                print(f"DEBUG: Model is in {'training' if unwrapped_model.training else 'eval'} mode")
+                print(f"DEBUG: Model memory address: {id(unwrapped_model)} (same as training model)")
+                music_llama = MusicLlama(unwrapped_model, tokenizer, model_config)
+                
+                # Create generation prompts based on generation mode, still just dummy for random_files and all_test_files
+                prompts = []
+                if train_config.generation_mode == "from_scratch":
+                    prompts = [[tokenizer.sos_token_compound] for _ in range(train_config.generation_num_samples)]
+                elif train_config.generation_mode == "random_files":
+                    prompts = [[tokenizer.sos_token_compound] for _ in range(train_config.generation_num_samples)]
+                elif train_config.generation_mode == "all_test_files":
+                    prompts = [[tokenizer.sos_token_compound] for _ in range(train_config.generation_num_samples)]
+                else:
+                    raise ValueError(f"Invalid generation_mode: {train_config.generation_mode}. Must be one of: 'from_scratch', 'random_files', 'all_test_files'")
+                
+                # Use existing music_completion
+                results = music_llama.music_completion(
+                    prompts,
+                    max_gen_len=train_config.generation_max_gen_len,
+                    temperature=train_config.generation_temperature,
+                    top_p=train_config.generation_top_p,
+                )
+                
+                print(f"Generated {len(results)} music sequences successfully!")
+                return results
+                
+            finally:
+                # Always restore original state
+                if model_device.type == 'cuda' and original_cuda_device is not None:
+                    torch.cuda.set_device(original_cuda_device)
+                
+                # Restore the original default tensor type (device + dtype)
+                try:
+                    if original_tensor_type == "torch.FloatTensor":
+                        torch.set_default_tensor_type(torch.FloatTensor)  # CPU Float
+                    elif original_tensor_type == "torch.cuda.FloatTensor":
+                        torch.set_default_tensor_type(torch.cuda.FloatTensor)  # CUDA Float
+                    elif original_tensor_type == "torch.cuda.HalfTensor":
+                        torch.set_default_tensor_type(torch.cuda.HalfTensor)  # CUDA Half
+                    elif original_tensor_type == "torch.cuda.BFloat16Tensor":
+                        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)  # CUDA BFloat16
+                    else:
+                        # Fallback to CPU float tensor for unknown types
+                        torch.set_default_tensor_type(torch.FloatTensor)
+                        print(f"Warning: Unknown original tensor type {original_tensor_type}, falling back to CPU FloatTensor")
+                    
+                    # Debug: Verify restoration worked
+                    restored_type = torch.tensor([]).type()
+                    print(f"DEBUG: Restored tensor type after generation: {restored_type}")
+                    
+                except Exception as e:
+                    # Fallback to just setting dtype if tensor type fails
+                    torch.set_default_dtype(original_default_dtype)
+                    print(f"Warning: Failed to restore tensor type, restored dtype only: {e}")
+        
+        # Run generation safely after DataLoader is finished  
+        results = run_generation_safe()
+        
+        # Put model back in train mode
+        unwrapped_model.train()
+        
+        # Save generation results
+        if results and (not train_config.enable_fsdp or local_rank == 0):
+            save_generation_results(results, train_config, epoch, step)
+        
+        return results
+        
+    except Exception as e:
+        # Ensure model is back in train mode even on error
+        try:
+            if 'unwrapped_model' in locals():
+                unwrapped_model.train()
+        except:
+            pass  # Ignore errors when setting train mode
+            
+        print(f"Error during music generation: {e}")
+        import traceback
+        traceback.print_exc()
+        return []
+
+
+def save_generation_results(generated_sequences, train_config, epoch=None, step=None):
+    """
+    Save generated sequences to MIDI files
+    """
+    try:
+        if not generated_sequences:
+            print("No generation results to save (generation was skipped or failed)")
+            return
+        # Create save directory
+        save_dir = train_config.generation_save_dir
+        if epoch is not None and step is not None:
+            save_dir = os.path.join(save_dir, f"epoch_{epoch}_step_{step}")
+        elif epoch is not None:
+            save_dir = os.path.join(save_dir, f"epoch_{epoch}")
+        
+        os.makedirs(save_dir, exist_ok=True)
+
+    
+        # Save individual sequences as MIDI files
+        saved_count = 0
+        for i, result in enumerate(generated_sequences):
+            try:
+                if 'generation' in result and 'content' in result['generation']:
+                    # Save generated MIDI
+                    midi_path = os.path.join(save_dir, f'generated_{i}.mid')
+                    result['generation']['content'].save(midi_path)
+                    
+                    # Check if the saved MIDI file is valid
+                    if is_valid_midi(midi_path):
+                        saved_count += 1
+                        print(f"Valid MIDI saved to {midi_path}")
+                    else:
+                        # Delete invalid MIDI file
+                        try:
+                            os.remove(midi_path)
+                            print(f"Invalid MIDI deleted: {midi_path}")
+                        except:
+                            print(f"Invalid MIDI (couldn't delete): {midi_path}")
+                        continue
+                    
+                    # Save prompt MIDI only for prompts longer than 1 token
+                    print(f"DEBUG: Generation mode = '{train_config.generation_mode}'")
+                    
+                    if (train_config.generation_mode != "from_scratch" and 
+                        'prompt' in result['generation'] and 
+                        'prompt_tokens' in result['generation']):
+                        
+                        prompt_tokens = result['generation']['prompt_tokens']
+                        print(f"DEBUG: Prompt has {len(prompt_tokens)} tokens for generated_{i}")
+                        
+                        if len(prompt_tokens) > 0:
+                            prompt_path = os.path.join(save_dir, f'generated_{i}_prompt.mid')
+                            result['generation']['prompt'].save(prompt_path)
+                            
+                            # Validate MIDI prompt
+                            if is_valid_midi(prompt_path):
+                                print(f"Valid prompt MIDI saved to {prompt_path}")
+                            else:
+                                try:
+                                    os.remove(prompt_path)
+                                    print(f"Invalid prompt MIDI deleted: {prompt_path}")
+                                except:
+                                    print(f"Invalid prompt MIDI (couldn't delete): {prompt_path}")
+                        else:
+                            print(f"Skipping prompt save for generated_{i} (empty prompt after SOS removal)")
+                    else:
+                        print(f"Skipping prompt save for generated_{i} (from_scratch mode or no prompt data)")
+                        
+            except Exception as e:
+                print(f"Error saving MIDI file {i}: {e} - Skipping this generation")
+                # Skip failed generations
+                continue
+        
+        print(f"Saved {saved_count}/{len(generated_sequences)} valid MIDI files to {save_dir}")
+        
+    except Exception as e:
+        print(f"Error saving generation results: {e}")
+
+
 def setup():
     """Initialize the process group for distributed training"""
     if is_ccl_available():
```