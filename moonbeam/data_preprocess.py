import os
import matplotlib.pyplot as plt
from llama_recipes.datasets.music_tokenizer import MusicTokenizer
import traceback
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from collections import Counter
import numpy as np
from collections import defaultdict
import argparse
import csv
from sklearn.model_selection import train_test_split
import pandas as pd
num_cores = multiprocessing.cpu_count()

def chunk_compounds(compounds, threshold=1024):
    """chunk the compounds such that long silences in between are not treated as long timeshifts"""

    onsets = [c[0] for c in compounds]
    onsets_padded = [0] + onsets
    timeshifts = [onsets_padded[i+1] - onsets_padded[i] for i in range(len(onsets_padded) - 1)]
    cur_pos = 0
    out = []
    for pointer in range(len(onsets)):
        if timeshifts[pointer] > threshold:
            out.append(compounds[cur_pos:pointer])
            cur_pos = pointer
    out.append(compounds[cur_pos:])

    for i, chunk in enumerate(out): #shift the onsets to 0
        first_onset = chunk[0][0]
        out[i] = [[comps[0]-first_onset] + comps[1:]  for comps in chunk]
    assert sum([len(chunk) for chunk in out]) == len(compounds)
    return out

def find_midi_files(folder):
    midi_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.midi') or file.endswith('.mid') or file.endswith('.MID'):
                midi_files.append(os.path.join(root, file))
    return midi_files

def find_midi_files_from_file(dataset_name, split_file, dataset_folder):
    if dataset_name =="GAPS":
        df = pd.read_csv(split_file)
        train_files = df[df['split'] == 'train_annotation']['filename'].tolist()
        test_files = df[df['split'] == 'test_annotation']['filename'].tolist()
        midi_files = train_files + test_files
        midi_files = [os.path.join(dataset_folder, f) for f in midi_files]
        splits = ['train']*len(train_files) + ['test']*len(test_files)

    elif dataset_name == "GuitarSet":
        import json
        with open(split_file, 'r') as file:
            splits = json.load(file)
        train_files = [os.path.join(dataset_folder, os.path.basename(f)) for f in splits['train_annotation']]
        test_files = [os.path.join(dataset_folder, os.path.basename(f)) for f in splits['test_annotation']]
        midi_files = train_files + test_files
        splits = ['train']*len(train_files) + ['test']*len(test_files)

    elif dataset_name == "maestro":
        df = pd.read_csv(split_file)
        train_files = df[df['split'] == 'train']['midi_filename'].tolist()
        val_files = df[df['split'] == 'validation']['midi_filename'].tolist()
        test_files = df[df['split'] == 'test']['midi_filename'].tolist()

        midi_files = train_files + val_files + test_files
        midi_files = [os.path.join(dataset_folder, f) for f in midi_files]

        splits = (
            ['train'] * len(train_files) +
            ['validation'] * len(val_files) +
            ['test'] * len(test_files)
        )

    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")


    assert len(midi_files) == len(splits)
    return midi_files, splits

def process_midi_file_safe_v2(midi_file, split, onset_vocab_size, dur_vocab_size, output_folder, log_file, silence_threshold = None):
    """
    input: midi_file, split, onset_vocab_size, dur_vocab_size, output_folder, log_file, silence_threshold

    output: a list of processed chunked dictionaries [{file: file_path, split: split, onsets: onset_counter, durations: duration_counter, length: length}, ...] or None
    """
    try:
        out = process_midi_file_v2(midi_file, split, onset_vocab_size, dur_vocab_size, output_folder, log_file, silence_threshold)
        if out is not None:
            for out_chunk in out:
                if out_chunk is not None:
                    np.save(out_chunk['file'], out_chunk['compounds'])
        return out
    except Exception as e:
        # Log failure with full traceback
        with open(log_file, 'a') as log:
            log.write(f'Failed to process {midi_file}:\n')
            log.write(traceback.format_exc())
            log.write('\n')
        return [None]

def detect_large_timeshifts_and_durations(compounds, onset_vocab_size, dur_vocab_size):
    #check if file contains large timeshits and durations
    onsets = [c[0] for c in compounds]
    onsets_padded = [0] + onsets
    timeshift_counter = Counter([onsets_padded[i+1] - onsets_padded[i] for i in range(len(onsets_padded) - 1)]) 
    duration_counter = Counter([c[1] for c in compounds])
    onsets_exceed_vocab_size = any(key > onset_vocab_size-3 for key in timeshift_counter.keys()) #why -3? onset_vocab_size -1, onset_vocab_size -2 are assigned to sos and eos, max value is onset_vocab_size - 3
    duration_exceed_vocab_size = any(key > dur_vocab_size-3 for key in duration_counter.keys()) #if dur_vocab_size=1026, 1024 and 1025 are reserved for sos and eos, the largest value becomes 1023
    return onsets_exceed_vocab_size, duration_exceed_vocab_size, timeshift_counter, duration_counter

def filter_large_ts_dur(compounds, output_file_path, split, onset_vocab_size, dur_vocab_size, log_file):
    """detect large timeshifts and durations, return None if any of them exceed the vocab size else return the processed dictionary"""

    onsets_exceed_vocab_size, duration_exceed_vocab_size, timeshift_counter, duration_counter = detect_large_timeshifts_and_durations(compounds, onset_vocab_size, dur_vocab_size)

    if onsets_exceed_vocab_size or duration_exceed_vocab_size:
        with open(log_file, 'a') as log:
            log.write(f'Failed to process {output_file_path}:\n')
            log.write(f'{output_file_path} contains large onsets: {onsets_exceed_vocab_size}, large durations: {duration_exceed_vocab_size}\n')
            # Optionally, print the largest onset and duration exceeding the vocab size
            if onsets_exceed_vocab_size:
                largest_onset = max(key for key in timeshift_counter.keys() if key > onset_vocab_size - 3)
                log.write(f'Largest onset: {largest_onset}\n')
            if duration_exceed_vocab_size:
                largest_duration = max(key for key in duration_counter.keys() if key > dur_vocab_size - 3)
                log.write(f'Largest duration: {largest_duration}\n')
        return None
    else:

        return {
            'file': output_file_path,
            'compounds': compounds,
            'split': split,
            'timeshifts':dict(timeshift_counter),
            'durations':dict(duration_counter),           
            'length_token': len(compounds),
            'length_duration': compounds[-1][0]+compounds[-1][1],
        }

def process_midi_file_v2(midi_file, split, onset_vocab_size, dur_vocab_size, output_folder, log_file, silence_threshold = None):
    #convert midi to compounds
    compounds = tokenizer.midi_to_compound(midi_file)
    output_file_name = midi_file.replace("/", "_").replace(".midi", ".npy").replace(".mid", ".npy") #TODO: this will avoid duplicates
    output_file_path = os.path.join(output_folder, output_file_name)
    if silence_threshold: #split compounds if timeshift exceeds threshold
        list_of_compounds = chunk_compounds(compounds, threshold=silence_threshold)
        if len(list_of_compounds)==1:
            return [filter_large_ts_dur(compounds, output_file_path, split, onset_vocab_size, dur_vocab_size, log_file)]
        else:
            list_of_output_file_path = [os.path.join(output_folder, 
                                                     midi_file.split("/")[-1].replace('.midi', f'_{i}.npy').replace('.mid', f'_{i}.npy')) 
                                                     for i in range(len(list_of_compounds))]
            return [filter_large_ts_dur(compounds, output_file_path, split, onset_vocab_size, dur_vocab_size, log_file) for (compounds, output_file_path) in zip(list_of_compounds, list_of_output_file_path)]
    else: 
        return [filter_large_ts_dur(compounds, output_file_path, split, onset_vocab_size, dur_vocab_size, log_file)]

def analyze(processed_midis):
    timeshift_counts_list, duration_counts_list, file_length_counts_list = zip(*[[midi['timeshifts'], midi['durations'], {midi['length_token']:1}] for midi in processed_midis])
    total_duration = sum([midi['length_duration'] for midi in processed_midis])
    total_length = sum([midi['length_token'] for midi in processed_midis])
    timeshift_counts = merge_dictionaries_parallel(timeshift_counts_list) #TODO: check correctness
    duration_counts = merge_dictionaries_parallel(duration_counts_list)
    file_length_counts = merge_dictionaries_parallel(file_length_counts_list)
    return timeshift_counts, duration_counts, file_length_counts, total_duration, total_length

def merge_dicts_chunk(chunk):
    result = {}
    for d in chunk:
        for key, value in d.items():
            result[key] = result.get(key, 0) + value
    return result

def merge_dictionaries_parallel(dicts):
    # Split dicts into chunks for parallel processing
    num_chunks = len(dicts)
    chunk_size = max(num_chunks // num_cores, 1)
    # chunk_size = 300
    chunks = [dicts[i:i + chunk_size] for i in range(0, num_chunks, chunk_size)]

    # Merge chunks in parallel
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        merged_dicts = list(tqdm(executor.map(merge_dicts_chunk, chunks), total=len(chunks), desc='Merging Dict'))
    # Combine results from parallel execution
    result = {}
    for merged_dict in merged_dicts:
        for key, value in merged_dict.items():
            result[key] = result.get(key, 0) + value
    result_sorted = {k: result[k] for k in sorted(result)}
    return result_sorted

def plot_histogram(input_dict, x_label, y_label, title, xscale='log', save_path = 'path/to/save'):
    plt.figure(figsize=(12, 6))
    keys = list(input_dict.keys())  # Convert keys to list
    values = list(input_dict.values())  # Convert values to list
    # Define the number of bins
    num_bins = 100  # adjust based on your data
    plt.hist(keys, bins=num_bins, weights=values, edgecolor='black')
    plt.xscale(xscale)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.savefig(save_path, dpi=200)
    plt.close()

# Main script execution
if __name__ == '__main__':
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Process MIDI files for the MIDI dataset.')

    # Define and Parse the arguments
    parser.add_argument('--dataset_name', type=str, help='Dataset Name')
    parser.add_argument('--dataset_folder', type=str, help='Path to the dataset folder containing MIDI files.')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where processed files will be saved.')
    parser.add_argument('--model_config', type=str, help='Model configuration file that decides the vocab size')
    parser.add_argument('--train_test_split_file', type=lambda x: None if x == "None" else str(x), help='Path to the split file.')
    parser.add_argument('--train_ratio', type=float, help='Training/Total')
    parser.add_argument('--ts_threshold', type=lambda x: None if x == "None" else int(x), help='If Timeshift exceeds this value, chunk the file')

    args = parser.parse_args()

    #get file paths
    midi_output_folder = args.output_folder+"/processed"
    log_file = args.output_folder + "/failed_midi_files.log"
    csv_file_path = args.output_folder+"/train_test_split.csv"
    train_stats_file = args.output_folder+"/train_tokens_stats.json"
    test_stats_file = args.output_folder+"/test_tokens_stats.json"
    os.makedirs(midi_output_folder, exist_ok=True)
    
    if args.train_ratio==1:
        midi_files = find_midi_files(args.dataset_folder)
        splits = ['train']*len(midi_files)
        print(f"{len(midi_files)} midi files found! all assigned to the train split")
    elif args.train_ratio==0:
        midi_files = find_midi_files(args.dataset_folder)
        splits = ['test']*len(midi_files)
        print(f"{len(midi_files)} midi files found! all assigned to the test split")
    else: #assert error 
        if args.train_test_split_file:
            midi_files, splits = find_midi_files_from_file(args.dataset_name, args.train_test_split_file, args.dataset_folder)
        else:
            # Find all MIDI files in the folder
            midi_files = find_midi_files(args.dataset_folder)

            # Split into train and test
            train_files, test_files = train_test_split(midi_files, train_size=args.train_ratio, random_state=42)

            # Assign labels
            splits = ['train'] * len(train_files) + ['test'] * len(test_files)
            midi_files = train_files + test_files

    #determine vocab size
    with open(args.model_config, 'r') as file:
        data = json.load(file)
        onset_vocab_size = data.get("onset_vocab_size", None) #value - 2 = max value of timeshift
        dur_vocab_size = data.get("dur_vocab_size", None) #value - 2 = max value of duration
        octave_vocab_size = data.get("octave_vocab_size", None)
        pitch_class_vocab_size = data.get("pitch_class_vocab_size", None)
        instrument_vocab_size = data.get("instrument_vocab_size", None)
        velocity_vocab_size = data.get("velocity_vocab_size", None)
        assert onset_vocab_size and dur_vocab_size
    print(f"processing using {num_cores} cpus. tokenizer config: max timeshift allowed: {onset_vocab_size-3}, max duration allowed: {dur_vocab_size-3}")
    tokenizer = MusicTokenizer(timeshift_vocab_size = onset_vocab_size, dur_vocab_size = dur_vocab_size, octave_vocab_size = octave_vocab_size, pitch_class_vocab_size = pitch_class_vocab_size, instrument_vocab_size = instrument_vocab_size, velocity_vocab_size = velocity_vocab_size)  

    # Open the CSV file for writing directly
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['file_base_name', 'split', 'length', 'duration'])  # Write the header

        # Process all MIDI files
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            for result in tqdm(
                executor.map(process_midi_file_safe_v2, midi_files, splits, [onset_vocab_size] * len(midi_files), [dur_vocab_size] * len(midi_files), [midi_output_folder] * len(midi_files), [log_file] * len(midi_files), [args.ts_threshold] * len(midi_files)),
                total=len(midi_files),
                desc="Processing MIDI files"
            ):
                if result is not None:  # Only process successful results
                    for sublist in result:  # Handle nested results
                        if sublist is not None:
                            # Write each result directly to the CSV
                            csv_writer.writerow([os.path.basename(sublist['file']), sublist['split'], sublist['length_token'], sublist['length_duration']])

    print(f'Processed {len(midi_files)} files. Results saved to {csv_file_path}, with {pd.read_csv(csv_file_path).shape[0]} successes. Success ratio: {pd.read_csv(csv_file_path).shape[0]/len(midi_files)* 100:.2f}%')