import os
import re
import json
import numpy as np
from collections import defaultdict

# === PART 1: Count lengths of npy files in 309M and 839M ===
def collect_npy_lengths(base_dir):
    lengths_by_year = defaultdict(dict)
    for file in os.listdir(base_dir):
        if file.endswith(".npy"):
            year_match = re.search(r"maestro_(\d{4})", file)
            if not year_match:
                continue
            year = year_match.group(1)
            # Remove "data_maestro_" and the year + underscore from filename
            name = file.replace("data_maestro_", "")
            # Remove year prefix like "2004_" or "2006_" from the name if present
            name = re.sub(r"^\d{4}_", "", name)
            name = name.replace(".npy", "")
            file_path = os.path.join(base_dir, file)
            try:
                data = np.load(file_path, allow_pickle=True)
                lengths_by_year[year][name] = len(data)
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
    return lengths_by_year

lengths_309M = collect_npy_lengths("preprocessed/309M/processed")
lengths_839M = collect_npy_lengths("preprocessed/839M/processed")

# === PART 2: Count MIDI chunks per base file across train/val/test ===
midi_dirs = [
    "/hpcwork/lect0148/experiments/mistral-155M_remi_maestro_v8/data_processed/maestro_train",
    "/hpcwork/lect0148/experiments/mistral-155M_remi_maestro_v8/data_processed/maestro_validation",
    "/hpcwork/lect0148/experiments/mistral-155M_remi_maestro_v8/data_processed/maestro_test"
]

midi_chunks = defaultdict(lambda: defaultdict(int))

for base_dir in midi_dirs:
    for year in os.listdir(base_dir):
        year_path = os.path.join(base_dir, year)
        if not os.path.isdir(year_path):
            continue
        for file in os.listdir(year_path):
            if file.endswith(".midi"):
                base_name = re.sub(r"_\d+\.midi$", "", file)
                midi_chunks[year][base_name] += 1

# Multiply chunk counts by 1024
for year in midi_chunks:
    for base_name in midi_chunks[year]:
        midi_chunks[year][base_name] *= 1024


# === PART 3: Compute averages per year ===
def average_dicts(dict_of_dicts):
    averages = {}
    for year, files in dict_of_dicts.items():
        if len(files) > 0:
            averages[year] = sum(files.values()) / len(files)
    return averages

avg_309M = average_dicts(lengths_309M)
avg_839M = average_dicts(lengths_839M)
avg_midi_chunks = average_dicts(midi_chunks)

# === PART 4: Calculate scaling factors ===
scaling_factors_309M = {}
scaling_factors_839M = {}

for year in avg_midi_chunks:
    if year in avg_309M:
        scaling_factors_309M[year] = avg_midi_chunks[year] / avg_309M[year]
    if year in avg_839M:
        scaling_factors_839M[year] = avg_midi_chunks[year] / avg_839M[year]

# === PART 5: Save to JSON ===
output = {
    "npy_lengths": {
        "309M": lengths_309M,
        "839M": lengths_839M
    },
    "midi_chunks": midi_chunks,
    "averages": {
        "309M": avg_309M,
        "839M": avg_839M,
        "midi_chunks": avg_midi_chunks
    },
    "scaling_factors": {
        "309M": scaling_factors_309M,
        "839M": scaling_factors_839M
    }
}

with open("analysis_results.json", "w") as f:
    json.dump(output, f, indent=2)

