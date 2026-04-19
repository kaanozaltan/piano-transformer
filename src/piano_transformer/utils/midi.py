import copy
import json

import pandas as pd
import numpy as np


def get_midi_file_lists_by_csv(csv_path, midi_dir):
    df = pd.read_csv(csv_path)
    return {
        split: [
            (midi_dir / f).resolve()
            for f in df[df["split"] == split]["midi_filename"].tolist()
        ]
        for split in ("train", "validation", "test")
    }


def get_midi_file_lists_by_random(midi_dir, pattern, seed, genre_filter=None):
    metadata_path = midi_dir / "metadata.json"
    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    all_files = sorted(midi_dir.rglob(pattern))

    if genre_filter:
        filtered_files = []
        for file in all_files:
            # Extract ID from filename (assuming "123.mid" → "123")
            midi_id = file.stem.split("_")[0].lstrip("0")
            if midi_id in metadata:
                file_genre = metadata[midi_id]["metadata"].get("genre", None)
                if file_genre == genre_filter:
                    filtered_files.append(file)
        all_files = filtered_files

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(all_files)

    n = len(shuffled)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_files = shuffled[:n_train]
    val_files = shuffled[n_train : n_train + n_val]
    test_files = shuffled[n_train + n_val :]

    return {
        "train": [f.resolve() for f in train_files],
        "validation": [f.resolve() for f in val_files],
        "test": [f.resolve() for f in test_files],
    }


def scale_tempo(score, tempo_factor):
    score_copy = copy.deepcopy(score)
    for track in score_copy.tracks:
        for note in track.notes:
            note.time = int(note.time * tempo_factor)
            note.duration = int(note.duration * tempo_factor)
        for control in track.controls:
            control.time = int(control.time * tempo_factor)
    return score_copy
