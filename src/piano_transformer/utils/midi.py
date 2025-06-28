import copy
import subprocess

import pandas as pd
from pathlib import Path

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

    
def get_midi_file_lists_by_random(midi_dir, pattern, seed):
    all_files = sorted(midi_dir.rglob(pattern))
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


def midi2wav(input_path, output_path, soundfont, quiet=True):
    soundfont_path = Path("assets") / soundfont
    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    def convert_file(midi_file):
        wav_filename = midi_file.stem + ".wav"
        wav_path = output_path / wav_filename

        command = [
            "fluidsynth",
            "-ni",
            *(["-q"] if quiet else []),
            str(soundfont_path),
            str(midi_file),
            "-F", str(wav_path),
            "-r", "44100"
        ]

        subprocess.run(command, check=True)

    if input_path.is_dir():
        for midi_file in input_path.iterdir():
            if midi_file.is_file() and midi_file.suffix.lower() == ".midi":
                convert_file(midi_file)
    elif input_path.is_file() and input_path.suffix.lower() == ".midi":
        convert_file(input_path)
    else:
        raise ValueError("Input path must be a .midi file or a directory containing .midi files")
