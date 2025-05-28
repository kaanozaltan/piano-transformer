import copy
from pathlib import Path

import numpy as np
import pandas as pd
from symusic import Score


def get_midi_file_lists(csv_path: Path, midi_dir: Path):
    df = pd.read_csv(csv_path)
    return {
        split: [
            (midi_dir / f).resolve()
            for f in df[df["split"] == split]["midi_filename"].tolist()
        ]
        for split in ("train", "validation", "test")
    }


def scale_tempo(score: Score, tempo_factor: float) -> Score:
    score_copy = copy.deepcopy(score)
    for track in score_copy.tracks:
        for note in track.notes:
            note.time = int(note.time * tempo_factor)
            note.duration = int(note.duration * tempo_factor)
        for control in track.controls:
            control.time = int(control.time * tempo_factor)
    return score_copy
