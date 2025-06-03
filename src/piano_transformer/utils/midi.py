import copy
import subprocess
from importlib import resources
from pathlib import Path

import pandas as pd


def get_midi_file_lists(csv_path, midi_dir):
    df = pd.read_csv(csv_path)
    return {
        split: [
            (midi_dir / f).resolve()
            for f in df[df["split"] == split]["midi_filename"].tolist()
        ]
        for split in ("train", "validation", "test")
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


def convert_midi_to_wav(input_path, output_path, soundfont, quiet=True):
    with resources.path("piano_transformer.resources.soundfonts", soundfont) as soundfont_path:
        input_path = Path(input_path)
        output_path = Path(output_path)

        output_path.mkdir(parents=True, exist_ok=True)

        def convert_file(midi_file):
            wav_filename = midi_file.stem + ".wav"
            wav_path = output_path / wav_filename

            command = [
                "fluidsynth",
                "-ni",
                "-q" if quiet else "",
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
