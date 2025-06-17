from pathlib import Path

from miditok import REMI, TokenizerConfig, Event
from miditok.attribute_controls import AttributeControl
from symusic.core import TrackTick
from typing import Sequence


GENRE_TOKENS = [
    "Ambient",
    "Blues", 
    "Children",
    "Classical",
    "Country",
    "Electronic",
    "Folk",
    "Jazz",
    "Latin",
    "Pop",
    "Rap",
    "Reggae", 
    "Religious",
    "Rock",
    "Soul",
    "Soundtracks",
    "Unknown",
    "World"
]

class AttributeControlGenre(AttributeControl):

    def __init__(self) -> None:
        # Create tokens with "GENRE" as type and each genre as value
        genre_tokens = [f"GENRE_{genre.upper()}" for genre in GENRE_TOKENS]
        super().__init__(tokens=genre_tokens)

    def compute(
            self,
            track: TrackTick,
            time_division: int,
            ticks_bars: Sequence[int],
            ticks_beats: Sequence[int],
            bars_idx: Sequence[int],
    ) -> list[Event]:
        print(f"Computing genre for track: {track.name}")

        for genre in GENRE_TOKENS:
            if genre.lower() in track.name.lower():
                print(f"Found genre: {genre}")
                return [Event("GENRE", genre.upper(), -1)]
        
        print(f"No genre found in track name: {track.name}")
        return [Event("GENRE", "UNKNOWN", -1)]
        



def create_remi_tokenizer(midi_files: list[Path], tokenizer_path: Path, overwrite: bool = False) -> REMI:
    if tokenizer_path.exists() and not overwrite:
        # Load existing tokenizer
        print(f"Skipping creating tokenizer, found existing tokenizer at {tokenizer_path.resolve()}")
        tokenizer = REMI(params=tokenizer_path)
        return tokenizer

    print(f"Creating new tokenizer at {tokenizer_path.resolve()}")
    config = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 1): 12, (1, 4): 8, (4, 12): 4},
        special_tokens=["PAD", "BOS", "EOS"],
        use_chords=True,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        use_sustain_pedals=True,
        num_velocities=32,
        num_tempos=32,
        tempo_range=(40, 250),
    )
    tokenizer = REMI(config)
    tokenizer.train(vocab_size=30000, files_paths=midi_files)
    tokenizer.save(tokenizer_path)
    return tokenizer


def load_remi_tokenizer(tokenizer_path: Path) -> REMI:
    tokenizer = REMI(params=tokenizer_path)
    return tokenizer


if __name__ == "__main__":
    # Use the processed dataset instead of original
    file_path = Path("data/maestro")
    # file_path = Path("adl-piano-midi-processed")

    midi_files = list(file_path.rglob("*.mid")) + list(file_path.rglob("*.midi"))
    print("MIDI files found:", len(midi_files))

    # print(midi_files[:5]) 

    config = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 1): 12, (1, 4): 8, (4, 12): 4},
        special_tokens=["PAD", "BOS", "EOS"],
        use_chords=True,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        use_sustain_pedals=True,
        num_velocities=32,
        num_tempos=32,
        tempo_range=(40, 250),
    )
    tokenizer = REMI(config)
    tokenizer.add_attribute_control(AttributeControlGenre())

    tokenizer.train(vocab_size=30000, files_paths=midi_files)

    # Specify which attribute controls to apply during encoding
    # Format: {track_idx: {attribute_control_idx: track_level_boolean_or_bar_indices}}
    attribute_controls_indexes = {
        0: {0: True}  # Apply attribute control 0 (genre) to track 0 (track-level)
    }

    tokSeqs = tokenizer.encode(midi_files[0], attribute_controls_indexes=attribute_controls_indexes)
    print(tokenizer.vocab)
    for tokSeq in tokSeqs:
        print(tokSeq.tokens[:10])
