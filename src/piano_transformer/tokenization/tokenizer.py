from pathlib import Path

from miditok import REMI, TokenizerConfig


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
