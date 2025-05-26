from pathlib import Path

from miditok import REMI, TokenizerConfig


def create_remi_tokenizer(midi_files: list[Path], tokenizer_path: Path) -> REMI:
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
