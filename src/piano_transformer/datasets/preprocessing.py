from pathlib import Path

from miditok import MusicTokenizer
from miditok.utils import split_files_for_training


def split_datasets_into_chunks(
    midi_files_lists: list[Path],
    tokenizer: MusicTokenizer,
    processed_path: Path,
    dataset_name: str,
    max_seq_len: int,
    num_overlap_bars: int,
):
    chunks_lists = {}
    for split in ["train", "validation", "test"]:
        save_dir = processed_path / f"{dataset_name}_{split}"
        split_files_for_training(
            files_paths=midi_files_lists[split],
            tokenizer=tokenizer,
            save_dir=save_dir,
            max_seq_len=max_seq_len,
            num_overlap_bars=num_overlap_bars,
        )
        chunks_lists[split] = list(
            (processed_path / f"{dataset_name}_{split}").glob("**/*.midi")
        )
    return chunks_lists
