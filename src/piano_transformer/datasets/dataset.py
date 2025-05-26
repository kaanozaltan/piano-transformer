from miditok import MusicTokenizer
from miditok.pytorch_data import DataCollator, DatasetMIDI

from piano_transformer.datasets.augmented_dataset_midi import AugmentedDatasetMIDI


def build_datasets(
    chunks_lists: list,
    tokenizer: MusicTokenizer,
    max_seq_len: int,
    aug_cfg: dict,
):
    dataset_kwargs = {
        "max_seq_len": max_seq_len,
        "tokenizer": tokenizer,
        "bos_token_id": tokenizer["BOS_None"],
        "eos_token_id": tokenizer["EOS_None"],
    }
    augmentation_kwargs = aug_cfg

    train_ds = AugmentedDatasetMIDI(
        chunks_lists["train"], **augmentation_kwargs, **dataset_kwargs
    )
    valid_ds = DatasetMIDI(chunks_lists["validation"], **dataset_kwargs)
    test_ds = DatasetMIDI(chunks_lists["test"], **dataset_kwargs)

    return train_ds, valid_ds, test_ds


def build_collator(tokenizer):
    return DataCollator(tokenizer["PAD_None"], copy_inputs_as_labels=True)
