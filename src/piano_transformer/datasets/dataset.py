from miditok import MusicTokenizer
from miditok.pytorch_data import DataCollator, DatasetMIDI

from piano_transformer.datasets.augmented_dataset_midi import AugmentedDatasetMIDI


def build_datasets(
    chunks_lists: list,
    tokenizer: MusicTokenizer,
    max_seq_len: int,
    aug_cfg: dict = None,
):
    dataset_kwargs = {
        "max_seq_len": max_seq_len,
        "tokenizer": tokenizer,
        "bos_token_id": tokenizer["BOS_None"],
        "eos_token_id": tokenizer["EOS_None"],
    }
    
    # Add attribute control parameters if tokenizer has attribute controls
    if tokenizer.attribute_controls:
        dataset_kwargs["ac_tracks_random_ratio_range"] = (1.0, 1.0)  # Always apply to all tracks
        
    augmentation_kwargs = aug_cfg

    if aug_cfg is not None:
        train_ds = AugmentedDatasetMIDI(chunks_lists["train"], **augmentation_kwargs, **dataset_kwargs)
    else:
        train_ds = DatasetMIDI(chunks_lists["train"], pre_tokenize=True, **dataset_kwargs)
    valid_ds = DatasetMIDI(chunks_lists["validation"], pre_tokenize=True, **dataset_kwargs)
    test_ds = DatasetMIDI(chunks_lists["test"], pre_tokenize=True, **dataset_kwargs)

    return train_ds, valid_ds, test_ds


def build_collator(tokenizer):
    return DataCollator(tokenizer["PAD_None"], copy_inputs_as_labels=True)
