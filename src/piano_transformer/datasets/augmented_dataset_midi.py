import random

from miditok.constants import SCORE_LOADING_EXCEPTION
from miditok.data_augmentation import augment_score
from miditok.pytorch_data import DatasetMIDI
from symusic import Score
from torch import LongTensor
from tqdm import tqdm

from piano_transformer.utils.midi import scale_tempo


# Pytorch dataset class for augmented MIDI data
# Subclassing DatasetMIDI of miditok
# allowing for augmentation of the data along multiple axes


class AugmentedDatasetMIDI(DatasetMIDI):
    def __init__(
        self,
        *args,
        pitch_offsets=[0],
        velocity_offsets=[0],
        duration_offsets=[0],
        tempo_factors=[1.0],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pitch_offsets = pitch_offsets
        self.velocity_offsets = velocity_offsets
        self.duration_offsets = duration_offsets
        self.tempo_factors = tempo_factors

        if self.pre_tokenize:
            raise ValueError("pre_tokenize is not supported for AugmentedDatasetMIDI")

        # Preload Score objects (in memory)
        self._scores = []
        for path in tqdm(
            self.files_paths,
            desc="Loading midi files into memory",
            miniters=int(len(self.files_paths) / 20),
            maxinterval=480,
        ):
            try:
                score = Score(path)
                self._scores.append(score)
            except SCORE_LOADING_EXCEPTION:
                self._scores.append(None)  # keep indices aligned


    def __getitem__(self, idx: int) -> dict[str, LongTensor]:
        """
        Return the ``idx`` elements of the dataset.

        If the dataset is pre-tokenized, the method will return the token ids.
        Otherwise, it will tokenize the ``idx``th file on the fly. If the file to is
        corrupted, the method will return a dictionary with ``None`` values.

        :param idx: idx of the file/sample.
        :return: the token ids, with optionally the associated label.
        """
        labels = None
        score = self._scores[idx]

        if score is None:
            item = {self.sample_key_name: None}
            if self.func_to_get_labels is not None:
                item[self.labels_key_name] = labels
            return item

        # Randomly sample the offsets from the lists
        pitch_offset = random.choice(self.pitch_offsets)
        velocity_offset = random.choice(self.velocity_offsets)
        duration_offset = random.choice(self.duration_offsets)
        tempo_factor = random.choice(self.tempo_factors)

        augmented_score = augment_score(score, pitch_offset, velocity_offset, duration_offset)
        augmented_score = scale_tempo(augmented_score, tempo_factor)

        tseq = self._tokenize_score(augmented_score)
        # If not one_token_stream, we only take the first track/sequence
        token_ids = tseq.ids if self.tokenizer.one_token_stream else tseq[0].ids

        if self.func_to_get_labels is not None:
            # tokseq can be given as a list of TokSequence to get the labels
            labels = self.func_to_get_labels(augmented_score, tseq, self.files_paths[idx])
            if not isinstance(labels, LongTensor):
                labels = LongTensor([labels] if isinstance(labels, int) else labels)

        item = {self.sample_key_name: LongTensor(token_ids)}
        if self.func_to_get_labels is not None:
            item[self.labels_key_name] = labels

        return item
