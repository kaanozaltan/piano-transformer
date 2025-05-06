import os
import numpy as np
import torch
from torch.utils.data import Dataset


class MAESTRO(Dataset):
    def __init__(self, data_dir, seq_len=512):
        self.seq_len = seq_len
        self.filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npy")]
        self.sequences = []

        for path in self.filepaths:
            ids = np.load(path)
            # Drop short sequences
            if len(ids) <= seq_len:
                continue

            # Slice into chunks of length seq_len + 1 (for input/target split)
            for i in range(0, len(ids) - seq_len):
                chunk = ids[i:i+seq_len+1]
                self.sequences.append(chunk)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": target_ids
        }