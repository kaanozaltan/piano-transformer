import copy
import json

import torch
from torch.utils.data import Dataset
import glob
import mido
from collections import defaultdict
import json
from concurrent.futures import ProcessPoolExecutor
import tqdm
import numpy as np
import pandas as pd
import os

class MergeDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        assert partition=="train" or partition=="test"
        self.data_dir = dataset_config.data_dir
        split_data = pd.read_csv(dataset_config.csv_file)
        file_basenames = split_data['file_base_name'].values
        splits = split_data['split'].values        
        self.file_basenames = [f for f, s in zip(file_basenames, splits) if s == partition]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.file_basenames)

    def __getitem__(self, index):
        raw_tokens = np.load(os.path.join(self.data_dir,self.file_basenames[index]))
        
        encoded_tokens = self.tokenizer.encode_series(raw_tokens, if_add_sos = True, if_add_eos = True) #SOS and EOS are added in the tokenizer
        
        encoded_tokens_label = self.tokenizer.encode_series_labels(encoded_tokens, if_added_sos = True, if_added_eos = True)

        return {
            "input_ids": encoded_tokens,
            "labels": encoded_tokens_label,
            "attention_mask":[] #Mask to be calculated dynamically during concatenation
        }