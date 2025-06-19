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

class Emophia_Con_Gen_Datasets(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        assert partition=="train" or partition=="test"
        self.data_dir = dataset_config.data_dir
        split_data = pd.read_csv(dataset_config.csv_file)
        file_basenames = split_data['file_base_name'].values
        splits = split_data['split'].values       
        labels = split_data['label'].values 
        self.file_basenames = [f for f, s in zip(file_basenames, splits) if s == partition]
        self.emotion_ids = [l for l, s in zip(labels, splits) if s == partition]
        self.emotion_ids2token = {0:tokenizer.emotion_token_4Q1, 1:tokenizer.emotion_token_4Q2, 2:tokenizer.emotion_token_4Q3, 3:tokenizer.emotion_token_4Q4}
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.file_basenames)

    def __getitem__(self, index):
        raw_tokens = np.load(os.path.join(self.data_dir, "processed",self.file_basenames[index]))
        emotion_id = self.emotion_ids[index]
        encoded_tokens = self.tokenizer.encode_series_con_gen_emotion(raw_tokens, if_add_sos = True, if_add_eos = True, emotion_token_4Q = self.emotion_ids2token[emotion_id]) #SOS and EOS are added in the tokenizer
        encoded_tokens_label = self.tokenizer.encode_series_labels_con_gen_emotion(encoded_tokens, if_added_sos = True, if_added_eos = True, if_added_emotion_token_4Q = True)

        return {
            "input_ids": encoded_tokens,
            "labels": encoded_tokens_label,
            "attention_mask":[] #Mask to be calculated dynamically during concatenation
        }


