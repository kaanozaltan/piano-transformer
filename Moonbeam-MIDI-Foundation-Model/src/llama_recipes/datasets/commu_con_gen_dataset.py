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
import ast

class Commu_Con_Gen_Datasets(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        assert partition=="train" or partition=="val"
        self.data_dir = dataset_config.data_dir
        split_data = pd.read_csv(dataset_config.csv_file)
        file_basenames = split_data['file_base_name'].values
        chord_file_base_name = split_data['chord_file_base_name'].values
        bar_beat_chord_file_basenames = split_data['beat_bar_base_name'].values
        splits = split_data['split'].values       
        labels = split_data['label'].values 
        self.file_basenames = [f for f, s in zip(file_basenames, splits) if s == partition]
        self.chord_file_basenames = [f for f, s in zip(chord_file_base_name, splits) if s == partition]
        self.bar_beat_chord_file_basenames = [f for f, s in zip(bar_beat_chord_file_basenames, splits) if s == partition]
        self.metadata_ids = [l for l, s in zip(labels, splits) if s == partition]
        self.metadata_ids = [list(ast.literal_eval(l)) for l in self.metadata_ids]
        # self.emotion_ids2token = {0:tokenizer.emotion_token_4Q1, 1:tokenizer.emotion_token_4Q2, 2:tokenizer.emotion_token_4Q3, 3:tokenizer.emotion_token_4Q4}
        self.tokenizer = tokenizer
        self.if_add_chords_in_transformer = dataset_config.if_add_chords_in_transformer
        self.if_add_metadata_in_transformer = dataset_config.if_add_metadata_in_transformer 
        
    def __len__(self):
        return len(self.file_basenames)

    def __getitem__(self, index):
        raw_tokens = np.load(os.path.join(self.data_dir, "processed",self.file_basenames[index]))
        raw_tokens_chord = np.load(os.path.join(self.data_dir, "processed",self.chord_file_basenames[index]))
        raw_bar_beat_chord = np.load(os.path.join(self.data_dir, "processed",self.bar_beat_chord_file_basenames[index]))
        metadata_id = [[x for _ in range(6)] for x in self.metadata_ids[index]] #TODO: think of a better way to do this
        encoded_tokens = self.tokenizer.encode_series_con_gen_commu(raw_tokens, raw_tokens_chord, metadata_tokens = metadata_id, if_add_chords_in_transformer = self.if_add_chords_in_transformer, if_add_metadata_in_transformer = self.if_add_metadata_in_transformer)  #meta_data_tokens,<SOC> chords, <EOC>, <SOS> music_seq, <EOS>              
        encoded_tokens_label = self.tokenizer.encode_series_labels_con_gen_commu(encoded_tokens)

        return {
            "input_ids": encoded_tokens,
            "labels": encoded_tokens_label,
            "metadata_condition": [x[0] for x in metadata_id],
            "bar_beat_chord_condition": raw_bar_beat_chord.tolist(),
            "attention_mask":[] #Mask to be calculated dynamically during concatenation
        }


if __name__ == "__main__":
    from dataclasses import dataclass
    @dataclass
    class commu_con_gen_dataset:
        dataset: str = "commu_con_gen_dataset"
        train_split: str = "train"
        test_split: str = "test"
        data_dir: str = "/data/scratch/acw753/finetune/commu_con_gen"
        csv_file: str = "/data/scratch/acw753/finetune/commu_con_gen/train_test_split.csv"
        additional_token_dict_path: str = "/data/scratch/acw753/datasets/commu/commu_midi/indexed_tokens_dict.json"
        if_add_chords_in_transformer: bool = True
        if_add_metadata_in_transformer: bool = False
    from music_tokenizer import MusicTokenizer
    tokenizer = MusicTokenizer(timeshift_vocab_size = 4099, dur_vocab_size = 4099, octave_vocab_size = 13, pitch_class_vocab_size = 14, instrument_vocab_size = 131, velocity_vocab_size = 130, sos_token = -1, eos_token = -2, pad_token = -3)
    
    # tokenizer.add_new_tokens(token_name = "soc_token_compound", token_val = -4)
    # tokenizer.add_new_tokens(token_name = "eoc_token_compound", token_val = -5)
    #Open json and add new tokens to the tokenizer 
    with open(commu_con_gen_dataset().additional_token_dict_path, "r") as f:
        additional_token_dict = json.load(f)
    for key, value in additional_token_dict.items():
        tokenizer.add_new_tokens(token_name = key, token_val = value)

    dataset = Commu_Con_Gen_Datasets(dataset_config = commu_con_gen_dataset(), tokenizer=tokenizer, partition= "train")
    all_lengths = []
    for i in range(len(dataset)):
        compounds = dataset[i]["input_ids"]
        label = dataset[i]["labels"]
        # print("compounds", i, len(compounds), np.array(compounds))
        # print("label", i, len(label), np.array(label))
        # file_name = f"/data/home/acw753/musicllama/archive_logs_con_gen_emo/{i}_{label}.mid"
        # tokenizer.compound_to_midi(compounds[2:-1]).save(file_name)
        all_lengths.append(len(compounds))
        # print("compounds", len(compounds), len(label))
    #analyze mean, median, max, min of all_lengths
    print("mean", np.mean(all_lengths)) #nometa_nochords: #with chords:256, without chords: 61.7
    print("median", np.median(all_lengths)) #nometa_nochords: #with chords:247, without chords: 45
    print("max", np.max(all_lengths)) #nometa_nochords: 514 #with chords:847, without chords: 525
    print("min", np.min(all_lengths)) #nometa_nochords: #with chords:102, without chords: 14
 