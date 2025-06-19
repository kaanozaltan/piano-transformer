# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"
    
    
@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"

@dataclass
class lakhmidi_dataset:
    dataset: str = "maestro"
    train_split: str = "train"
    test_split: str = "test"
    data_dir: str = "preprocessed/maestro/"
    csv_file: str = "preprocessed/maestro/train_test_split.csv"

@dataclass
class maestro_309M:
    dataset: str = "maestro_309M"
    train_split: str = "train"
    test_split: str = "test"
    data_dir: str = "preprocessed/309M/"
    csv_file: str = "preprocessed/309M/train_test_split.csv"

@dataclass
class maestro_839M:
    dataset: str = "maestro_839M"
    train_split: str = "train"
    test_split: str = "test"
    data_dir: str = "preprocessed/839M/"
    csv_file: str = "preprocessed/839M/train_test_split.csv"

@dataclass
class merge_dataset:
    dataset: str = "merge_dataset"
    train_split: str = "train"
    test_split: str = "test"
    data_dir: str = "/PATH/TO/DATA/DIR"
    csv_file: str = "/PATH/TO/CSV"

@dataclass
class emophia_con_gen_dataset:
    dataset: str = "emophia_con_gen_dataset"
    train_split: str = "train"
    test_split: str = "test"
    data_dir: str = "/PATH/TO/DATA/DIR"
    csv_file: str = "/PATH/TO/CSV"

@dataclass
class commu_con_gen_dataset:
    dataset: str = "commu_con_gen_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_dir: str = "/PATH/TO/DATA/DIR"
    csv_file: str = "/PATH/TO/CSV"
    additional_token_dict_path: str = "/PATH/TO/JSON"
    