# Moonbeam: A MIDI Foundation Model Using Both Absolute and Relative Music Attributes

[Paper Link](https://arxiv.org/pdf/2505.15559)  
[Demo Website](https://aim-qmul.github.io/moonbeam-midi-foundation-model/)  
[Checkpoint](https://huggingface.co/guozixunnicolas/moonbeam-midi-foundation-model) 

## Introduction

Moonbeam is a transformer-based foundation model for symbolic music, pretrained on a large and diverse collection of MIDI data totaling 81.6K and 18 billion tokens. Unlike most existing transformer-based models for symbolic music, Moonbeam incorporates music-domain inductive biases by capturing both absolute and relative musical attributes through the introduction of a novel domain-knowledge-inspired tokenization method and a novel attention mechanism

## Instructions

### 1. Install Dependencies
```bash
conda create --name moonbeam_test python=3.12
pip install . 
pip install src/llama_recipes/transformers_minimal/.
```

### 2. Download Checkpoints 
Download the pre-trained checkpoints from the following link:
- [Pre-trained Checkpoint](https://huggingface.co/guozixunnicolas/moonbeam-midi-foundation-model)
- [Unconditional Generation Checkpoint (ATEPP-Bach)](https://TODO)
- [Conditional Generation Checkpoint (Commu)](https://TODO)

## Finetuning
If you modify any configuration files in the `src/llama_recipes` directory, remember to reinstall the package.
```bash
pip install src/llama_recipes/transformers_minimal/.
```
### 1. Unconditional Music Generation 
#### Data Preprocessing: 
Download your dataset to `/PATH/TO/MIDI/FOLDER`. The below script will recursively find all the midi data; preprocess them; and randomly select 90% as training data and 10% as test data. If your dataset includes a predefined train/test split file, specify its path using the `--train_test_split_file` flag. The data will then be split according to the predefined split file.
```bash
python data_preprocess.py \
  --dataset_name /DATASET/NAME \
  --dataset_folder /PATH/TO/MIDI/FOLDER \
  --output_folder /PATH/TO/PREPROCESSED/DATA \
  --model_config src/llama_recipes/configs/model_config.json \
  --train_test_split_file None \
  --train_ratio 0.9 \
  --ts_threshold None
```

#### Update the dataset configuration:
Edit the `lakhmidi_dataset` class in `src/llama_recipes/configs/datasets.py`. Set the correct paths for `data_dir` and `csv_file` to match your dataset location (`/PATH/TO/PREPROCESSED/DATA`). Then reinstall the package: `pip install src/llama_recipes/transformers_minimal/.`.

#### Finetuning: 
```bash
torchrun --nnodes 1 --nproc_per_node 1 recipes/finetuning/real_finetuning_uncon_gen.py \
  --lr 3e-4 \
  --val_batch_size 2 \
  --run_validation True \
  --validation_interval 10 \
  --save_metrics True \
  --dist_checkpoint_root_folder checkpoints/finetuned_checkpoints/ATEPP_bach_uncon_gen \
  --dist_checkpoint_folder ddp \
  --trained_checkpoint_path /PATH/TO/PRETRAINED/CHECKPOINT \
  --pure_bf16 True \
  --enable_ddp True \
  --use_peft True \
  --peft_method lora \
  --quantization False \
  --model_name ATEPP_bach \
  --dataset lakhmidi_dataset \
  --output_dir checkpoints/finetuned_checkpoints/ATEPP_bach_uncon_gen \
  --batch_size_training 2 \
  --context_length 2048 \
  --num_epochs 300 \
  --use_wandb False \
  --gamma 0.99
```
#### Inferencing: 
```bash
torchrun --nproc_per_node 1 recipes/inference/custom_music_generation/unconditional_music_generation.py \
  --csv_file /PATH/TO/CSV \
  --top_p 0.95 \
  --temperature 0.9 \
  --model_config_path src/llama_recipes/configs/model_config.json \
  --ckpt_dir /PATH/TO/PRETRAINED/CHECKPOINT \
  --finetuned_PEFT_weight_path /PATH/TO/PEFT/WEIGHT \
  --tokenizer_path tokenizer.model \
  --max_seq_len 512 \
  --max_gen_len 512 \
  --max_batch_size 6 \
  --num_test_data 20 \
  --prompt_len 50
```
### 2. Conditional Music Generation and Music Infilling 
#### Switch to the branch for conditional music generation and music infilling: 
```bash
git checkout conditional_gen_commu
```
#### Data Preprocessing:
```bash
python data_preprocess.py \
  --dataset_name commu_con_gen \
  --dataset_folder /PATH/TO/COMMU/MIDI \
  --output_folder /PATH/TO/PREPROCESSED/DATA \
  --model_config src/llama_recipes/configs/model_config.json \
  --train_test_split_file /PATH/TO/COMMU/META/CSV \
  --train_ratio None \
  --ts_threshold None
```

#### Inferencing:
```bash
torchrun --nproc_per_node 1 recipes/inference/custom_music_generation/conditional_music_generation_batch.py \
  --csv_file /PATH/TO/COMMU/META/CSV \
  --top_p 0.6 \
  --temperature 0.7 \
  --model_config_path src/llama_recipes/configs/model_config_commu_con_gen.json \
  --ckpt_dir /PATH/TO/PRETRAINED/CHECKPOINT \
  --finetuned_PEFT_weight_path /PATH/TO/PEFT/WEIGHT \
  --additional_token_dict_path /PATH/TO/ADDITIONAL/TOKEN/DICT/PATH \
  --tokenizer_path tokenizer.model \
  --max_seq_len 600 \
  --max_gen_len 600 \
  --max_batch_size 4 \
  --if_add_chords_in_transformer True \
  --if_add_metadata_in_transformer True
```

#### Finetuning: 
```bash
torchrun --nnodes 1 --nproc_per_node 1 recipes/finetuning/real_finetuning_con_gen.py \
  --lr 3e-4 \
  --val_batch_size 15 \
  --run_validation True \
  --validation_interval 120 \
  --save_metrics True \
  --dist_checkpoint_root_folder /PATH/TO/OUTPUT/FOLDER \
  --dist_checkpoint_folder ddp \
  --trained_checkpoint_path /PATH/TO/PRETRAINED/CHECKPOINT \
  --pure_bf16 True \
  --enable_ddp True \
  --use_peft True \
  --peft_method lora \
  --quantization False \
  --model_name commu_con_gen \
  --dataset commu_con_gen_dataset \
  --output_dir /PATH/TO/OUTPUT/FOLDER \
  --batch_size_training 15 \
  --context_length 848 \
  --num_epochs 300 \
  --use_wandb True
```

### 3. Music Classification
#### Switch to the branch for music classification: 
```bash
git checkout finetune_player_classification
```
#### Data Preprocessing:

To preprocess datasets for music classification, run `data_preprocess.py` with the appropriate dataset configuration as shown below

| Dataset Name       | Dataset Folder                                      | Output Folder                                      | Train-Test Split File                              |
|--------------------|----------------------------------------------------|----------------------------------------------------|----------------------------------------------------|
| `pijama30`         | `datasets/classification/pijama30`                 | `processed_datasets/classification/pijama30_album_split_0` | `datasets/classification/pijama30/pijama30_finetune.csv` |
| `pianist8`        | `datasets/classification/pianist8`                | `processed_datasets/classification/pianist8`       | `dummy.csv`                                        |
| `emopia`           | `datasets/classification/emopia2.2/midis`         | `processed_datasets/classification/emopia2.2_1071_clips` | `datasets/classification/emopia2.2/split`         |
| `Giant_Piano_MIDI` | `datasets/classification/gpm30/surname_checked_midis` | `processed_datasets/classification/gpm30`         | `datasets/classification/gpm30/gpm30_finetune.csv` |

#### Example for Giant_Piano_MIDI:

```bash
python data_preprocess.py \
  --dataset_name Giant_Piano_MIDI \
  --dataset_folder datasets/classification/gpm30/surname_checked_midis \
  --output_folder processed_datasets/classification/gpm30 \
  --model_config src/llama_recipes/configs/model_config.json \
  --train_test_split_file datasets/classification/gpm30/gpm30_finetune.csv \
  --train_ratio 1 \
  --ts_threshold None
```
#### Finetuning:

To finetune a music classification model, update the following configuration files based on the dataset-specific parameters in the table below:

| Dataset Name       | Sliding Window Length | Finetuning Sequence Length | LoRA Target Modules | Learning Rate | Number of Classes |
|--------------------|-----------------------|----------------------------|---------------------|---------------|-------------------|
| `pijama30`         | 1200                  | 1203                       | q, k, v, o          | 3.00E-04      | 30                |
| `pianist8`         | 900                   | 903                        | q, k, v, o          | 1.60E-04      | 8                 |
| `emopia`           | 130                   | 133                        | q, k                | 2.00E-04      | 4                 |
| `Giant_Piano_MIDI` | 1200                  | 1203                       | q, k, v, o          | 2.40E-04      | 30                |

#### Configuration Files to Update:
- `src/llama_recipes/configs/player_classification_config.json`: Set `num_classes` to the "Number of Classes" from the table.
- `src/llama_recipes/configs/dataset.py` (in `player_classification_dataset` class):
  - Set `seq_len` to the "Sliding Window Length" from the table.
  - Set `data_dir` to the "Output Folder" from the preprocessing table.
  - Set `csv_file` to the "Train-Test Split File" from the preprocessing table.
- `src/llama_recipes/configs/peft.py`: Set `target_modules` to the "LoRA Target Modules" from the table.

#### Run the finetuning script with the appropriate dataset parameters:
```bash
torchrun --nnodes 1 --nproc_per_node 1 recipes/finetuning/real_finetuning_player_classification.py \
  --lr <LR> \
  --val_batch_size 22 \
  --run_validation True \
  --validation_interval 399 \
  --save_metrics True \
  --dist_checkpoint_root_folder /PATH/TO/OUTPUT/FOLDER \
  --dist_checkpoint_folder ddp \
  --trained_checkpoint_path /PATH/TO/PRETRAINED/CHECKPOINT \
  --pure_bf16 True \
  --enable_ddp True \
  --use_peft True \
  --peft_method lora \
  --quantization False \
  --model_name player_classification \
  --dataset player_classification \
  --output_dir /PATH/TO/OUTPUT/FOLDER \
  --batch_size_training 5 \
  --context_length <Finetuning Sequence Length in Above Table> \
  --num_epochs 300 \
  --use_wandb True \
  --use_cache False \
  --individual_eval True
```
## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Bibtex

```bibtex
@article{guo2025moonbeam,
  title={Moonbeam: A MIDI Foundation Model Using Both Absolute and Relative Music Attributes},
  author={Guo, Zixun and Dixon, Simon},
  journal={arXiv preprint arXiv:2505.15559},
  year={2025},
  url={https://arxiv.org/abs/2505.15559}
}
