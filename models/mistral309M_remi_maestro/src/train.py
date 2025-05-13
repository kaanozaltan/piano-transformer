import os
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import wandb
from dotenv import load_dotenv
from evaluate import load as load_metric
from miditok import REMI, TokenizerConfig
from miditok.data_augmentation import augment_dataset
from miditok.pytorch_data import DataCollator, DatasetMIDI
from miditok.utils import split_files_for_training
from sklearn.model_selection import train_test_split
from torch import argmax
from torch.cuda import is_available as cuda_available
from torch.cuda import is_bf16_supported
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    MistralConfig,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed

load_dotenv()

SEED = 222
set_seed(SEED)

TRANSFORMER_NAME = "mistral-309M"
TOKENIZER_NAME = "remi"
DATASET_NAME = "maestro"
MODEL_VERSION = "1"

MODEL_NAME = f"{TRANSFORMER_NAME}_{TOKENIZER_NAME}_{DATASET_NAME}_v{MODEL_VERSION}"

print(f"Model:\n{MODEL_NAME}")

# BASE_PATH = Path(".")
BASE_PATH = Path("/hpcwork/gd010186/piano-transformer")

DATA_RAW_PATH = BASE_PATH / "data"

MODEL_BASE_PATH = BASE_PATH / "models" / MODEL_NAME
DATA_PROCESSED_PATH = MODEL_BASE_PATH / "data_processed"
MODEL_PATH = MODEL_BASE_PATH / "model"
RUNS_PATH = MODEL_BASE_PATH / "runs"
OUTPUT_PATH = MODEL_BASE_PATH / "output"

os.environ["WANDB_PROJECT"] = "piano-transformer"
os.environ["WANDB_ENTITY"] = "jonathanlehmkuhl-rwth-aachen-university"
wandb.login()

tokenizer = REMI(params=MODEL_BASE_PATH / "tokenizer.json")

# Create pytorch datasets
midi_files_train = list((DATA_PROCESSED_PATH / "maestro_train").glob("**/*.midi"))
midi_files_valid = list((DATA_PROCESSED_PATH / "maestro_valid").glob("**/*.midi"))
midi_files_test = list((DATA_PROCESSED_PATH / "maestro_train").glob("**/*.midi"))

dataset_kwargs = {
    "max_seq_len": 2048,
    "tokenizer": tokenizer,
    "bos_token_id": tokenizer["BOS_None"],
    "eos_token_id": tokenizer["EOS_None"],
}
dataset_train = DatasetMIDI(midi_files_train, **dataset_kwargs)
dataset_valid = DatasetMIDI(midi_files_valid, **dataset_kwargs)
dataset_test = DatasetMIDI(midi_files_test, **dataset_kwargs)

collator = DataCollator(tokenizer["PAD_None"], copy_inputs_as_labels=True)

model_config = MistralConfig(
    vocab_size=len(tokenizer),
    hidden_size=896,
    intermediate_size=896 * 4,
    num_hidden_layers=24,
    num_attention_heads=14,
    num_key_value_heads=14,
    sliding_window=2048,
    max_position_embeddings=2048,
    pad_token_id=tokenizer["PAD_None"],
    bos_token_id=tokenizer["BOS_None"],
    eos_token_id=tokenizer["EOS_None"],
)
model = AutoModelForCausalLM.from_config(model_config)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


def preprocess_logits(logits, _):
    pred_ids = argmax(logits, dim=-1)
    return pred_ids


USE_CUDA = cuda_available()
if not cuda_available():
    FP16 = FP16_EVAL = BF16 = BF16_EVAL = False
elif is_bf16_supported():
    BF16 = BF16_EVAL = True
    FP16 = FP16_EVAL = False
else:
    BF16 = BF16_EVAL = False
    FP16 = FP16_EVAL = True

if USE_CUDA:
    print("Using GPU")

trainer_config = TrainingArguments(
    output_dir=RUNS_PATH,
    overwrite_output_dir=False,
    do_train=True,
    do_eval=True,
    do_predict=False,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=3,
    eval_strategy="epoch",
    eval_accumulation_steps=None,
    # eval_steps=5,
    learning_rate=1e-4,
    weight_decay=0.01,
    max_grad_norm=3.0,
    # max_steps=5,
    num_train_epochs=20,
    lr_scheduler_type="cosine_with_restarts",
    warmup_ratio=0.3,
    log_level="debug",
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="epoch",
    # save_steps=5,
    # save_total_limit=5,
    no_cuda=not USE_CUDA,
    seed=SEED,
    fp16=FP16,
    fp16_full_eval=FP16_EVAL,
    bf16=BF16,
    bf16_full_eval=BF16_EVAL,
    load_best_model_at_end=True,
    label_smoothing_factor=0.0,
    optim="adamw_torch",
    report_to=["wandb"],
    run_name=MODEL_NAME,
    gradient_checkpointing=True,
)
trainer = Trainer(
    model=model,
    args=trainer_config,
    data_collator=collator,
    train_dataset=dataset_train,
    eval_dataset=dataset_valid,
    # compute_metrics=compute_metrics,
    callbacks=None,
    preprocess_logits_for_metrics=preprocess_logits,
)

train_result = trainer.train()
trainer.save_model(MODEL_PATH)
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()
