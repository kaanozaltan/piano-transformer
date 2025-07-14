import os

from miditok.pytorch_data import DataCollator
from torch.cuda import is_available as cuda_available
from torch.cuda import is_bf16_supported
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def make_trainer(
    cfg: dict,
    model: AutoModelForCausalLM,
    data_collator: DataCollator,
    train_dataset: Dataset,
    eval_dataset: Dataset,
):
    """
    Constructs and returns a HuggingFace Trainer configured for training, evaluation, and optional prediction.
    """
    # Determine precision settings based on CUDA and BF16 support
    use_cuda = cuda_available()
    if not use_cuda:
        fp16 = bf16 = fp16_eval = bf16_eval = False
    elif is_bf16_supported():
        bf16 = bf16_eval = True
        fp16 = fp16_eval = False
    else:
        bf16 = bf16_eval = False
        fp16 = fp16_eval = True

    # Prepare TrainingArguments
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        do_predict=False,
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        eval_accumulation_steps=None,
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        lr_scheduler_kwargs={"min_lr_rate": cfg["min_lr_rate"]},
        warmup_ratio=cfg["warmup_ratio"],
        log_level="debug",
        logging_strategy="steps",
        logging_steps=cfg["logging_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        num_train_epochs=cfg["num_train_epochs"],
        seed=cfg["seed"],
        data_seed=cfg["data_seed"],
        bf16=bf16,
        fp16=fp16,
        bf16_full_eval=bf16_eval,
        fp16_full_eval=fp16_eval,
        run_name=cfg["run_name"],
        optim=cfg["optim"],
        report_to=["wandb"],
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=False,
        max_steps=cfg["max_steps"] if "max_steps" in cfg else -1,
    )

    # Instantiate Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # preprocess_logits_for_metrics=preprocess_logits,
        # compute_metrics=compute_metrics,
    )

    return trainer
