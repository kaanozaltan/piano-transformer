# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str="PATH/to/Model"
    tokenizer_name: str=None
    enable_fsdp: bool=False
    enable_ddp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    validation_interval: int=200
    batch_size_training: int=4
    batching_strategy: str="packing" #alternative: padding
    context_length: int=1024
    gradient_accumulation_steps: int=1
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=3
    max_train_step: int=0
    max_eval_step: int=0
    num_workers_dataloader: int=4
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    scheduler_type: str = "steplr" # Learning rate scheduler: "steplr" or "cosine"
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "samsum_dataset"
    peft_method: str = "lora" # None, llama_adapter (Caution: llama_adapter is currently not supported with FSDP)
    use_peft: bool=False
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    trained_checkpoint_path: str = "PATH/to/saved/trained/model"
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    use_wandb: bool = False # Enable wandb for experient tracking
    save_metrics: bool = False # saves training metrics to a json file for later plotting
    flop_counter: bool = False # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
    flop_counter_start: int = 3 # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
    use_profiler: bool = False # Enable pytorch profiler, can not be used with flop counter at the same time.
    profiler_dir: str = "PATH/to/save/profiler/results" # will be used if using profiler
    # Generation parameters
    enable_generation: bool = False # Enable music generation during evaluation
    generation_temperature: float = 1 # Temperature for generation sampling
    generation_top_p: float = 0.95 # Top-p value for nucleus sampling
    generation_max_gen_len: int = 256 # Maximum generation length
    generation_prompt_len: int = 512 # Length of prompt for generation (when using data for prompts)
    generation_num_samples: int = 20 # Number of samples to generate during evaluation
    generation_mode: str = "random_files" # Generation mode: "from_scratch", "random_files", or "all_test_files"
    generation_max_prompt_samples: int = 200 # Max number of prompts to sample for prompt-based generation (will duplicate if fewer test files)
    generation_save_dir: str = "PATH/to/save/generation/results" # Directory to save generated music
    # Evaluation parameters
    enable_evaluation: bool = False # Enable evaluation against training set after generation
    evaluation_ref_dir: str = "PATH/to/training/data" # Reference directory containing training MIDI files
