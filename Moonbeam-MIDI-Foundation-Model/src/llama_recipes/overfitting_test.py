# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import json
import dataclasses
import fire
import random
import torch
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_kbit_training
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM, 
    LlamaConfig,
)
from llama_recipes.datasets.music_tokenizer import MusicTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import ddp_config as DDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.data.concatenator import ConcatDataset
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset

from llama_recipes.utils.fsdp_utils import hsdp_device_mesh
from llama_recipes.utils.train_utils import (
    train,
    train_overfit,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
from accelerate.utils import is_xpu_available

def setup_wandb(train_config, fsdp_config, llama_config, **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    from llama_recipes.configs import wandb_config as WANDB_CONFIG
    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    run.config.update(fsdp_config, allow_val_change=True)

    # Convert the llama_config to a dictionary and then to a JSON string
    config_dict = llama_config.to_dict()
    config_json = json.dumps(config_dict, indent=4)
    
    # Get the wandb run directory
    from pathlib import Path
    # Define the file path within the wandb run directory
    folder_name = (train_config.dist_checkpoint_root_folder+ "/"+ train_config.dist_checkpoint_folder+ "-"+ train_config.model_name)
    save_dir = Path.cwd() / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    config_file_path = os.path.join(save_dir, 'llama_config.json')

    # Write the JSON string to the file
    with open(config_file_path, 'w') as f:
        f.write(config_json)
        print(f"config file saved to {config_file_path}!")
    return run
from torch.utils.data import Dataset, DataLoader

class ExtendedDataset(Dataset):
    def __init__(self, dataset, target_length):
        self.dataset = dataset
        self.target_length = target_length

    def __len__(self):
        return self.target_length  # Set the artificially extended length

    def __getitem__(self, index):
        # Use modulo to repeat the dataset if the index exceeds the actual dataset length
        return self.dataset[index % len(self.dataset)]

def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config, ddp_config = TRAIN_CONFIG(), FSDP_CONFIG(), DDP_CONFIG()
    model_config_path = "src/llama_recipes/configs/model_config.json"
    llama_config = LlamaConfig.from_pretrained(model_config_path)

    update_config((train_config, fsdp_config, ddp_config), **kwargs)
    print("updated training config", train_config)
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp or train_config.enable_ddp: #TODO: what is this? Read this  https://arxiv.org/pdf/2304.11277#:~:text=Fully%20Sharded%20Data%20Parallel%20(FSDP)%20is%20capable%20of%20scaling%20to,by%20sharding%20the%20dense%20parameters. 
        setup() #enable nccl / ccl
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    wandb_run = None

    model_dictionary = {
        "LlamaForCausalLM": LlamaForCausalLM
    }

    llama_model = model_dictionary[llama_config.architectures]

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp or train_config.enable_ddp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        if rank == 0:
            model = llama_model.from_pretrained( #TODO: If new model then need to change here! source code: transformers/src/transformers/models/llama/modeling_llama.py
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
                attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            )
        else:
            llama_config = LlamaConfig.from_pretrained(train_config.model_name)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = llama_model(llama_config)

    else: #DDP and non-distributed training
        llama_config.use_cache = use_cache
        model = llama_model(llama_config) 


    if train_config.use_wandb: #TODO update ddp config
        if not train_config.enable_fsdp or rank==0:
            wandb_run = setup_wandb(train_config, fsdp_config, llama_config, **kwargs)


    # Load the tokenizer and add special tokens
    tokenizer = MusicTokenizer(timeshift_vocab_size = llama_config.onset_vocab_size, dur_vocab_size = llama_config.dur_vocab_size, octave_vocab_size = llama_config.octave_vocab_size, pitch_class_vocab_size = llama_config.pitch_class_vocab_size, instrument_vocab_size = llama_config.instrument_vocab_size, velocity_vocab_size = llama_config.velocity_vocab_size, sos_token = llama_config.sos_token, eos_token = llama_config.eos_token, pad_token = llama_config.pad_token)

    # If there is a mismatch between tokenizer vocab size and embedding matrix, 
    # throw a warning and then expand the embedding matrix
    # if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
    #     print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
    #     model.resize_token_embeddings(len(tokenizer)) #Commented out since there's no tokenizer here

    print_model_size(model, train_config, rank if train_config.enable_fsdp or train_config.enable_ddp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    if train_config.enable_ddp and ddp_config.pure_bf16:
        model.to(torch.bfloat16)

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs) #TODO: Understand what is PEFT, FSDP
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        if wandb_run:
            wandb_run.config.update(peft_config)

    hsdp_device_mesh = None #TODO change this to include ddp
    if fsdp_config.hsdp and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
        hsdp_device_mesh = hsdp_device_mesh(replica_group_size=fsdp_config.replica_group_size, sharding_group_size=fsdp_config.sharding_group_size)
        print("HSDP device mesh is ready")

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:

            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer) #TODO: dangerous

        device_id = 0
        if is_xpu_available():
            device_id = torch.xpu.current_device()
        elif torch.cuda.is_available():
            device_id = torch.cuda.current_device()

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_mesh=hsdp_device_mesh,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=(lambda module: module.to_empty(device=torch.device("cuda"), recurse=False))
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model) #TODO: Add DDP
    elif train_config.enable_ddp: #wrap ddp code
        mixed_precision_policy, wrapping_policy = get_policies(ddp_config, rank)
        model.to(local_rank)
        model = DDP(model,
                    mixed_precision=mixed_precision_policy if not ddp_config.pure_bf16 else None, 
                    device_mesh=hsdp_device_mesh,
                    device_ids=[local_rank],
                    find_unused_parameters=True #this is changed from True
                    )
    elif not train_config.quantization and not train_config.enable_fsdp:
        if is_xpu_available():
            model.to("xpu:0")
        elif torch.cuda.is_available():
            model.to("cuda")

    dataset_config = generate_dataset_config(train_config, kwargs)

     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    if not train_config.enable_fsdp or rank == 0:
            print(f"--> Validation Set Length = {len(dataset_val)}")

    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length, split="train",data_dir = dataset_config.data_dir) 
        dataset_train = ExtendedDataset(dataset_train, 466789) #avoid small gradient during training
    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")

    """manually change shuffle to False"""
    import torch.distributed as dist
    from torch.utils.data import DistributedSampler

    train_dl_kwargs["sampler"] = DistributedSampler(
    dataset_train,
    rank=dist.get_rank(),
    num_replicas=dist.get_world_size(),
    shuffle=False,
    drop_last=True,
    )
    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length, split="val", data_dir = dataset_config.data_dir )
            dataset_val = ExtendedDataset(dataset_val, 466789) #avoid small gradient

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # Start the training process
    batch = next(iter(train_dataloader))
    """reduce the length, otherwise too many tokens"""
    for key in batch.keys():
        batch[key] = batch[key][:, :128]
        
    results = train_overfit(
        model,
        batch,
        train_dataloader,
        train_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        ddp_config if train_config.enable_ddp else None,
        local_rank if (train_config.enable_fsdp or train_config.enable_ddp) else None, #TODO: change this and train_utils
        rank if (train_config.enable_fsdp or train_config.enable_ddp) else None,#TODO: change this and train_utils
        wandb_run,
    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]
        if train_config.use_wandb:
            for k,v in results.items():
                wandb_run.summary[k] = v

if __name__ == "__main__":
    fire.Fire(main)
