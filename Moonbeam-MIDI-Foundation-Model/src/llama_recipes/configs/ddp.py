# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

@dataclass
class ddp_config:
    mixed_precision: bool=True
    use_fp16: bool=False
    hsdp : bool =False # Require HYBRID_SHARD to be set. This flag can extend the HYBRID_SHARD by allowing sharding a model on customized number of GPUs (Sharding_group) and Replicas over Sharding_group.
    sharding_group_size : int=0 # requires hsdp to be set. This specifies the sharding group size, number of GPUs that you model can fit into to form a replica of a model.
    replica_group_size: int=0 #requires hsdp to be set. This specifies the replica group size, which is world_size/sharding_group_size.    fsdp_activation_checkpointing: bool=True
    fsdp_cpu_offload: bool=False
    pure_bf16: bool = False
    optimizer: str= "AdamW"
    
