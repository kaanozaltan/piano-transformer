# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from pkg_resources import packaging
from datetime import datetime
import contextlib


import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
import json
from mido import MidiFile

from llama_recipes.model_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint, save_model_checkpoint_ddp, save_peft_checkpoint
from llama_recipes.policies import fpSixteen,bfSixteen, get_llama_wrapper
from llama_recipes.utils.memory_utils import MemoryTrace
from accelerate.utils import is_xpu_available, is_ccl_available
from llama_recipes.utils.flop_utils import FlopMeasure


# Generation imports
import sys
import numpy as np
import random
import pickle
import tempfile
import subprocess
from transformers import LlamaConfig
from llama_recipes.datasets.music_tokenizer import MusicTokenizer
# Import existing generation functionality
recipes_path = os.path.join(os.path.dirname(__file__), '../../..', 'recipes/inference/custom_music_generation')
if recipes_path not in sys.path:
    sys.path.append(recipes_path)
from generation import MusicLlama

# Imports for evaluation functionality
import shutil
import importlib.util
from pathlib import Path as PathLib

# Import evaluation functions from piano_transformer metrics module
try:
    from llama_recipes.utils.metrics import create_subset, evaluate_mgeval_combined
    print("Successfully imported evaluation functions from Moonbeam metrics")
    
except ImportError as e:
    print(f"Error: {e}")
    create_subset = None
    evaluate_mgeval_combined = None

def is_valid_midi(file_path):
    try:
        MidiFile(file_path)
        return True
    except Exception as e:
        print(f"[INVALID MIDI] {file_path} - {e}")
        return False

def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

@contextlib.contextmanager
def profile(cfg, local_rank=None):
    use_profiler: bool = cfg.use_profiler
    use_flop_counter: bool = cfg.flop_counter
    if use_flop_counter and use_profiler:
        raise ValueError("Cannot use both profiler and flop counter")
    if use_profiler:
        # profiler needs a warmup stage to get the accurate profiling results
        wait_step, warmup_step, active_step = 1, 2, 3
        min_step = wait_step + warmup_step + active_step + 1
        if cfg.max_train_step > 0 and cfg.max_train_step < min_step:
            raise ValueError(f"pytorch profiler requires at least {min_step} train steps to finish the warm-up and recording stage, {wait_step} for wait_step, {warmup_step} for warmup_step, {active_step} for profiling step, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        print(f"pytorch profiling is activated and results will be saved in {cfg.profiler_dir}")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait_step, warmup=warmup_step, active=active_step, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                cfg.profiler_dir
            ),
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    elif use_flop_counter:
        if cfg.max_train_step > 0 and cfg.max_train_step <= cfg.flop_counter_start:
            raise ValueError(f"flop counter requires at least {cfg.flop_counter_start + 1} train steps, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        with FlopMeasure(rank=local_rank,warmup_step=cfg.flop_counter_start) as flop_counter:
            yield flop_counter
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


def train(model, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_scheduler, starting_epoch, starting_step,gradient_accumulation_steps, train_config, fsdp_config=None, ddp_config=None, local_rank=None, rank=None, wandb_run=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])



    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    if train_config.save_metrics:
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # Start the training loop
    for epoch in range(starting_epoch, train_config.num_epochs):
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch}", total=total_length, dynamic_ncols=True)
            with profile(train_config,local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader):
                    if step < starting_step and epoch == starting_epoch:  #skip until the starting step in the first continuing epoch
                        continue
                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank==0:
                            print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)
                        break
                    for key in batch.keys():
                        if train_config.enable_fsdp:
                            if is_xpu_available():
                                batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                            else:
                                batch[key] = batch[key].to(local_rank)
                        else:

                            if is_xpu_available():
                                batch[key] = batch[key].to('xpu:0')
                            else:
                                batch[key] = batch[key].to('cuda:0')
                    with autocast():
                        loss = model(**batch).loss
                    loss = loss / gradient_accumulation_steps
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                    total_loss += loss.detach().float()
                    if train_config.use_fp16:
                        # if fp16 is enabled, use gradient scaler to handle gradient update
                        scaler.scale(loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                scaler.unscale_(optimizer)
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            pbar.update(1)
                    else:
                        # regular backpropagation when fp16 is not used
                        loss.backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run:
                        if not train_config.enable_fsdp or rank==0:
                            wandb_run.log({
                                'train/epoch': epoch + 1,
                                'train/step': epoch * len(train_dataloader) + step,
                                'train/loss': loss.detach().float(),
                                'train/learning_rate': lr_scheduler.get_last_lr()[0],
                            })

                    pbar.set_description(f"Training Epoch: {epoch}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
                
                
                    #TODO: More frequent evaluation; Remember to switch on model.train again
                    if step%train_config.validation_interval==0 and train_config.run_validation:
                        
                        eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run, epoch=epoch, step=step, music_tokenizer=tokenizer)
                        if train_config.save_metrics:
                            val_step_loss.extend(temp_val_loss)
                            val_step_perplexity.extend(temp_step_perplexity)

                        checkpoint_start_time = time.perf_counter()
                        if train_config.save_model and eval_epoch_loss < best_val_loss:
                            if train_config.enable_fsdp:
                                dist.barrier()
                            if train_config.use_peft:
                                if train_config.enable_fsdp:
                                    if rank==0:
                                        print(f"we are about to save the PEFT modules")
                                else:
                                    print(f"we are about to save the PEFT modules")
                                model.save_pretrained(train_config.output_dir)
                                if train_config.enable_fsdp:
                                    if rank==0:
                                        print(f"PEFT modules are saved in {train_config.output_dir} directory")
                                else:
                                    print(f"PEFT modules are saved in {train_config.output_dir} directory")

                            else: #since we are training a smaller model, we are not using FDSP and PEFT
                                if train_config.enable_fsdp:
                                    if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:

                                        save_model_checkpoint(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                    elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                                        print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                                        print("=====================================================")

                                        save_model_and_optimizer_sharded(model, rank, train_config)
                                        if train_config.save_optimizer:
                                            save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                                            print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                                            print("=====================================================")

                                    if not train_config.use_peft and  train_config.save_optimizer:
                                        save_optimizer_checkpoint(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                        print(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                                        print("=====================================================")
                                elif train_config.enable_ddp: 
                                    if not train_config.use_peft:
                                        # Only save every 10 epochs in non-PEFT mode to save storage
                                        if epoch % 10 == 0:
                                            save_model_checkpoint_ddp(
                                                model, optimizer, rank, train_config, epoch=epoch, step=step
                                            )
                                            print(" Saving the DDP model checkpoints and optimizer using FULL_STATE_DICT")
                                            print("=====================================================")
                                        else:
                                            print(f" Skipping checkpoint save (non-PEFT mode, epoch {epoch}, next save at epoch {((epoch // 10) + 1) * 10})")
                                            print("=====================================================")
                                    else:
                                        print("Warning! Model Checkpoints are not saved properly")
                                        print("=====================================================")
                            if train_config.enable_fsdp:
                                dist.barrier()
                        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                        checkpoint_times.append(checkpoint_end_time)
                        if eval_epoch_loss < best_val_loss:
                            best_val_loss = eval_epoch_loss
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    print(f"best eval loss on epoch {epoch} is {best_val_loss}")
                            else:
                                print(f"best eval loss on epoch {epoch} is {best_val_loss}")
                        val_loss.append(float(best_val_loss))
                        val_prep.append(float(eval_ppl))     

                        """IMPORTANT"""         
                        model.train()
                
                
                
                
                pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if not train_config.enable_fsdp or rank==0:
            memtrace.print_stats()

        # Update the learning rate as needed
        lr_scheduler.step()

        # Log learning rate and epoch metrics to wandb
        if wandb_run:
            if not train_config.enable_fsdp or rank==0:
                wandb_run.log({
                    'epoch/learning_rate': lr_scheduler.get_last_lr()[0],
                    'epoch/train_loss': train_epoch_loss,
                    'epoch/train_perplexity': train_perplexity,
                    'epoch/epoch_time': epoch_end_time,
                    'epoch/epoch_number': epoch,
                })

        if train_config.enable_fsdp or train_config.enable_ddp:
            if rank==0:
                print(f"Epoch {epoch}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
        
        # Perform music generation after each epoch completes (safe from DataLoader conflicts)
        # Check evaluation frequency - only run if it's the right epoch
        should_evaluate = (epoch + 1) % train_config.evaluation_frequency_epochs == 0
        if train_config.enable_generation and should_evaluate:
            print(f"\n=== Music Generation for Epoch {epoch} (Frequency: every {train_config.evaluation_frequency_epochs} epochs) ===")
            try:
                generation_results = perform_music_generation(
                    model, tokenizer, train_config, local_rank, epoch=epoch, step=None
                )
                print(f"Generated {len(generation_results)} music sequences for epoch {epoch}")
                
                # Evaluate generated music against training set
                if train_config.enable_evaluation:
                    print(f"\n=== Evaluation Against Train Set for Epoch {epoch} ===")
                    try:
                        eval_metrics = evaluate_against_train_set(
                            generation_results, train_config, local_rank, epoch=epoch, step=None, wandb_run=wandb_run
                        )
                        print(f"Evaluation completed for epoch {epoch}")
                    except Exception as e:
                        print(f"Error during evaluation for epoch {epoch}: {e}")
                        import traceback
                        traceback.print_exc()
                        
            except Exception as e:
                print(f"Error during music generation for epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()

    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    if train_config.flop_counter:
        results["model_tflops"]= TFlops
    #saving the training params including fsdp setting for reference.
    if (train_config.enable_fsdp or train_config.enable_ddp) and not train_config.use_peft and rank==0:
        save_train_params(train_config, fsdp_config, rank)

    return results

def train_overfit(model, batch, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, fsdp_config=None, ddp_config=None, local_rank=None, rank=None, wandb_run=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        eval_dataloader: same as train_dataloader
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])

    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    if train_config.save_metrics:
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # Start the training loop
    for epoch in range(train_config.num_epochs):
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch}", total=total_length, dynamic_ncols=True)
            with profile(train_config,local_rank) as profile_context:

                for step, batch_unused in enumerate(train_dataloader):
                    # print("batch train: ", batch['input_ids'])
                    """
                    save data as npy file for visualization
                    """

                    # Save data as npy files for the first few steps for visualization
                    if step < 5:
                        import numpy as np
                        for key in batch.keys():
                            # Convert the tensor to a NumPy array (move to CPU if needed)
                            data_array = batch[key].cpu().numpy()
                            
                            # Save the NumPy array to a file with a unique name per key and step
                            np.save(f'/data/home/acw753/musicllama/dataset_analysis/{key}_step_{step}.npy', data_array)

                    if step > 1000:
                        break

                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank==0:
                            print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)
                        break
                    for key in batch.keys():
                        if train_config.enable_fsdp:
                            if is_xpu_available():
                                batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                            else:
                                batch[key] = batch[key].to(local_rank)
                        else:

                            if is_xpu_available():
                                batch[key] = batch[key].to('xpu:0')
                            else:
                                batch[key] = batch[key].to('cuda:0')
                    with autocast():
                        loss = model(**batch).loss
                    loss = loss / gradient_accumulation_steps
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                    total_loss += loss.detach().float()
                    if train_config.use_fp16:
                        # if fp16 is enabled, use gradient scaler to handle gradient update
                        scaler.scale(loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                scaler.unscale_(optimizer)
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            pbar.update(1)
                    else:
                        # regular backpropagation when fp16 is not used
                        loss.backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run:
                        if not train_config.enable_fsdp or rank==0:
                            wandb_run.log({
                                'train/epoch': epoch + 1,
                                'train/step': epoch * len(train_dataloader) + step,
                                'train/loss': loss.detach().float(),
                                'train/learning_rate': lr_scheduler.get_last_lr()[0],
                            })

                    pbar.set_description(f"Training Epoch: {epoch}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
                
                
                    #TODO: More frequent evaluation; Remember to switch on model.train again
                    if step%train_config.validation_interval==0 and train_config.run_validation:
                        
                        eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity, generation_logits, generation_hidden_state, logits_shrinked = evaluation_overfit(model, train_config, batch, eval_dataloader, local_rank, tokenizer, wandb_run)

                        if train_config.save_metrics:
                            val_step_loss.extend(temp_val_loss)
                            val_step_perplexity.extend(temp_step_perplexity)

                        checkpoint_start_time = time.perf_counter()
                        if train_config.save_model and eval_epoch_loss < best_val_loss:
                            if train_config.enable_fsdp:
                                dist.barrier()
                            if train_config.use_peft:
                                if train_config.enable_fsdp:
                                    if rank==0:
                                        print(f"we are about to save the PEFT modules")
                                else:
                                    print(f"we are about to save the PEFT modules")
                                model.save_pretrained(train_config.output_dir)
                                if train_config.enable_fsdp:
                                    if rank==0:
                                        print(f"PEFT modules are saved in {train_config.output_dir} directory")
                                else:
                                    print(f"PEFT modules are saved in {train_config.output_dir} directory")

                            else: #since we are training a smaller model, we are not using FDSP and PEFT
                                if train_config.enable_fsdp:
                                    if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:

                                        save_model_checkpoint(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                    elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                                        print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                                        print("=====================================================")

                                        save_model_and_optimizer_sharded(model, rank, train_config)
                                        if train_config.save_optimizer:
                                            save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                                            print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                                            print("=====================================================")

                                    if not train_config.use_peft and  train_config.save_optimizer:
                                        save_optimizer_checkpoint(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                        print(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                                        print("=====================================================")
                                elif train_config.enable_ddp: 
                                    if not train_config.use_peft:
                                        save_model_checkpoint_ddp(
                                            model, optimizer, rank, train_config, epoch=epoch, step=step
                                        )
                                        torch.save(generation_logits, f'/data/scratch/acw753/MusicLlama/ddp-MusicLlama-decoder_overfitting/generation_logits_epoch_{epoch}_step_{step}.pt')
                                        torch.save(generation_hidden_state, f'/data/scratch/acw753/MusicLlama/ddp-MusicLlama-decoder_overfitting/generation_hidden_state_epoch_{epoch}_step_{step}.pt')
                                        torch.save(logits_shrinked, f'/data/scratch/acw753/MusicLlama/ddp-MusicLlama-decoder_overfitting/logits_shrinked_epoch_{epoch}_step_{step}.pt')
                                        print(f"generation logits and hidden states saved to /data/scratch/acw753/MusicLlama/ddp-MusicLlama-decoder_overfitting/generation_logits_epoch_{epoch}_step_{step}.pt")
                                        print(" Saving the DDP model checkpoints and optimizer using FULL_STATE_DICT")
                                        print("=====================================================")
                                    else:
                                        print("Warning! Model Checkpoints are not saved properly")
                                        print("=====================================================")
                            if train_config.enable_fsdp:
                                dist.barrier()
                        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                        checkpoint_times.append(checkpoint_end_time)
                        if eval_epoch_loss < best_val_loss:
                            best_val_loss = eval_epoch_loss
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    print(f"best eval loss on epoch {epoch} is {best_val_loss}")
                            else:
                                print(f"best eval loss on epoch {epoch} is {best_val_loss}")
                        val_loss.append(float(best_val_loss))
                        val_prep.append(float(eval_ppl))     

                        """IMPORTANT"""         
                        model.train()
                
                
                
                
                pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if not train_config.enable_fsdp or rank==0:
            memtrace.print_stats()

        # Update the learning rate as needed
        lr_scheduler.step()

        # Log learning rate and epoch metrics to wandb
        if wandb_run:
            if not train_config.enable_fsdp or rank==0:
                wandb_run.log({
                    'epoch/learning_rate': lr_scheduler.get_last_lr()[0],
                    'epoch/train_loss': train_epoch_loss,
                    'epoch/train_perplexity': train_perplexity,
                    'epoch/epoch_time': epoch_end_time,
                    'epoch/epoch_number': epoch,
                })

        if train_config.enable_fsdp or train_config.enable_ddp:
            if rank==0:
                print(f"Epoch {epoch}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
        
        # Perform music generation after each epoch completes (safe from DataLoader conflicts)
        # Check evaluation frequency - only run if it's the right epoch
        should_evaluate = (epoch + 1) % train_config.evaluation_frequency_epochs == 0
        if train_config.enable_generation and should_evaluate:
            print(f"\n=== Music Generation for Epoch {epoch} (Frequency: every {train_config.evaluation_frequency_epochs} epochs) ===")
            try:
                generation_results = perform_music_generation(
                    model, tokenizer, train_config, local_rank, epoch=epoch, step=None
                )
                print(f"Generated {len(generation_results)} music sequences for epoch {epoch}")
                
                # Evaluate generated music against training set
                if train_config.enable_evaluation:
                    print(f"\n=== Evaluation Against Train Set for Epoch {epoch} ===")
                    try:
                        eval_metrics = evaluate_against_train_set(
                            generation_results, train_config, local_rank, epoch=epoch, step=None, wandb_run=wandb_run
                        )
                        print(f"Evaluation completed for epoch {epoch}")
                    except Exception as e:
                        print(f"Error during evaluation for epoch {epoch}: {e}")
                        import traceback
                        traceback.print_exc()
                        
            except Exception as e:
                print(f"Error during music generation for epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()

    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    if train_config.flop_counter:
        results["model_tflops"]= TFlops
    #saving the training params including fsdp setting for reference.
    if (train_config.enable_fsdp or train_config.enable_ddp) and not train_config.use_peft and rank==0:
        save_train_params(train_config, fsdp_config, rank)

    return results

def train_con_gen(model, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_scheduler, starting_epoch, starting_step,gradient_accumulation_steps, train_config, fsdp_config=None, ddp_config=None, local_rank=None, rank=None, wandb_run=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])



    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    if train_config.save_metrics:
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # Start the training loop
    for epoch in range(starting_epoch, train_config.num_epochs):
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch}", total=total_length, dynamic_ncols=True)
            with profile(train_config,local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader):
                    if step < starting_step and epoch == starting_epoch:  #skip until the starting step in the first continuing epoch
                        continue
                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank==0:
                            print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)
                        break
                    for key in batch.keys():
                        if train_config.enable_fsdp:
                            if is_xpu_available():
                                batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                            else:
                                batch[key] = batch[key].to(local_rank)
                        else:

                            if is_xpu_available():
                                batch[key] = batch[key].to('xpu:0')
                            else:
                                batch[key] = batch[key].to('cuda:0')
                    with autocast():
                        loss = model(**batch).loss
                    loss = loss / gradient_accumulation_steps
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                    total_loss += loss.detach().float()
                    if train_config.use_fp16:
                        # if fp16 is enabled, use gradient scaler to handle gradient update
                        scaler.scale(loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                scaler.unscale_(optimizer)
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            pbar.update(1)
                    else:
                        # regular backpropagation when fp16 is not used
                        loss.backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run:
                        if not train_config.enable_fsdp or rank==0:
                            wandb_run.log({
                                'train/epoch': epoch + 1,
                                'train/step': epoch * len(train_dataloader) + step,
                                'train/loss': loss.detach().float(),
                                'train/learning_rate': lr_scheduler.get_last_lr()[0],
                            })

                    pbar.set_description(f"Training Epoch: {epoch}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
                
                
                    #TODO: More frequent evaluation; Remember to switch on model.train again
                    if step%train_config.validation_interval==0:
                        eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run, epoch=epoch, step=step, music_tokenizer=tokenizer)
                        if train_config.save_metrics:
                            val_step_loss.extend(temp_val_loss)
                            val_step_perplexity.extend(temp_step_perplexity)

                        checkpoint_start_time = time.perf_counter()
                        if train_config.save_model:
                            if train_config.enable_fsdp:
                                dist.barrier()
                            if train_config.use_peft:
                                if train_config.enable_fsdp:
                                    if rank==0:
                                        print(f"we are about to save the PEFT modules")
                                else:
                                    print(f"we are about to save the PEFT modules")
                                # model.save_pretrained(train_config.output_dir)
                                save_peft_checkpoint(model, train_config.output_dir, epoch=epoch, step = step)
                                if train_config.enable_fsdp:
                                    if rank==0:
                                        print(f"PEFT modules are saved in {train_config.output_dir} directory")
                                else:
                                    print(f"PEFT modules are saved in {train_config.output_dir} directory")

                            else: #since we are training a smaller model, we are not using FDSP and PEFT
                                if train_config.enable_fsdp:
                                    if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:

                                        save_model_checkpoint(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                    elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                                        print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                                        print("=====================================================")

                                        save_model_and_optimizer_sharded(model, rank, train_config)
                                        if train_config.save_optimizer:
                                            save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                                            print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                                            print("=====================================================")

                                    if not train_config.use_peft and  train_config.save_optimizer:
                                        save_optimizer_checkpoint(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                        print(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                                        print("=====================================================")
                                elif train_config.enable_ddp: 
                                    if not train_config.use_peft:
                                        # Only save every 10 epochs in non-PEFT mode to save storage
                                        if epoch % 10 == 0:
                                            save_model_checkpoint_ddp(
                                                model, optimizer, rank, train_config, epoch=epoch, step=step
                                            )
                                            print(" Saving the DDP model checkpoints and optimizer using FULL_STATE_DICT")
                                            print("=====================================================")
                                        else:
                                            print(f" Skipping checkpoint save (non-PEFT mode, epoch {epoch}, next save at epoch {((epoch // 10) + 1) * 10})")
                                            print("=====================================================")
                                    else:
                                        print("Warning! Model Checkpoints are not saved properly")
                                        print("=====================================================")
                            if train_config.enable_fsdp:
                                dist.barrier()
                        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                        checkpoint_times.append(checkpoint_end_time)
                        if eval_epoch_loss < best_val_loss:
                            best_val_loss = eval_epoch_loss
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    print(f"best eval loss on epoch {epoch} is {best_val_loss}")
                            else:
                                print(f"best eval loss on epoch {epoch} is {best_val_loss}")
                        val_loss.append(float(best_val_loss))
                        val_prep.append(float(eval_ppl))    

                        #IMPORTANT        
                        model.train()
                
                
                
                
                pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if not train_config.enable_fsdp or rank==0:
            memtrace.print_stats()

        # Update the learning rate as needed
        lr_scheduler.step()

        # Log learning rate and epoch metrics to wandb
        if wandb_run:
            if not train_config.enable_fsdp or rank==0:
                wandb_run.log({
                    'epoch/learning_rate': lr_scheduler.get_last_lr()[0],
                    'epoch/train_loss': train_epoch_loss,
                    'epoch/train_perplexity': train_perplexity,
                    'epoch/epoch_time': epoch_end_time,
                    'epoch/epoch_number': epoch,
                })

        if train_config.enable_fsdp or train_config.enable_ddp:
            if rank==0:
                print(f"Epoch {epoch}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
        
        # Perform music generation after each epoch completes (safe from DataLoader conflicts)
        # Check evaluation frequency - only run if it's the right epoch
        should_evaluate = (epoch + 1) % train_config.evaluation_frequency_epochs == 0
        if train_config.enable_generation and should_evaluate:
            print(f"\n=== Music Generation for Epoch {epoch} (Frequency: every {train_config.evaluation_frequency_epochs} epochs) ===")
            try:
                generation_results = perform_music_generation(
                    model, tokenizer, train_config, local_rank, epoch=epoch, step=None
                )
                print(f"Generated {len(generation_results)} music sequences for epoch {epoch}")
                
                # Evaluate generated music against training set
                if train_config.enable_evaluation:
                    print(f"\n=== Evaluation Against Train Set for Epoch {epoch} ===")
                    try:
                        eval_metrics = evaluate_against_train_set(
                            generation_results, train_config, local_rank, epoch=epoch, step=None, wandb_run=wandb_run
                        )
                        print(f"Evaluation completed for epoch {epoch}")
                    except Exception as e:
                        print(f"Error during evaluation for epoch {epoch}: {e}")
                        import traceback
                        traceback.print_exc()
                        
            except Exception as e:
                print(f"Error during music generation for epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()

    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    if train_config.flop_counter:
        results["model_tflops"]= TFlops
    #saving the training params including fsdp setting for reference.
    if (train_config.enable_fsdp or train_config.enable_ddp) and not train_config.use_peft and rank==0:
        save_train_params(train_config, fsdp_config, rank)

    return results


def evaluation(model,train_config, eval_dataloader, local_rank, tokenizer, wandb_run, epoch=None, step=None, music_tokenizer=None):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions
        epoch: Current training epoch (for logging)
        step: Current training step (for logging)
        music_tokenizer: Optional pre-created MusicTokenizer (unused, kept for compatibility)

    Returns: eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    # eval_preds = []
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0  # Initialize evaluation loss
    total_eval_steps = 0
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            total_eval_steps += 1
            # stop when the maximum number of eval steps is reached
            if train_config.max_eval_step > 0 and total_eval_steps > train_config.max_eval_step:
                if not train_config.enable_fsdp or local_rank==0:
                    print("max eval steps reached, stopping evaluation, total_eval_steps: ", total_eval_steps - 1)
                break
            for key in batch.keys():
            # Move to correct device
                if train_config.enable_fsdp:
                    if is_xpu_available():
                        batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                    else:
                        batch[key] = batch[key].to(local_rank)
                else:
                    if is_xpu_available():
                        batch[key] = batch[key].to('xpu:0')
                    else:
                        batch[key] = batch[key].to('cuda:0')

            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                if train_config.save_metrics:
                    val_step_loss.append(loss.detach().float().item())
                    val_step_perplexity.append(float(torch.exp(loss.detach().float())))

                eval_loss += loss.detach().float()

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank==0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=}")

    # Music generation moved to after epoch completion to avoid DataLoader conflicts
    generation_results = []

    if wandb_run:
        wandb_log_data = {
            'eval/perplexity': eval_ppl,
            'eval/loss': eval_epoch_loss,
        }
        # Log generation statistics if available
        # Generation results logging moved to epoch completion
        
        wandb_run.log(wandb_log_data, commit=False)

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity


def evaluation_overfit(model,train_config, batch, eval_dataloader, local_rank, tokenizer, wandb_run):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    # eval_preds = []
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0  # Initialize evaluation loss
    total_eval_steps = 0
    with MemoryTrace() as memtrace:
        for step, batch_unused in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            if step > 1:
                break
            total_eval_steps += 1
            # stop when the maximum number of eval steps is reached
            if train_config.max_eval_step > 0 and total_eval_steps > train_config.max_eval_step:
                if not train_config.enable_fsdp or local_rank==0:
                    print("max eval steps reached, stopping evaluation, total_eval_steps: ", total_eval_steps - 1)
                break
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    if is_xpu_available():
                        batch[key] = batch[key].to('xpu:0')
                    else:
                        batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                """ check generation logits and targets  """

                generation_logits = outputs.generation_logits #batch * len_x, decoder_vocab_size

                batch_size = batch['input_ids'].shape[0]
                length = batch['input_ids'].shape[1]-1 
                no_attributes = 6


                generation_logits_reshaped = torch.reshape(generation_logits, (batch_size, length, no_attributes, -1))

                # print(f"generation_logits:{generation_logits_reshaped.shape}")
                max_values, max_indices = torch.max(generation_logits_reshaped, dim=-1)
                # print(f"max_indices:{max_indices.shape}, {max_indices}")
                torch.save(generation_logits_reshaped, "/data/scratch/acw753/MusicLlama/ddp-MusicLlama-decoder_overfitting/batch_data_train_logits.pth")
                torch.save(max_indices, "/data/scratch/acw753/MusicLlama/ddp-MusicLlama-decoder_overfitting/batch_data_train_logits_max.pth")

                
                try:
                    decoded_tokens = tokenizer.convert_from_language_tokens(torch.max(generation_logits, dim=-1))
                    torch.save(torch.tensor(decoded_tokens), "/data/scratch/acw753/MusicLlama/ddp-MusicLlama-decoder_overfitting/batch_data_train_logits_max_decoded_tokens.pth")
                    print(f"decoded_tokens:{decoded_tokens}")
                except:
                    print(f"failed to decode tokens")

                if train_config.save_metrics:
                    val_step_loss.append(loss.detach().float().item())
                    val_step_perplexity.append(float(torch.exp(loss.detach().float())))

                eval_loss += loss.detach().float()

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    # eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_epoch_loss = eval_loss / 2
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank==0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=}")

    if wandb_run:
        wandb_run.log({
                        'eval/perplexity': eval_ppl,
                        'eval/loss': eval_epoch_loss,
                    }, commit=False)

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity, outputs.generation_logits, outputs.generation_hidden_state, outputs.logits


def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False


def check_frozen_layers_peft_model(model):
     for i, layer in enumerate(model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def get_model_config(model):
    """
    Safely get model config, handling wrapped models (DDP, FSDP)
    """
    if hasattr(model, 'module'):
        # Model is wrapped (DDP, FSDP, etc.)
        return getattr(model.module, 'config', None)
    else:
        # Model is not wrapped
        return getattr(model, 'config', None)


def evaluate_against_train_set(generation_results, train_config, local_rank, epoch=None, step=None, wandb_run=None):
    """
    Evaluate generated sequences against training set using mgeval metrics
    Similar to EvalCallback from piano_transformer but without FMD
    """
    if not train_config.enable_evaluation:
        return {}
    
    if create_subset is None or evaluate_mgeval_combined is None:
        print("Evaluation functions not available, skipping evaluation")
        return {}
    
    if not generation_results:
        print("No generation results to evaluate")
        return {}
    
    # Only run evaluation on main process
    if train_config.enable_fsdp and local_rank != 0:
        return {}
    
    try:
        print(f"[Evaluation] Starting evaluation against train set with {len(generation_results)} generated samples...")
        
        # Create temporary directory for generated MIDI files
        with tempfile.TemporaryDirectory() as temp_gen_dir:
            temp_gen_path = PathLib(temp_gen_dir)
            
            # Convert generation results to MIDI files
            midi_count = 0
            for i, result in enumerate(generation_results):
                try:
                    if 'generation' in result and 'content' in result['generation']:
                        # Save generated MIDI
                        midi_path = temp_gen_path / f'generated_{i}.mid'
                        result['generation']['content'].save(str(midi_path))
                        midi_count += 1
                    else:
                        print(f"Warning: No MIDI content found in generation result {i}")
                except Exception as e:
                    print(f"Error saving MIDI file {i} for evaluation: {e}")
            
            if midi_count == 0:
                print("No valid MIDI files generated for evaluation")
                return {}
            
            print(f"[Evaluation] Saved {midi_count} MIDI files for evaluation")
            
            # Create subset of training data
            ref_dir = PathLib(train_config.evaluation_ref_dir)
            if not ref_dir.exists():
                print(f"Warning: Reference directory {ref_dir} does not exist")
                return {}
            
            train_dir_subset = create_subset(ref_dir, train_config.generation_num_samples)
            print(f"[Evaluation] Created training subset with {train_config.generation_num_samples} files from {ref_dir}")
            
            # Compute evaluation metrics
            metrics = {}
            
            # Compute mgeval metrics
            try:
                print(f"[Evaluation] Starting mgeval evaluation between {train_dir_subset} and {temp_gen_path}")
                relative_summary = evaluate_mgeval_combined(
                    dataset1_path=train_dir_subset,
                    dataset2_path=temp_gen_path,
                )
                
                print(f"[Evaluation] Got results: {len(relative_summary)} relative items")
                
                # Process only relative summary for averages
                kld_sum = 0
                oa_sum = 0
                for item in relative_summary:
                    kld_sum += item["KLD"]
                    oa_sum += item["OA"]
                
                if len(relative_summary) > 0:
                    metrics["KLD_average"] = kld_sum / len(relative_summary)
                    metrics["OA_average"] = oa_sum / len(relative_summary)
                    print(f"[Evaluation] Computed averages - KLD: {metrics['KLD_average']:.4f}, OA: {metrics['OA_average']:.4f}")
                else:
                    metrics["KLD_average"] = 0.0
                    metrics["OA_average"] = 0.0
                    print(f"[Evaluation] No relative summary data, setting averages to 0.0")
                
                print(f"[Evaluation] Total metrics computed: {len(metrics)}")
                
            except Exception as e:
                print(f"Error computing mgeval metrics: {e}")
                import traceback
                traceback.print_exc()
            
            # Log metrics to wandb
            if wandb_run and "KLD_average" in metrics and "OA_average" in metrics:
                try:
                    # Log metrics if available
                    wandb_log_data = {
                        "custom_eval/KLD_average": float(metrics["KLD_average"]),
                        "custom_eval/OA_average": float(metrics["OA_average"])
                    }
                    
                    # Create success message with available metrics
                    log_message = f"[Evaluation] Successfully logged KLD_average={metrics['KLD_average']:.4f}, OA_average={metrics['OA_average']:.4f}"
                    
                except Exception as e:
                    print(f"Error logging to wandb: {e}")
                    import traceback
                    traceback.print_exc()
            elif wandb_run is None:
                print("[Evaluation] wandb_run is None - wandb logging disabled")
            elif not ("KLD_average" in metrics and "OA_average" in metrics):
                print(f"[Evaluation] Missing required metrics - KLD_average: {'KLD_average' in metrics}, OA_average: {'OA_average' in metrics}")
            else:
                print("[Evaluation] Unknown condition preventing wandb logging")
            
            print(f"[Evaluation] Evaluation metrics: {metrics}")
            
            # Print the averages
            if "KLD_average" in metrics and "OA_average" in metrics:
                key_metrics_msg = f"[Evaluation] Key metrics - KLD_average: {metrics['KLD_average']:.4f}, OA_average: {metrics['OA_average']:.4f}"
                print(key_metrics_msg)
            return metrics
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {}


def perform_music_generation(model, tokenizer, train_config, local_rank, epoch=None, step=None):
    """
    Perform music generation during evaluation using MusicLlama library
    """
    if not train_config.enable_generation:
        return []
    
    print(f"Starting music generation with {train_config.generation_num_samples} samples...")
    
    try:
        # Handle DDP wrapping
        if hasattr(model, 'module'):
            unwrapped_model = model.module
            model_config = unwrapped_model.config
        else:
            # Model is not wrapped
            unwrapped_model = model
            model_config = model.config
        
        unwrapped_model.eval()
        
        # Get model device and dtype
        model_device = next(unwrapped_model.parameters()).device
        model_dtype = next(unwrapped_model.parameters()).dtype
        print(f"Model device: {model_device}, dtype: {model_dtype}")
        
        
        def run_generation_safe():
            """Run generation after DataLoader is finished"""
            
            # Save current default tensor type and device
            original_default_dtype = torch.get_default_dtype()
            original_cuda_device = torch.cuda.current_device() if model_device.type == 'cuda' else None
            
            # Save the device or dtype
            try:
                original_tensor_type = torch.tensor([]).type()  # Get default type
                print(f"DEBUG: Original tensor type before generation: {original_tensor_type}")
            except:
                original_tensor_type = "torch.FloatTensor"  # Fallback to CPU float tensor
                print(f"DEBUG: Failed to get original tensor type, using fallback: {original_tensor_type}")
            
            try:
                print(f"Running music generation with {train_config.generation_num_samples} samples...")
                
                # Set CUDA context for generation
                if model_device.type == 'cuda':
                    torch.cuda.set_device(model_device)
                    
                    # Set default tensor type
                    if model_dtype == torch.bfloat16:
                        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
                    elif model_dtype == torch.float16:
                        torch.set_default_tensor_type(torch.cuda.HalfTensor)
                    else:
                        torch.set_default_tensor_type(torch.cuda.FloatTensor)
                
                # Create MusicLlama wrapper
                print(f"DEBUG: Using fine-tuned model with {sum(p.numel() for p in unwrapped_model.parameters())} parameters")
                print(f"DEBUG: Model is in {'training' if unwrapped_model.training else 'eval'} mode")
                print(f"DEBUG: Model memory address: {id(unwrapped_model)} (same as training model)")
                music_llama = MusicLlama(unwrapped_model, tokenizer, model_config)
                
                # Create generation prompts based on generation mode, still just dummy for random_files and all_test_files
                prompts = []
                if train_config.generation_mode == "from_scratch":
                    prompts = [[tokenizer.sos_token_compound] for _ in range(train_config.generation_num_samples)]
                elif train_config.generation_mode == "random_files":
                    prompts = [[tokenizer.sos_token_compound] for _ in range(train_config.generation_num_samples)]
                elif train_config.generation_mode == "all_test_files":
                    prompts = [[tokenizer.sos_token_compound] for _ in range(train_config.generation_num_samples)]
                else:
                    raise ValueError(f"Invalid generation_mode: {train_config.generation_mode}. Must be one of: 'from_scratch', 'random_files', 'all_test_files'")
                
                # Use existing music_completion
                results = music_llama.music_completion(
                    prompts,
                    max_gen_len=train_config.generation_max_gen_len,
                    temperature=train_config.generation_temperature,
                    top_p=train_config.generation_top_p,
                )
                
                print(f"Generated {len(results)} music sequences successfully!")
                return results
                
            finally:
                # Always restore original state
                if model_device.type == 'cuda' and original_cuda_device is not None:
                    torch.cuda.set_device(original_cuda_device)
                
                # Restore the original default tensor type (device + dtype)
                try:
                    if original_tensor_type == "torch.FloatTensor":
                        torch.set_default_tensor_type(torch.FloatTensor)  # CPU Float
                    elif original_tensor_type == "torch.cuda.FloatTensor":
                        torch.set_default_tensor_type(torch.cuda.FloatTensor)  # CUDA Float
                    elif original_tensor_type == "torch.cuda.HalfTensor":
                        torch.set_default_tensor_type(torch.cuda.HalfTensor)  # CUDA Half
                    elif original_tensor_type == "torch.cuda.BFloat16Tensor":
                        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)  # CUDA BFloat16
                    else:
                        # Fallback to CPU float tensor for unknown types
                        torch.set_default_tensor_type(torch.FloatTensor)
                        print(f"Warning: Unknown original tensor type {original_tensor_type}, falling back to CPU FloatTensor")
                    
                    # Debug: Verify restoration worked
                    restored_type = torch.tensor([]).type()
                    print(f"DEBUG: Restored tensor type after generation: {restored_type}")
                    
                except Exception as e:
                    # Fallback to just setting dtype if tensor type fails
                    torch.set_default_dtype(original_default_dtype)
                    print(f"Warning: Failed to restore tensor type, restored dtype only: {e}")
        
        # Run generation safely after DataLoader is finished  
        results = run_generation_safe()
        
        # Put model back in train mode
        unwrapped_model.train()
        
        # Save generation results
        if results and (not train_config.enable_fsdp or local_rank == 0):
            save_generation_results(results, train_config, epoch, step)
        
        return results
        
    except Exception as e:
        # Ensure model is back in train mode even on error
        try:
            if 'unwrapped_model' in locals():
                unwrapped_model.train()
        except:
            pass  # Ignore errors when setting train mode
            
        print(f"Error during music generation: {e}")
        import traceback
        traceback.print_exc()
        return []


def save_generation_results(generated_sequences, train_config, epoch=None, step=None):
    """
    Save generated sequences to MIDI files
    """
    try:
        if not generated_sequences:
            print("No generation results to save (generation was skipped or failed)")
            return
        # Create save directory
        save_dir = train_config.generation_save_dir
        if epoch is not None and step is not None:
            save_dir = os.path.join(save_dir, f"epoch_{epoch}_step_{step}")
        elif epoch is not None:
            save_dir = os.path.join(save_dir, f"epoch_{epoch}")
        
        os.makedirs(save_dir, exist_ok=True)

    
        # Save individual sequences as MIDI files
        saved_count = 0
        for i, result in enumerate(generated_sequences):
            try:
                if 'generation' in result and 'content' in result['generation']:
                    # Save generated MIDI
                    midi_path = os.path.join(save_dir, f'generated_{i}.mid')
                    result['generation']['content'].save(midi_path)
                    
                    # Check if the saved MIDI file is valid
                    if is_valid_midi(midi_path):
                        saved_count += 1
                        print(f"Valid MIDI saved to {midi_path}")
                    else:
                        # Delete invalid MIDI file
                        try:
                            os.remove(midi_path)
                            print(f"Invalid MIDI deleted: {midi_path}")
                        except:
                            print(f"Invalid MIDI (couldn't delete): {midi_path}")
                        continue
                    
                    # Save prompt MIDI only for prompts longer than 1 token
                    print(f"DEBUG: Generation mode = '{train_config.generation_mode}'")
                    
                    if (train_config.generation_mode != "from_scratch" and 
                        'prompt' in result['generation'] and 
                        'prompt_tokens' in result['generation']):
                        
                        prompt_tokens = result['generation']['prompt_tokens']
                        print(f"DEBUG: Prompt has {len(prompt_tokens)} tokens for generated_{i}")
                        
                        if len(prompt_tokens) > 0:
                            prompt_path = os.path.join(save_dir, f'generated_{i}_prompt.mid')
                            result['generation']['prompt'].save(prompt_path)
                            
                            # Validate MIDI prompt
                            if is_valid_midi(prompt_path):
                                print(f"Valid prompt MIDI saved to {prompt_path}")
                            else:
                                try:
                                    os.remove(prompt_path)
                                    print(f"Invalid prompt MIDI deleted: {prompt_path}")
                                except:
                                    print(f"Invalid prompt MIDI (couldn't delete): {prompt_path}")
                        else:
                            print(f"Skipping prompt save for generated_{i} (empty prompt after SOS removal)")
                    else:
                        print(f"Skipping prompt save for generated_{i} (from_scratch mode or no prompt data)")
                        
            except Exception as e:
                print(f"Error saving MIDI file {i}: {e} - Skipping this generation")
                # Skip failed generations
                continue
        
        print(f"Saved {saved_count}/{len(generated_sequences)} valid MIDI files to {save_dir}")
        
    except Exception as e:
        print(f"Error saving generation results: {e}")


def setup():
    """Initialize the process group for distributed training"""
    if is_ccl_available():
        # distributed training on xpus
        dist.init_process_group("ccl")
    else:
        dist.init_process_group("nccl")        


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    if is_xpu_available():
        torch.xpu_empty_cache()
    else:
        torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes

def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params / 1e6:.2f} Million")
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")
        print(f"Trainable %: {(trainable_params / total_params) * 100:.2f}%\n")


def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""


    verify_bfloat_support = ((
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    ) or
    (is_xpu_available()))


    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy

def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    + "-"
    + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            print(f"training params are saved in {file_name}")

def save_to_json(output_filename, train_step_loss, train_epoch_loss, train_step_ppl, train_epoch_ppl, val_step_loss, val_epoch_loss, val_step_ppl, val_epoch_ppl):
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_perplexity": train_step_ppl,
        "train_epoch_perplexity": train_epoch_ppl,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "val_step_perplexity": val_step_ppl,
        "val_epoch_perplexity": val_epoch_ppl
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)
