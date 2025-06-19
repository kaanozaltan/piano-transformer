import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict, Dict
from accelerate.utils import is_xpu_available
import time
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaForCausalLM_Conditional_Generation,
    LlamaConfig,
)
from llama_recipes.datasets.music_tokenizer import MusicTokenizer

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)


class MusicLlama:
    @staticmethod
    def build(
        ckpt_dir: str,
        model_config_path: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        finetuned_PEFT_weight_path: str,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "MusicLlama":
        """
        Build a Llama instance by initializing and loading a model checkpoint.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.
        """

        # Set the seeds for reproducibility
        if is_xpu_available():
            torch.xpu.manual_seed(seed)
        else:
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

        llama_config = LlamaConfig.from_pretrained(model_config_path)
        model = LlamaForCausalLM(llama_config) 
        start_time = time.time()
        checkpoint = torch.load(ckpt_dir)
        checkpoint = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('module.'): # Check if the keys have 'module.' prefix and remove it if necessary
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if finetuned_PEFT_weight_path is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, finetuned_PEFT_weight_path)
            print("PEFT model loaded successfully")
  
        if is_xpu_available():
            model.to("xpu")
        else:
            model.to("cuda")
        model.eval()

        tokenizer = MusicTokenizer(timeshift_vocab_size = llama_config.onset_vocab_size, dur_vocab_size = llama_config.dur_vocab_size, octave_vocab_size = llama_config.octave_vocab_size, pitch_class_vocab_size = llama_config.pitch_class_vocab_size, instrument_vocab_size = llama_config.instrument_vocab_size, velocity_vocab_size = llama_config.velocity_vocab_size)
        
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
            model = model.to(torch.bfloat16)  # Explicitly cast the entire model to BF16 precision.
            print("model precision set to BF16")
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor) 

        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return MusicLlama(model, tokenizer, llama_config)

    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[List[int]]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """

        bsz = len(prompt_tokens)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= self.config.max_len 
        total_len = min(self.config.max_len, max_gen_len + max_prompt_len) 

        pad_id = self.tokenizer.pad_token_compound
        pad_tensor = torch.tensor(pad_id, dtype=torch.long, device="cuda").unsqueeze(0).unsqueeze(0) #create a tensor with shape: (bsz, total_len, 6) filled with pad_id
        tokens = pad_tensor.expand(bsz, total_len, -1).clone() #6, --> bsz, total_len, 6

        for k, t in enumerate(prompt_tokens): 
            t_tensor = torch.tensor(t, dtype=torch.long, device="cuda")  # (len_t, 6) 
            tokens[k, :len(t)] = t_tensor  #tokens[k, :len(t)] --> len_t, 6


        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_mask = torch.all(tokens != pad_tensor, dim=-1).unsqueeze(-1) #(batch, len, 1)

        """KV Cache"""
        past_key_values = None
        for cur_pos in range(min_prompt_len, total_len): #recursively generate new tokens in parallel
            print(f"{cur_pos}/{total_len} generated")
            output = self.model.forward(input_ids = tokens[:, prev_pos:cur_pos], past_key_values = past_key_values, use_cache = True, attention_mask = None) #output logtis: (batch, len, dim 
            next_decoder_token = torch.tensor(self.tokenizer.sos_out).to(tokens).expand(tokens.shape[0]*(cur_pos - prev_pos), 1) #batch*len_x, len_y = 1
            next_decoder_token_out = next_decoder_token
            hidden_state = output.logits  #first forward pass: batch, len_x, dim --> batch*len_x, dim  --> num_layer, batch*len_x, dim; 
            hidden_state = hidden_state.view(hidden_state.shape[0]*hidden_state.shape[1], hidden_state.shape[2]).unsqueeze(0).expand(self.model.decoder.num_hidden_layers, -1, -1).contiguous() #batch, len_x, dim --> num_layer, batch*len_x, dim; 

            for attribute in ["timeshift_dict_decode", "duration_dict_decode", "octave_dict_decode", "pitch_dict_decode", "instrument_dict_decode", "velocity_dict_decode"]:
                output_decoder = self.model.forward(decoded_hidden_state = hidden_state, decoded_language_tokens = next_decoder_token, attention_mask = None)
                generation_logits = output_decoder.generation_logits ##batch*len_x, len_y, decode_vocab_size
                hidden_state = output_decoder.generation_hidden_state ##num_layers, batch*len_x, dim

                sample_indices = list(getattr(self.tokenizer, attribute).keys())
                sample_indices_set = set(sample_indices)
                if temperature > 0:
                    probs = torch.softmax(generation_logits[:, -1, : ]/ temperature, dim=-1) 
                    next_decoder_token = sample_top_p(probs, top_p) 
                    
                    for i in range(next_decoder_token.size(0)):  # Ensure that all next_decoder_token values are in sample_indices, print warning if probability mass of all allowed indices is smaller than 0.8
                        start_time = time.time()
                        while next_decoder_token[i, 0].item() not in sample_indices_set:  # Check if token is valid
                            if time.time() - start_time > 15:  # If sampling takes too long, mask invalid indices
                                print(f"Warning: Resampling for token {i} exceeded 15 seconds. Masking invalid logits and Resampling...")
                                # Set logits of invalid indices to -inf
                                mask = torch.full_like(probs, float('-inf'))
                                mask[:, sample_indices] = probs[:, sample_indices]  
                                # Recompute probabilities with the mask
                                probs = torch.softmax(mask, dim=-1)
                            next_decoder_token[i, 0] = sample_top_p(probs, top_p)[i, 0]  
                else:
                    probs = torch.softmax(generation_logits[:, -1, :], dim=-1)  #batch*len_x, len_y (last), decode_vocab_size
                    sample_indices_tensor = torch.tensor(sample_indices, device=probs.device)  # Ensure it's on the same device as probs
                    probs_at_sample_indices = probs[:, sample_indices_tensor]  # Shape: [batch_size, num_sample_indices]
                    next_token_index_in_subset = probs_at_sample_indices.argmax(dim=-1, keepdim=True)  # Shape: [batch_size, 1]
                    next_decoder_token = sample_indices_tensor[next_token_index_in_subset.squeeze(-1)].unsqueeze(-1)   # Shape: [batch_size]

                # Get the cumulative probability at sample_indices, if c_p is smaller than 0.8, print warning
                probs_at_sample_indices = probs[:, sample_indices]  # Extract probabilities for the sampled indices
                cumulative_prob = probs_at_sample_indices.sum(dim=-1)  # Sum over the sampled indices to get cumulative prob
                num_samples_below_threshold = (cumulative_prob < 0.8).sum().item()  # Count the number of True values
                if num_samples_below_threshold > 0:
                    print(f"{num_samples_below_threshold} / {cumulative_prob.shape[0]} samples have a cumulative probability < 0.8 at the allowed indices")

                next_decoder_token_out = torch.cat([next_decoder_token_out, next_decoder_token], dim=-1) #batch*len_x, len_y
            
            #remove the sos_out token 
            next_decoder_token_out_reshaped = next_decoder_token_out[:, 1:].view(tokens.shape[0], -1 ,6) #batch*len_x, 6 --> batch, len_x, 6
            next_decoder_token_lang = self.tokenizer.convert_from_language_tokens(next_decoder_token_out_reshaped) #batch, lenx, 6 
            
            previous_onset = tokens[:, cur_pos-1, 0] #batch, 
            new_onset = previous_onset + next_decoder_token_lang.clone().detach()[:, -1, 0].to(previous_onset) #batch, + batch --> batch
            next_decoder_token_onset = torch.cat ([new_onset.unsqueeze(-1) ,next_decoder_token_lang.clone().detach()[:, -1, 1:]],dim=-1).to(tokens) #batch, 1  cat  batch, 5
            next_token = torch.where(
                input_mask[:, cur_pos], tokens[:, cur_pos], next_decoder_token_onset
            ) 
            tokens[:, cur_pos] = next_token

            """check if next token is eos"""
            eos_conditions_onset= next_decoder_token_lang.clone().detach()[:, -1, 0] == self.tokenizer.eos_timeshift #batch, 
            eos_conditions_dur = next_decoder_token_lang.clone().detach()[:, -1, 1] == self.tokenizer.eos_dur #batch,
            eos_conditions_oct = next_decoder_token_lang.clone().detach()[:, -1, 2] == self.tokenizer.eos_octave #batch,
            eos_conditions_pitch = next_decoder_token_lang.clone().detach()[:, -1, 3] == self.tokenizer.eos_pitch_class #batch,
            eos_conditions_instr = next_decoder_token_lang.clone().detach()[:, -1, 4] == self.tokenizer.eos_instrument #batch,
            eos_conditions_vel = next_decoder_token_lang.clone().detach()[:, -1, 5] == self.tokenizer.eos_velocity #batch,
            eos_conditions_all_attr = torch.stack([eos_conditions_onset, eos_conditions_dur, eos_conditions_oct, eos_conditions_pitch, eos_conditions_instr, eos_conditions_vel], dim = -1) #batch, 6
            eos_conditions = torch.any(eos_conditions_all_attr, dim = -1).to(input_mask) # batch, 1 

            # Update eos_reached based on the mask and EOS conditions
            eos_reached |= (~input_mask[:, cur_pos].squeeze(-1)) & eos_conditions   
            prev_pos = cur_pos
            past_key_values = output.past_key_values
            if all(eos_reached): #wait until all sequences reach eos
                print("eos reached!")
                break 
        tokens = tokens[:, 1:, :] #remove SOS token

         #TODO: in the future, return logprob
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            """if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]"""
            # cut to after eos tok if any
            for j, stop_token in enumerate([self.tokenizer.eos_timeshift, self.tokenizer.eos_dur, self.tokenizer.eos_octave, self.tokenizer.eos_pitch_class, self.tokenizer.eos_instrument, self.tokenizer.eos_velocity]):
                if j==0: #skip onset
                    continue
                try:
                    eos_idx = [row[j] for row in toks].index(stop_token)
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx] if logprobs else None
                except ValueError:
                    pass
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)
    def music_completion(
        self,
        prompt_tokens: List[List[List[int]]],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ):
  
        if max_gen_len is None:
            max_gen_len = self.config.max_len - 1 

        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo = True
        )

        prompt_tokens = [t[1:] for t in prompt_tokens] #remove SOS token

        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t),
                    },
                    "tokens": [self.tokenizer.decode([x]) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
        ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.compound_to_midi(t), 
                    "prompt": self.tokenizer.compound_to_midi(p),
                    "prompt_tokens": p,
                    "tokens": t,
                },
            }
            for t, p in zip(generation_tokens, prompt_tokens) 
        ]

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token