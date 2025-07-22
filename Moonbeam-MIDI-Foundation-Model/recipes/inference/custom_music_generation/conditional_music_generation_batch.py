# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional

import fire
import pandas as pd
import numpy as np
import os
import re
# from llama import Dialog, Llama
from generation import MusicLlama
import random
import ast
import json
import miditoolkit
from mido import MidiFile, MidiTrack, Message, bpm2tempo, MetaMessage
from music21 import chord
from music21 import harmony
# Function to convert chord symbols to MIDI notes using music21
def chord_to_midi(chord_symbol):
    # Create a ChordSymbol object
    chord_obj = harmony.ChordSymbol(chord_symbol)
    # Get the pitches of the chord
    pitches = chord_obj.pitches
    # Return the pitch names
    out = [p.midi  for p in pitches]
    return out

def create_midi_from_chords(chord_list, is_incomplete_measure, ticks_per_bar, bpm=220):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Convert BPM to MIDI tempo (microseconds per quarter note)
    tempo = bpm2tempo(bpm)

    # Start time accounting for incomplete measure
    start_time = is_incomplete_measure * ticks_per_bar

    # Set the tempo using a MetaMessage
    track.append(MetaMessage('set_tempo', tempo=tempo, time=0))

    # Set instrument (optional)
    track.append(Message('program_change', program=0, time=start_time))

    for chord_symbol in chord_list:
        notes = chord_to_midi(chord_symbol)  # Convert chord to MIDI note numbers
        for note in notes:
            track.append(Message('note_on', note=note, velocity=64, time=0))

        ts = mid.ticks_per_beat // 2  # Set duration to an eighth note
        for note in notes:
            track.append(Message('note_off', note=note, velocity=64, time=ts))
            ts = 0  # Ensure only first note-off has delay

    return mid

def main(
    ckpt_dir: str,
    csv_file: str,
    tokenizer_path: str,
    model_config_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    if_add_chords_in_transformer: bool = True,
    if_add_metadata_in_transformer: bool = False,
    max_gen_len: Optional[int] = None,
    finetuned_PEFT_weight_path: Optional[str] = None,
    additional_token_dict_path: Optional[str] = None,
    chord_dict_path: Optional[str] = None
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    # Set the random seed for CPU and GPU
    seed = 42
    import torch
    torch.manual_seed(seed)
    random.seed(seed)  # You can choose any seed value, 42 is commonly used
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.    

    with open(additional_token_dict_path, "r") as f:
        additional_token_dict = json.load(f)

    save_folder = os.path.join(finetuned_PEFT_weight_path, os.path.basename(ckpt_dir), f"temperature_{temperature}_top_p_{top_p}")
    os.makedirs(save_folder, exist_ok=True)

    additional_token_dict_inv = {v: k for k, v in additional_token_dict.items()}
    generator = MusicLlama.build_commu_con_gen(
        ckpt_dir=ckpt_dir,
        model_config_path = model_config_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        finetuned_PEFT_weight_path = finetuned_PEFT_weight_path,
        additional_token_dict = additional_token_dict
    ) 
    if_add_chords_in_decoder = generator.model.if_add_chord_in_decoder
    if_add_metadata_in_decoder = generator.model.if_add_metadata_in_decoder
    print(f"if_add_chords_in_transformer: {if_add_chords_in_transformer}, if_add_metadata_in_transformer: {if_add_metadata_in_transformer}")
    print(f"if_add_chords_in_decoder: {if_add_chords_in_decoder}, if_add_metadata_in_decoder: {if_add_metadata_in_decoder}")
    df = pd.read_csv(csv_file)
    import time
    # Start time
    start_time = time.time()
    finished_idx = 0
    prompts = []
    metadata_condition_decoder = []
    chord_condition_decoder = []
    bpm_condition = []
    time_signature_condition = []
    num_measures_condition = []
    midi_save_paths = []
    for i, (_, row) in enumerate(df.iterrows()):
        if row["split_data"] == "train":
            finished_idx +=1
            continue

        if i == min(finished_idx + max_batch_size, len(df)-1):
            #generate midi based on the given conditions
            condition_token_lengths = [len(x) for x in prompts]
            if if_add_chords_in_transformer:
                chord_token_indices = [[x.index(generator.tokenizer.soc_token_compound), x.index(generator.tokenizer.eoc_token_compound)] for x in prompts]
            else:
                chord_token_indices = None

            results = generator.music_completion(
                prompts,
                bpm_condition = bpm_condition,
                time_signature_condition = time_signature_condition,
                num_measures_condition = num_measures_condition,
                metadata_condition = metadata_condition_decoder,
                chord_condition = chord_condition_decoder, 
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                condition_token_lengths = condition_token_lengths, #remove sos token and emotion token
                chord_token_indices = chord_token_indices,
                chord_dict_path = chord_dict_path, 
                if_return_chords = False
            )
            #save midi
            for midi_save_path, result in zip(midi_save_paths, results):
                result['generation']['content'][0].save(midi_save_path)
                print(f"Generated MIDI saved at: {i}, {midi_save_path}")

            #empty the conditions buffer 
            finished_idx = i
            prompts = []
            metadata_condition_decoder = []
            chord_condition_decoder = []
            bpm_condition = []
            time_signature_condition = []
            num_measures_condition = []
            midi_save_paths = []
        
        
        #first batch the input: prompts, bpm_condition, time_signature_condition, num_measures_condition, metadata_condition_decoder, chord_condition_decoder, 
        #set save path
        chord_save_path  = os.path.join(save_folder, f"{row['id']}_chords.mid")
        midi_save_paths.append(os.path.join(save_folder, f"{row['id']}.mid"))
        metadata_save_path = os.path.join(save_folder, f"{row['id']}_metadata.json")
        
        #batch necessary conditions
        chord_condition_str = ast.literal_eval(row["chord_progressions"])[0]
        metadata_id = [
            additional_token_dict["audio_key_" + row["audio_key"]],
            additional_token_dict["pitch_range_" + row["pitch_range"]],
            additional_token_dict["num_measures_" + str(row["num_measures"])],
            additional_token_dict["bpm_" + str(row["bpm"])],
            additional_token_dict["genre_" + row["genre"]],
            additional_token_dict["track_role_" + row["track_role"]],
            additional_token_dict["inst_" + row["inst"].split("-")[0]],
            additional_token_dict["sample_rhythm_" + row["sample_rhythm"]],
            additional_token_dict["time_signature_" + row["time_signature"]],
            additional_token_dict["min_velocity_" + str(row["min_velocity"])],
            additional_token_dict["max_velocity_" + str(row["max_velocity"])]
        ]      
        bpm_condition.append(row["bpm"])
        time_signature_condition.append(row["time_signature"])
        num_measures_condition.append(row["num_measures"])
        if if_add_metadata_in_decoder:
            metadata_condition_decoder.append(metadata_id)  
        else:
            metadata_condition_decoder = None
        if if_add_chords_in_decoder:
            chord_condition_decoder.append(chord_condition_str)
        else:
            chord_condition_decoder = None
        
        #create and save chord midi and metadata
        if row["num_measures"]%4==0:
            is_incomplete_measure = False
        else:
            is_incomplete_measure = True
        numerator = int(row["time_signature"].split("/")[0])
        denominator = int(row["time_signature"].split("/")[1])
        ticks_per_beat = 480
        normalized_bpm = 120
        beats_per_bar = numerator / denominator * 4
        ticks_per_bar = int(ticks_per_beat * beats_per_bar)

        chord_midi = create_midi_from_chords(chord_condition_str, is_incomplete_measure, ticks_per_bar, normalized_bpm)
        chord_midi.save(chord_save_path)

        #construct metadat diction json
        metadata_dict = {
            "key": row["audio_key"],
            "pitch_range": row["pitch_range"],
            "num_measures": row["num_measures"],
            "bpm": row["bpm"],
            "genre": row["genre"],
            "track_role": row["track_role"],
            "inst": row["inst"],
            "sample_rhythm": row["sample_rhythm"],
            "time_signature": row["time_signature"],
            "min_velocity": row["min_velocity"],
            "max_velocity": row["max_velocity"],
            "chord_progression": row["chord_progressions"]
        }
        with open(metadata_save_path, "w") as f:
            json.dump(metadata_dict, f)

        raw_tokens_chord = generator.tokenizer.midi_to_compound(chord_save_path, calibate_to_default_tempo = True)
        prompts.append(generator.tokenizer.encode_series_con_gen_commu(None, raw_tokens_chord, metadata_tokens = [[x for _ in range(6)] for x in metadata_id], if_only_keep_condition_tokens = True, if_add_chords_in_transformer = if_add_chords_in_transformer, if_add_metadata_in_transformer = if_add_metadata_in_transformer))

    # After the for-loop ends, check for remaining unprocessed rows
    if prompts:
        # Generate MIDI for the remaining batch
        condition_token_lengths = [len(x) for x in prompts]
        if if_add_chords_in_transformer:
            chord_token_indices = [[x.index(generator.tokenizer.soc_token_compound), x.index(generator.tokenizer.eoc_token_compound)] for x in prompts]
        else:
            chord_token_indices = None

        results = generator.music_completion(
            prompts,
            bpm_condition=bpm_condition,
            time_signature_condition=time_signature_condition,
            num_measures_condition=num_measures_condition,
            metadata_condition=metadata_condition_decoder,
            chord_condition=chord_condition_decoder, 
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            condition_token_lengths=condition_token_lengths,
            chord_token_indices=chord_token_indices,
            chord_dict_path=chord_dict_path, 
            if_return_chords=False
        )

        # Save the remaining MIDI files
        for midi_save_path, result in zip(midi_save_paths, results):
            result['generation']['content'][0].save(midi_save_path)
            print(f"Generated MIDI saved at: {midi_save_path}")

    # End time
    end_time = time.time()

    # Calculate time taken in seconds
    time_taken_seconds = end_time - start_time

    # Convert to minutes
    time_taken_minutes = time_taken_seconds / 60

    print(f"The loop took {time_taken_minutes} minutes to execute.")
if __name__ == "__main__":
    fire.Fire(main)
