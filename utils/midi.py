import pretty_midi
import numpy as np
import json
import os
import pickle
from miditok import REMI, MIDILike
from miditoolkit import MidiFile

VELOCITY_BINS = np.linspace(1, 127, 32, dtype=int)


def quantize_velocity(velocity):
    idx = np.abs(VELOCITY_BINS - velocity).argmin()
    return f"SET_VELOCITY_{VELOCITY_BINS[idx]}"


def quantize_time_shift(ms):
    steps = list(range(10, 1001, 10))  # 10ms to 1000ms
    idx = np.abs(np.array(steps) - ms).argmin()
    return f"TIME_SHIFT_{steps[idx]}"


def tokenize(input_path, output_path, save=True):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    midi = pretty_midi.PrettyMIDI(str(input_path))

    events = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        notes = sorted(instrument.notes, key=lambda n: n.start)

        prev_time = 0.0
        current_velocity = None

        for note in notes:
            delta_time = note.start - prev_time
            ms = int(delta_time * 1000)
            while ms > 0:
                shift = min(ms, 1000)
                events.append(quantize_time_shift(shift))
                ms -= shift

            if note.velocity != current_velocity:
                events.append(quantize_velocity(note.velocity))
                current_velocity = note.velocity

            events.append(f"NOTE_ON_{note.pitch}")
            duration_ms = int((note.end - note.start) * 1000)
            events.append(quantize_time_shift(duration_ms))
            events.append(f"NOTE_OFF_{note.pitch}")

            prev_time = note.start

    if save:
        with open(output_path, "w") as f:
            json.dump(events, f)
        
    return events


def tokenize_remi(input_path, output_path, save=True):
    tokenizer = REMI()
    midi = MidiFile(input_path)
    tokens = tokenizer(midi)
    if save:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(tokens[0].tokens, f)
    else:
        return tokens[0].tokens


def tokenize_midilike(input_path, output_path, save=True):
    tokenizer = MIDILike()
    midi = MidiFile(input_path)
    tokens = tokenizer(midi)
    if save:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(tokens[0].tokens, f)
    else:
        return tokens[0].tokens


def build_vocabulary(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    token_set = set()

    for filename in os.listdir(input_path):
        if filename.endswith(".json"):
            with open(os.path.join(input_path, filename)) as f:
                tokens = json.load(f)
                token_set.update(tokens)

    sorted_tokens = sorted(token_set)
    token2id = {tok: i for i, tok in enumerate(sorted_tokens)}
    id2token = {i: tok for tok, i in token2id.items()}

    with open(os.path.join(output_path, "token2id.pkl"), "wb") as f:
        pickle.dump(token2id, f)
    with open(os.path.join(output_path, "id2token.pkl"), "wb") as f:
        pickle.dump(id2token, f)


def encode_tokens(token_path, token2id, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(token_path) as f:
        tokens = json.load(f)
    ids = [token2id[tok] for tok in tokens if tok in token2id]

    np.save(output_path, np.array(ids, dtype=np.int32))