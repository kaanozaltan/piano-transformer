import pretty_midi
from miditok import REMI
from symusic import Score


def read(filepath, n=10):
    midi = pretty_midi.PrettyMIDI(filepath)
    for i, instrument in enumerate(midi.instruments):
        print(f"Instrument {i}: {instrument.name}")
        for note in instrument.notes[:n]:
            print(f"Note {note.pitch}, start: {note.start:.2f}s, end: {note.end:.2f}s, velocity: {note.velocity}")


def tokenize(filepath):
    tokenizer = REMI()
    score = Score(filepath)
    tokens = tokenizer(score)
    return tokens[0].tokens