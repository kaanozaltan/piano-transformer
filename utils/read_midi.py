import pretty_midi

midi = pretty_midi.PrettyMIDI(
    'data/maestro/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi'
)

for i, instrument in enumerate(midi.instruments):
    print(f"Instrument {i}: {instrument.name}")

    for note in instrument.notes:
        print(f"Note {note.pitch}, start: {note.start:.2f}s, end: {note.end:.2f}s, velocity: {note.velocity}")