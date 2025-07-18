import sys
import os
from miditok import REMI, TokenizerConfig
sys.path.append(os.path.abspath("src"))

from src.llama_recipes.datasets.music_tokenizer import MusicTokenizer

midi_file= "data/maestro/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi"
compound = MusicTokenizer.midi_to_compound(midi_file)

# tokenizer_config = TokenizerConfig(
#     pitch_range=(21, 109),
#     beat_res={(0, 1): 12, (1, 4): 8, (4, 12): 4},
#     special_tokens=["PAD", "BOS", "EOS"],
#     use_chords=True,
#     use_rests=True,
#     use_tempos=True,
#     use_time_signatures=True,
#     use_sustain_pedals=True,
#     num_velocities=16,
#     num_tempos=32,
#     tempo_range=(50, 200),
# )
# tokenizer = REMI(tokenizer_config)



