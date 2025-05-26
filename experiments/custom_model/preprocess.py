import glob
import os
import pickle

from utils.midi import build_vocabulary, encode_tokens, tokenize

midi_root = "data/maestro"
token_dir = "data/tokens"
vocab_dir = "data/vocab"
id_dir = "data/ids"

for midi_path in glob.glob(os.path.join(midi_root, "**", "*.midi"), recursive=True):
    filename = os.path.splitext(os.path.basename(midi_path))[0] + ".json"
    out_path = os.path.join(token_dir, filename)
    if not os.path.exists(out_path):
        try:
            tokenize(
                midi_path, out_path
            )  # Can also be tokenize_remi or tokenize_midilike
            print(f"Tokenized: {filename}")
        except Exception as e:
            print(f"Failed to tokenize {filename}: {e}")

print("Tokenization complete")

build_vocabulary(token_dir, vocab_dir)
print("Vocabulary complete")

token2id_path = os.path.join(vocab_dir, "token2id.pkl")
with open(token2id_path, "rb") as f:
    token2id = pickle.load(f)

for json_path in glob.glob(os.path.join(token_dir, "*.json")):
    filename = os.path.splitext(os.path.basename(json_path))[0] + ".npy"
    npy_path = os.path.join(id_dir, filename)
    if not os.path.exists(npy_path):
        try:
            encode_tokens(json_path, token2id, npy_path)
            print(f"Encoded: {filename}")
        except Exception as e:
            print(f"Failed to encode {filename}: {e}")

print("Encoding complete")
