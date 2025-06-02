import os

import numpy as np
import traceback
from scipy.special import rel_entr
from scipy.stats import gaussian_kde
from miditok import REMI
from frechet_music_distance import FrechetMusicDistance
from symusic import Score


TOKENIZER_PATH = "/hpcwork/lect0148/experiments/mistral-162M_remi_maestro_v1/tokenizer.json"


def extract_features(
    directory,
    tokenizer_path=TOKENIZER_PATH,
    features=("pitch", "duration", "velocity")
):
    tokenizer = REMI(params=tokenizer_path)
    pitch_vals, duration_vals, velocity_vals = [], [], []

    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith(".midi"):
                continue

            midi_path = os.path.join(root, file)
            try:
                midi = Score(midi_path)
                token_ids = tokenizer(midi)

                # Take only the piano track
                if isinstance(token_ids, list):
                    token_ids = token_ids[0]

                # Convert token IDs to string tokens
                vocab_inv = {v: k for k, v in tokenizer.vocab.items()}
                tokens = [vocab_inv[i] for i in token_ids if i in vocab_inv]

            except Exception as e:
                print(f"Error tokenizing {file}: {type(e).__name__}: {e}")
                traceback.print_exc()
                continue

            for tok in tokens:
                if "pitch" in features and "Pitch_" in tok:
                    try:
                        pitch_vals.append(int(tok.split("_")[-1]))
                    except:
                        continue
                if "duration" in features and "Duration_" in tok:
                    try:
                        duration_vals.append(int(tok.split("_")[-1]))
                    except:
                        continue
                if "velocity" in features and "Velocity_" in tok:
                    try:
                        velocity_vals.append(int(tok.split("_")[-1]))
                    except:
                        continue

    return {
        "pitch": pitch_vals,
        "duration": duration_vals,
        "velocity": velocity_vals
    }

def compute_pdf(values, num_points=1000):
    values = np.array(values).astype(np.float64)
    if len(np.unique(values)) < 2:
        return None  # Cannot build KDE from constant data

    kde = gaussian_kde(values)
    x = np.linspace(np.min(values), np.max(values), num_points)
    pdf = kde.evaluate(x)
    pdf /= np.sum(pdf)  # Normalize
    return pdf


# Kullback-Leibler Divergence
def kld(p, q, epsilon=1e-10):
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    p /= np.sum(p)
    q /= np.sum(q)
    return np.sum(rel_entr(p, q))


# Overlapping Area
def oa(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    p /= np.sum(p)
    q /= np.sum(q)
    return np.sum(np.minimum(p, q))


# Frechet Music Distance
def fmd(ref_dir, gen_dir):
    fmd_metric = FrechetMusicDistance(
        feature_extractor='clamp2',
        gaussian_estimator='mle', 
        verbose=True
    )

    return fmd_metric.score(
        reference_path=ref_dir,
        test_path=gen_dir
    )


def compare(ref_dir, gen_dir, metric, tokenizer_path=TOKENIZER_PATH, feature="pitch"):
    if metric == fmd:
        print("Using FMD (input feature ignored)")
        return fmd(ref_dir, gen_dir)

    ref_feats = extract_features(ref_dir, tokenizer_path, features=(feature,))
    gen_feats = extract_features(gen_dir, tokenizer_path, features=(feature,))

    ref_vals = ref_feats.get(feature, [])
    gen_vals = gen_feats.get(feature, [])

    if len(ref_vals) < 2 or len(gen_vals) < 2:
        return None

    ref_pdf = compute_pdf(ref_vals)
    gen_pdf = compute_pdf(gen_vals)

    if ref_pdf is None or gen_pdf is None or len(ref_pdf) != len(gen_pdf):
        return None

    return metric(ref_pdf, gen_pdf)
