import json
import os

import numpy as np
from scipy.special import rel_entr
from scipy.stats import gaussian_kde


# May have to change this for different tokenizers
def extract_features(directory, features=("pitch", "duration", "velocity")):
    pitch_vals = []
    duration_vals = []
    velocity_vals = []

    for file in os.listdir(directory):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(directory, file), "r") as f:
            tokens = json.load(f)

        for tok in tokens:
            if "pitch" in features and ("NOTE_ON_" in tok or "Note-On_" in tok):
                try:
                    pitch_vals.append(int(tok.split("_")[-1]))
                except:
                    continue

            if "duration" in features and ("TIME_SHIFT_" in tok or "Duration_" in tok):
                try:
                    dur = int(tok.split("_")[-1].replace(".", ""))
                    duration_vals.append(dur)
                except:
                    continue

            if "velocity" in features and (
                "SET_VELOCITY_" in tok or "Velocity_" in tok
            ):
                try:
                    velocity_vals.append(int(tok.split("_")[-1]))
                except:
                    continue

    return {"pitch": pitch_vals, "duration": duration_vals, "velocity": velocity_vals}


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


def compare(ref_dir, gen_dir, metric, feature="pitch"):
    ref_feats = extract_features(ref_dir, features=(feature,))
    gen_feats = extract_features(gen_dir, features=(feature,))

    ref_vals = ref_feats.get(feature, [])
    gen_vals = gen_feats.get(feature, [])

    # Not enough data
    if len(ref_vals) < 2 or len(gen_vals) < 2:
        return None

    ref_pdf = compute_pdf(ref_vals)
    gen_pdf = compute_pdf(gen_vals)

    if ref_pdf is None or gen_pdf is None or len(ref_pdf) != len(gen_pdf):
        return None

    return metric(ref_pdf, gen_pdf)
