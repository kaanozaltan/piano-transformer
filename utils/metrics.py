import os
import numpy as np
from collections import Counter
import torch


def load_token_counts_from_dir(directory):
    counter = Counter()
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            ids = np.load(os.path.join(directory, filename))
            counter.update(ids.tolist())
    return counter


def normalize_counter(counter, vocab_size):
    dist = np.zeros(vocab_size, dtype=np.float64)
    total = sum(counter.values())
    for idx, count in counter.items():
        dist[idx] = count / total
    return dist


def kl_divergence(p, q):
    p = torch.tensor(p, dtype=torch.float32)
    q = torch.tensor(q, dtype=torch.float32)
    return torch.nn.functional.kl_div(torch.log(q + 1e-12), p, reduction='sum').item()


def js_divergence(p, q):
    p = torch.tensor(p, dtype=torch.float32)
    q = torch.tensor(q, dtype=torch.float32)
    m = 0.5 * (p + q)
    kl_pm = torch.nn.functional.kl_div(torch.log(m + 1e-12), p, reduction='sum')
    kl_qm = torch.nn.functional.kl_div(torch.log(m + 1e-12), q, reduction='sum')
    return 0.5 * (kl_pm + kl_qm).item()


def compare_distributions(test_dir, gen_dir, vocab_size):
    real_counts = load_token_counts_from_dir(test_dir)
    gen_counts = load_token_counts_from_dir(gen_dir)

    p_real = normalize_counter(real_counts, vocab_size)
    p_gen = normalize_counter(gen_counts, vocab_size)

    kl = kl_divergence(p_real, p_gen)
    js = js_divergence(p_real, p_gen)

    return {"kl_divergence": kl, "js_divergence": js}