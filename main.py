from utils import metrics


kld = metrics.dirs_kld("data/debug_1", "data/debug_2", feature="pitch")
print("KLD (pitch):", kld)