from piano_transformer.utils import metrics

kld_val = metrics.compare(
    "data/debug_1", "data/debug_2", metric=metrics.kld, feature="pitch"
)
oa_val = metrics.compare(
    "data/debug_1", "data/debug_2", metric=metrics.oa, feature="pitch"
)
print("KLD (pitch):", kld_val)
print("OA (pitch):", oa_val)
