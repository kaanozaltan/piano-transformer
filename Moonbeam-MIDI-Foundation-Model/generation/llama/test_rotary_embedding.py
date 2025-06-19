def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    print("dim", dim) #only apply to q and k; dim = n_dim / n_heads
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    print("freqs", freqs)
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    print("t", t.shape)
    freqs = torch.outer(t, freqs)
    print("freqs", freqs.shape)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64 # function is used to convert Cartesian coordinates to polar coordinates
    print("freqs_cis", freqs_cis.shape)
    assert 1==2
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))#last dimension split into 2 groups
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


if __name__=="__main__":
    dim = 256
    length = 1024
    rope_theta = 50000
    