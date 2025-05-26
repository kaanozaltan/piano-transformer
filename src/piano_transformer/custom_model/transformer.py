import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # shape: (1, max_len, dim)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# Piano Transformer 1
class PT1(nn.Module):
    def __init__(
        self, vocab_size, dim=256, n_heads=4, n_layers=4, dropout=0.1, max_len=2048
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_enc = PositionalEncoding(dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)  # (batch, seq_len, dim)
        x = self.pos_enc(x)  # Add positional encoding
        x = self.transformer(x)  # (batch, seq_len, dim)
        return self.fc_out(x)  # (batch, seq_len, vocab_size)
