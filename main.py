import torch
from models.transformer import PT1


if __name__ == "__main__":
    vocab_size = 512
    seq_len = 512
    batch_size = 4

    # Dummy input tensor: (batch_size, seq_len)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Initialize model
    model = PT1(vocab_size=vocab_size, dim=256, n_heads=4, n_layers=4)

    # Forward pass
    output = model(input_ids)

    print("Input shape: ", input_ids.shape)  # Expected: (4, 512)
    print("Output shape:", output.shape)  # Expected: (4, 512, 512