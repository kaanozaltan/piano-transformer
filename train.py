import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import pickle
from datasets.maestro import MAESTRO
from models.transformer import PT1


def train():
    # Config
    data_dir = "data/ids"
    vocab_path = "data/vocab/token2id.pkl"
    batch_size = 8
    seq_len = 512
    dim = 256
    n_heads = 4
    n_layers = 4
    lr = 1e-4
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab size
    with open(vocab_path, "rb") as f:
        token2id = pickle.load(f)
    vocab_size = len(token2id)

    dataset = MAESTRO(data_dir, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PT1(vocab_size=vocab_size, dim=dim, n_heads=n_heads, n_layers=n_layers).to(device)
    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)  # (batch, seq_len, vocab_size)
            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train()