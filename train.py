import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import os
import pickle
import wandb
from datasets.maestro import MAESTRO
from models.transformer import PT1


def train():
    config = {
        "model": "PT1",
        "data_dir": "data/ids",
        "vocab_path": "data/vocab/token2id.pkl",
        "batch_size": 4,
        "seq_len": 256,
        "dim": 128,
        "n_heads": 2,
        "n_layers": 2,
        "lr": 1e-4,
        "n_epochs": 1
    }

    wandb.init(
        project="music-transformer",
        name=(
            f"{config['model'].lower()}_"
            f"d{config['dim']}_"
            f"l{config['n_layers']}_"
            f"h{config['n_heads']}_"
            f"s{config['seq_len']}"
        ),
        config=config
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config["vocab_path"], "rb") as f:
        token2id = pickle.load(f)
    vocab_size = len(token2id)

    dataset = MAESTRO(config["data_dir"], seq_len=config["seq_len"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = PT1(
        vocab_size=vocab_size,
        dim=config["dim"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"]
    ).to(device)

    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config["lr"])

    for epoch in range(config["n_epochs"]):
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
        print(f"Epoch {epoch + 1}/{config['n_epochs']}, Loss: {avg_loss:.4f}")

        wandb.log({"epoch": epoch + 1, "loss": avg_loss})

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/{wandb.run.name}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")


if __name__ == "__main__":
    train()