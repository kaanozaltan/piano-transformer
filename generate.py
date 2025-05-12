import torch
import pickle
from models.transformer import PT1


def generate(model, token2id, id2token, max_len=512, temperature=1.0, device="cpu", prompt=None):
    model.eval()
    input_ids = [token2id[tok] for tok in prompt] if prompt else [token2id["TIME_SHIFT_10"]]
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)

    for _ in range(max_len - input_ids.shape[1]):
        with torch.no_grad():
            logits = model(input_ids)  # (1, seq_len, vocab_size)
            next_token_logits = logits[0, -1] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.argmax(probs).item()

            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)

    generated_ids = input_ids[0].tolist()
    return [id2token[i] for i in generated_ids]