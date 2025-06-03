from miditok import MusicTokenizer
from transformers import AutoModelForCausalLM, MistralConfig


def build_mistral_model(
    cfg: dict, tokenizer: MusicTokenizer, max_seq_len: int
) -> AutoModelForCausalLM:
    """
    Builds a Mistral causal language model based on configuration and tokenizer.
    """
    model_config = MistralConfig(
        vocab_size=len(tokenizer),
        num_hidden_layers=cfg["num_hidden_layers"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_head=cfg[
            "num_attention_heads"
        ],  # Assuming num_key_value_head is same as num_attention_heads
        sliding_window=max_seq_len,  # Use max_seq_len as sliding window size
        max_position_embeddings=cfg["max_position_embeddings"],
        pad_token_id=tokenizer["PAD_None"],
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
        attention_dropout=cfg["attention_dropout"],
    )
    model = AutoModelForCausalLM.from_config(model_config)
    return model
