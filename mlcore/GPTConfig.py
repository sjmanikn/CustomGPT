from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 256  # max sequence length
    vocab_size: int = 65  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 6  # number of layers
    n_head: int = 6  # number of heads
    n_embd: int = 384  # embedding dimension