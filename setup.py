import tiktoken
import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer
from gpt2 import GPT, from_pretrained
from mlcore.GPTConfig import GPTConfig

model_path = "log/model_00099"

# Load the model state dictionary (device agnostic)
#model = from_pretrained("gpt2")

# n_layer, n_head and n_embd are determined from model_type
config_args = {
    'gpt2': dict(n_layer=6, n_head=6, n_embd=384),  # 124M params
    'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
    'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
    'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
}["gpt2"]
config_args['vocab_size'] = 50304  # always 50257 for GPT model checkpoints
config_args['block_size'] = 256  # always 1024 for GPT model checkpoints

config = GPTConfig(**config_args)
model = GPT(config)
model.load_state_dict(torch.load(model_path))

# Load the state dictionary into the model


num_return_sequences = 1
max_length = 100
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("How does the Android App collect data?")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
sample_rng = torch.Generator(device="cpu")
sample_rng.manual_seed(42)
xgen = tokens.to("cpu")
while xgen.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits, loss = model(xgen) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        xgen = torch.cat((xgen, xcol), dim=1)
# print the generated text
for i in range(num_return_sequences):
    tokens = xgen[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"rank {0} sample {i}: {decoded}")
