"""
This file contains the model definition for the GPT model.
The model is based on the GPT-2 model by OpenAI.
The model is defined using PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, config, head_size):
        super().__init__()
        assert config.n_embd is not None
        assert config.block_size is not None
        assert config.dropout is not None

        self.config = config
        self.key = nn.Linear(self.config.n_embd, head_size, bias=False)
        self.query = nn.Linear(self.config.n_embd, head_size, bias=False)
        self.value = nn.Linear(self.config.n_embd, head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(self.config.block_size, self.config.block_size)
            ),
        )

        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, config, num_heads, head_size):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList(
            [Head(config, head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(self.config.n_embd, self.config.n_embd)
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, config, n_embd):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(self.config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, config, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.config = config
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(self.config, n_head, head_size)
        self.ffwd = FeedFoward(self.config, n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(
            self.config.vocab_size, self.config.n_embd
        )
        self.position_embedding_table = nn.Embedding(
            self.config.block_size, self.config.n_embd
        )
        self.blocks = nn.Sequential(
            *[
                Block(
                    self.config, self.config.n_embd, n_head=self.config.n_head
                )
                for _ in range(self.config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(self.config.n_embd)  # final layer norm
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.config.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config.block_size :]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
