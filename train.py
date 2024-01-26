"""
This file is used to train the model.
"""

import logging
import time

import torch
from pydantic import BaseModel

from model import BigramLanguageModel

# Setting up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -----------
# # Hyperparameters
BATCH_SIZE = 32  # how many independent sequences will we process in parallel?
BLOCK_SIZE = 256  # what is the maximum context length for predictions?
MAX_ITERATIONS = 10000
EVALUATION_INTERVAL = 500
LEARNING_RATE = 4e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVALUATION_ITERATIONS = 200
NUMBER_OF_EMBEDDINGS = 512
NUMBER_OF_HEADS = 8
NUMBER_OF_LAYERS = 8
DROPOUT = 0.0
# -----------


#: Configuration for the model
class Config(BaseModel):
    vocab_size: int  # size of the vocabulary
    batch_size: int  # how many independent sequences will we process in parallel?
    block_size: int  # what is the maximum context length for predictions?
    max_iters: int
    eval_interval: int
    learning_rate: float
    device: str
    eval_iters: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float


config = Config(
    vocab_size=123,  # size of the vocabulary
    batch_size=BATCH_SIZE,
    block_size=BLOCK_SIZE,
    max_iters=MAX_ITERATIONS,
    eval_interval=EVALUATION_INTERVAL,
    learning_rate=LEARNING_RATE,
    device=DEVICE,
    eval_iters=EVALUATION_ITERATIONS,
    n_embd=NUMBER_OF_EMBEDDINGS,
    n_head=NUMBER_OF_HEADS,
    n_layer=NUMBER_OF_LAYERS,
    dropout=DROPOUT,
)


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i : i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + config.block_size + 1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":
    start_time = time.time()

    # load the dataset
    with open("data/data.txt", "r") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # Encoding
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]  # encoder: take a string and return numerical values
    decode = lambda l: "".join(
        [itos[i] for i in l]
    )  # decoder: take a list of integers and return a corresponding string

    # Encode entire text and store as torch.Tensor
    data = torch.tensor(encode(text), dtype=torch.long)
    # Split data for training and validation
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # create the model
    model = BigramLanguageModel(config=config).to(config.device)
    logging.info(f"{sum(p.numel() for p in model.parameters()) / 1e6} M parameters")

    # create the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    start_train_time = time.time()
    logging.info("Data loaded in {:.2f}s".format(start_train_time - start_time))
    logging.info("Config: %s", config)
    logging.info("Training started...")

    for iter in range(config.max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss()
            logging.info(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        # save the model parameters
        if iter % 2000 == 0 or iter == config.max_iters - 1:
            time_elapsed = time.time() - start_train_time
            logging.info("Iteration {} completed in {:.2f}s".format(iter, time_elapsed))
            torch.save(model.state_dict(), f"models/model_{iter}.pt")

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final_time = time.time() - start_train_time
    logging.info("Training completed in {:.2f}s".format(final_time))
