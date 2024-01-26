"""
This file is used to run the model.
"""

import os
import time

import torch
import torch.nn as nn
from model import BigramLanguageModel
from train import config

if __name__ == "__main__":
    start_time = time.time()
    print("Loading model...")
    print("Config: {}".format(config))
    # load the dataset
    with open("data/data.txt", "r") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    # Encoding
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [
        stoi[c] for c in s
    ]  # encoder: take a string and return numerical values
    decode = lambda l: "".join(
        [itos[i] for i in l]
    )  # decoder: take a list of integers and return a corresponding string

    # Example model load
    state_dict = torch.load("models/model_9999.pt", map_location=torch.device('cpu'))
    model = BigramLanguageModel(config=config)
    model.load_state_dict(state_dict)
    print("Model loaded in {:.2f}s".format(time.time() - start_time))

    # generate
    start_generation_time = time.time()
    context = torch.zeros((1, 1), dtype=torch.long)
    print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
    print(
        "Generation completed in {:.2f}s".format(
            time.time() - start_generation_time
        )
    )
