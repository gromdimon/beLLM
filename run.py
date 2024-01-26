"""
This file is used to run the model.
"""

import argparse
import logging
import time

import torch

from model import BigramLanguageModel
from train import config

# Setting up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate text using the Bigram Language Model.")
    parser.add_argument("--prompt", type=str, default="", help="Initial text to start generation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    start_time = time.time()
    logging.info("Loading model...")
    logging.info("Config: %s", config)

    # load the dataset
    with open("data/data.txt", "r") as f:
        text = f.read()
    chars = sorted(list(set(text)))

    # Encoding
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]  # encoder
    decode = lambda l: "".join([itos[i] for i in l])  # decoder

    # Example model load
    state_dict = torch.load("models/model_9999.pt", map_location=torch.device(config.device))
    model = BigramLanguageModel(config=config)
    model.load_state_dict(state_dict)
    logging.info("Model loaded in {:.2f}s".format(time.time() - start_time))

    # generate
    start_generation_time = time.time()
    initial_context = args.prompt if args.prompt else " "
    if len(initial_context) > 256:
        raise ValueError("Initial context must be less than 256 characters.")
    context = torch.tensor([encode(initial_context)], dtype=torch.long)
    generated_text = decode(model.generate(context, max_new_tokens=2000)[0].tolist())
    # save the generated text in generations/ folder
    with open(f"generations/{generated_text[:6]}.txt", "w") as f:
        f.write(generated_text)
    logging.info(generated_text)
    logging.info("Generation completed in {:.2f}s".format(time.time() - start_generation_time))
