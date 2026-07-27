# NanoGPT

A tiny transformer language model that trains on `input.txt` using a minimal PyTorch implementation.

## What is included

- `nanogpt.py` — main training and generation script
- `attention.py` — transformer block, attention head, feed-forward, and residual connections
- `hyperparams.py` — hyperparameters for batch size, block size, model size, optimizer, and device
- `input.txt` — training corpus
- `requirements.txt` — Python dependencies

## Features

- character-level language modeling
- token and position embeddings
- multi-head self-attention
- layer normalization
- mini-batch training with `AdamW`

## Usage

Install dependencies and run:

```bash
python -m pip install -r requirements.txt
python nanogpt.py
```

The script trains a small transformer model and prints generated text after training.