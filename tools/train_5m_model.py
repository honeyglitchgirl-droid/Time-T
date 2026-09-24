"""Train Time-T 5M Language Model to produce coherent language.

Learns character/byte level language modeling over structured linguistic patterns.
Saves model checkpoint in Time-T native format (.ttck) and Safetensors.
"""
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import timet.nn as nn
import timet.optim as optim
from timet.tensor import Tensor, tensor
import timet.checkpoint as checkpoint
from models.transformer_lm_5m import TimeTLanguageModel5M


def load_corpus(path: str = "data/corpus.txt") -> bytes:
    with open(path, "rb") as f:
        return f.read()


def train(epochs: int = 50, seq_len: int = 32, lr: float = 3e-3):
    corpus = load_corpus()
    print("=" * 65)
    print("Training Time-T 5M Causal Language Model")
    print(f"Corpus size: {len(corpus)} bytes")
    print(f"Sequence length: {seq_len}, Learning rate: {lr}")
    print("=" * 65)

    model = TimeTLanguageModel5M(max_seq_len=seq_len, seed=42)
    pcount = model.count_parameters()
    print(f"Model parameters: {pcount:,} (5.16M)")

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Prepare training sequences
    indices = []
    for i in range(0, len(corpus) - seq_len, 16):
        indices.append(i)

    print(f"Total training windows: {len(indices)}")
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        np.random.seed(epoch)
        np.random.shuffle(indices)

        for idx in indices:
            chunk = list(corpus[idx : idx + seq_len])
            x_arr = np.array([chunk[:-1]], dtype=np.int64)
            y_arr = np.array([chunk[1:]], dtype=np.int64)

            x = tensor(x_arr)
            y = tensor(y_arr)

            logits = model(x)
            B, S, V = logits.shape
            loss = nn.cross_entropy_loss(logits.reshape(B * S, V), y.reshape(B * S))

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(indices)
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.perf_counter() - start_time
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | Elapsed: {elapsed:.1f}s")
            # Sample text generation
            sample = model.generate("The ", max_new_tokens=30, temperature=0.6)
            print(f"  Sample: {repr(sample)}")

    # Save checkpoints
    out_dir = Path("models/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "timet_5m_lm.ttck"
    checkpoint.save_bin(model.lm, ckpt_path)
    print("=" * 65)
    print(f"Model checkpoint successfully saved to: {ckpt_path}")

    # Generate test prompts
    prompts = [
        "The ",
        "Knowledge is ",
        "Every journey ",
        "In the heart of ",
    ]
    print("\n--- Model Coherent Generation Demonstrations ---")
    for p in prompts:
        gen = model.generate(p, max_new_tokens=45, temperature=0.5)
        print(f"Prompt: '{p}' -> {repr(gen)}")
    print("=" * 65)


if __name__ == "__main__":
    train(epochs=25, seq_len=32, lr=3e-3)
