"""Time-T 5M Parameter Causal Language Model (TransformerLM-5M).

Specifications:
- Parameters: 5,162,080 (~5.16 Million parameters)
- Vocab Size: 256 (Byte-level encoding, supports arbitrary UTF-8 text without OOV)
- Embedding Dimension: 288
- Transformer Layers: 5 Pre-LN Transformer Blocks
- Attention Heads: 6 (dimension 48 per head)
- Max Sequence Length: 64 (configurable up to 256)
- Checkpoint Format: Native Time-T Binary v2 (.ttck) + HuggingFace Safetensors
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional, List

import numpy as np

import timet.nn as nn
import timet.optim as optim
from timet.tensor import Tensor, tensor
import timet.checkpoint as checkpoint


class TimeTLanguageModel5M(nn.Module):
    def __init__(self, vocab_size: int = 256, embed_dim: int = 288,
                 max_seq_len: int = 64, num_heads: int = 6,
                 num_layers: int = 5, seed: int = 42):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.lm = nn.TransformerLM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            num_layers=num_layers,
            seed=seed,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.lm(token_ids)

    def parameters(self) -> List[Tensor]:
        return self.lm.parameters()

    def count_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters())

    def generate(self, prompt: str, max_new_tokens: int = 50,
                 temperature: float = 0.7, top_k: int = 40) -> str:
        """Autoregressive text generation given a prompt string."""
        tokens = list(prompt.encode("utf-8"))
        for _ in range(max_new_tokens):
            ctx = tokens[-self.max_seq_len:]
            inp = tensor(np.array([ctx], dtype=np.int64))
            logits = self.lm(inp).data[0, -1, :]  # shape: (vocab_size,)

            if temperature <= 0.0:
                next_tok = int(np.argmax(logits))
            else:
                scaled = logits / max(temperature, 1e-5)
                # Top-K filtering
                if top_k > 0 and top_k < len(scaled):
                    indices_to_remove = np.argsort(scaled)[:-top_k]
                    scaled[indices_to_remove] = -1e9
                # Stable softmax
                exp_l = np.exp(scaled - np.max(scaled))
                probs = exp_l / np.sum(exp_l)
                next_tok = int(np.random.choice(len(probs), p=probs))

            tokens.append(next_tok)

        return bytes(tokens).decode("utf-8", errors="replace")


def create_model(seed: int = 42) -> TimeTLanguageModel5M:
    return TimeTLanguageModel5M(seed=seed)


if __name__ == "__main__":
    m = create_model()
    pcount = m.count_parameters()
    print(f"Time-T 5M Language Model created successfully.")
    print(f"Architecture: 5 Layers, 6 Heads, 288 Embed Dim, 256 Vocab")
    print(f"Exact Parameter Count: {pcount:,} parameters ({pcount / 1e6:.2f}M)")
