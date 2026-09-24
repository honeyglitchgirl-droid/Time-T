"""Interactive / CLI Text Generation with Time-T 5M Language Model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import timet.checkpoint as checkpoint
from models.transformer_lm_5m import TimeTLanguageModel5M


def run_generation(prompt: str = "The sun ", max_tokens: int = 50, temperature: float = 0.6):
    model = TimeTLanguageModel5M(max_seq_len=32)
    ckpt_path = Path("models/checkpoints/timet_5m_lm.ttck")
    if ckpt_path.exists():
        checkpoint.load_into(model.lm, ckpt_path)
        print(f"Loaded weights from {ckpt_path}")
    else:
        print("Using freshly initialized weights")

    print(f"\nPrompt: {prompt}")
    output = model.generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
    print(f"Generated text:\n{output}\n")


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The "
    run_generation(prompt)
