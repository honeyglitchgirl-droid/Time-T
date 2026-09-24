# Time-T 5M Language Model (TransformerLM-5M)

A pure Time-T Transformer Causal Language Model engineered with ~5.16 Million parameters for coherent autoregressive text generation.

## Model Specifications

- **Total Parameters**: 5,162,080 (5.16M)
- **Architecture**: Decoder-only Causal Transformer (Pre-LN)
- **Layers**: 5 Transformer Blocks
- **Attention Heads**: 6 Multi-Head Attention heads (dimension 48 per head)
- **Embedding Dimension**: 288
- **Vocabulary Size**: 256 (Byte-level tokenization, zero OOV errors across all UTF-8 characters)
- **Context Window**: 32 tokens (expandable up to 256)
- **Serialization Format**: Time-T Binary Checkpoint (`.ttck`)

## Training & Generation

```bash
# Train the model
python3 tools/train_5m_model.py

# Generate text from a prompt
python3 tools/generate_5m.py "The "
```
