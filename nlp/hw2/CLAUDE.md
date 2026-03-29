# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

COMS 4705 NLP Homework 2 — implement GPT-2 from scratch in PyTorch, then fine-tune it for text classification on 20 Newsgroups.

## Commands

```bash
# Activate venv first
source .venv/bin/activate

# Run all tests
pytest tests/hw2_test.py -v

# Run a single test
pytest tests/hw2_test.py::test_loads_and_forward_pass -v
pytest tests/hw2_test.py::test_probability_tolerance -v
pytest tests/hw2_test.py::test_greedy_sampling -v
pytest tests/hw2_test.py::test_nucleus_sampling -v
pytest tests/hw2_test.py::test_classifier_loads_and_forward_pass -v
pytest tests/hw2_test.py::test_classifier_accuracy -v

# Run tests with custom parameters
pytest tests/hw2_test.py -v --num-samples 100 --batch-size 32 --gen-tokens 64
```

Key pytest CLI options (defined in `tests/conftest.py`):
- `--num-samples` (default: 256), `--max-length` (default: 512), `--batch-size` (default: 16)
- `--prefill-tokens` (default: 128), `--gen-tokens` (default: 128)
- `--temperature` (default: 0.7), `--top-p` (default: 0.95)
- `--gpt2-bin-path` (default: `./checkpoints/gpt2_model.pth`)
- `--classifier-bin-path` (default: `./checkpoints/classifier_model.pth`)
- `--classifier-accuracy-threshold` (default: 0.65)

## Architecture

### Part 1: `src/gpt2.py` — GPT-2 Language Model

Two main classes to implement:

**`GPT2LMHeadModel`** — decoder-only transformer (standard GPT-2 architecture):
- Config: vocab_size=50257, max_ctx_len=1024, d_model=768, n_layer=12, n_head=12
- `forward(input_ids, past_key_values)` → `CausalLMOutput` with logits `[batch, seq_len, vocab_size]`
- `generate(input_ids, temperature, top_p, max_new_tokens)` → `ModelOutput` with sequences

**`GPT2ForSequenceClassification`** — GPT-2 with a classification head for 20-class prediction:
- `forward(input_ids)` → `SequenceClassifierOutput` with logits `[batch, 20]`

### Part 2: `src/train.py` — Training Loop

Fine-tune `GPT2ForSequenceClassification` on 20 Newsgroups to ≥65% validation accuracy. Save checkpoint to `./checkpoints/classifier_model.pth`.

## Critical Implementation Notes

**Checkpoint weight transposition:** The official GPT-2 weights in `checkpoints/gpt2_model.pth` store linear layer weights as `[in_features, out_features]` — opposite of PyTorch's convention `[out_features, in_features]`. You must transpose them when loading.

**Shared embeddings:** The token embedding `wte.weight` (shape `[50257, 768]`) is reused as the LM head projection matrix — do not create a separate weight.

**GELU variant:** Use `F.gelu(x, approximate="tanh")` (not the exact GELU).

**Causal masking:** Multi-head attention must mask future tokens (upper-triangular mask).

**KV caching:** `past_key_values` is a list of `(key, value)` tensors per layer, used in `generate()` to avoid recomputing attention for already-processed tokens.

**Nucleus sampling:** When `temperature > 0`, sample from the top-p probability mass. When `temperature == 0.0`, use greedy (argmax).

**Numerical tolerance:** The `test_probability_tolerance` test checks that your forward pass matches HuggingFace GPT-2 within 1e-4.

## Data

- `data/openwebtext_1k_tokenized.jsonl` — pre-tokenized LM data, fields: `token_ids`, `text`
- `data/20_newsgroups_train.jsonl` / `_val.jsonl` — classification data, fields: `token_ids`, `label` (0–19), `label_text`
