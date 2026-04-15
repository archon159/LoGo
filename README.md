# LoGo: LoRA on the Go

Official repository of **"LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging"** (ACL 2026)

LoGo dynamically selects and merges multiple LoRA adapters at inference time based on each input instance, rather than using a single fixed adapter. For each input, LoGo projects the hidden state through all available LoRA adapters and selects the top-k most relevant ones using a norm- or entropy-based scoring metric. The selected adapters are then merged with weights derived from those scores during generation.

## LoRA Adapters

We trained LoRA adapters on 260 FlanV2 tasks for each supported base model and released them on Hugging Face:

- **Hub:** [archon159/LoGo-loras-collection](https://huggingface.co/archon159/LoGo-loras-collection)

Adapters are automatically downloaded when you run the code.

## Environment Setup

**Python:** 3.12.10 | **CUDA:** 11.8

```bash
pip install -r requirements.txt
```

| Package | Version |
|---|---|
| torch | 2.7.1 |
| transformers | 4.54.0 |
| peft | 0.16.0 |
| datasets | 4.0.0 |
| numpy | 1.26.4 |
| pandas | 2.3.1 |
| nltk | 3.9.1 |
| rouge_score | 0.1.2 |

## Quick Start

```bash
python3 main.py --dataset <DATASET> --base_model <BASE_MODEL> --gpu <GPU>
```

**Example:**
```bash
python3 main.py --dataset bbh.boolean_expressions --base_model Llama-3.1-8B --gpu 0
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `''` (all) | Dataset name, e.g. `bbh.boolean_expressions`. Empty string runs all datasets. |
| `--base_model` | `Llama-3.1-8B` | Base model. Options: `Llama-3.1-8B`, `Qwen2.5_7b`, `deepseek-llm-7b-base` |
| `--selection_strategy` | `norm` | Adapter selection metric. Options: `norm` (L2 norm), `entropy` (inverse entropy) |
| `--n_selected_modules` | `20` | Number of LoRA adapters to select per instance |
| `--seed` | `0` | Random seed for reproducibility |
