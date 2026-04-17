# LoGo: LoRA on the Go

Official repository of **"LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging"** (ACL 2026)

LoGo dynamically selects and merges multiple LoRA adapters at inference time based on each input instance, rather than using a single fixed adapter. For each input, LoGo projects the hidden state through all available LoRA adapters and selects the top-k most relevant ones using a norm- or entropy-based scoring metric. The selected adapters are then merged with weights derived from those scores during generation.

## LoRA Adapters

We trained LoRA adapters on 260 FlanV2 tasks for each supported base model and released them on Hugging Face:

- **Hub:** [archon159/LoGo-loras-collection](https://huggingface.co/archon159/LoGo-loras-collection)

Adapters are automatically downloaded when you run the code.

## Environment Setup

**Python:** 3.12.10 | **CUDA:** 11.8

**1. Create and activate a virtual environment**
```bash
python3 -m venv logo_env
source logo_env/bin/activate
```

**2. Install dependencies**
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

## Hugging Face Setup

**Cache directory**

By default, models and adapters are cached in `~/.cache/huggingface`, which may fill up quickly (~30GB per model). Set `HF_HOME` to a directory with sufficient space before running:

```bash
export HF_HOME=/path/to/large/storage/.cache/huggingface
```

**Access token (required for Llama)**

Llama-3.1-8B is a gated model. You need to accept the license on Hugging Face and authenticate:

```bash
huggingface-cli login
```

Or set the token as an environment variable:

```bash
export HF_TOKEN=<your_token>
```

Qwen and DeepSeek models are publicly accessible and do not require a token.

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
| `--base_model` | `Llama-3.1-8B` | Base model. Options: `Llama-3.1-8B`, `Qwen2.5-7B`, `deepseek-llm-7b-base` |
| `--selection_strategy` | `norm` | Adapter selection metric. Options: `norm` (L2 norm), `entropy` (inverse entropy) |
| `--n_selected_modules` | `20` | Number of LoRA adapters to select per instance |
| `--seed` | `0` | Random seed for reproducibility |
| `--generation_early_stop` | `False` | Stop generation at the first newline. Enable this for faster generation. |

**Expected output:**
```
Accuracy: 71.33
results/Llama-3.1-8B_selection_norm_n_20_seed_0/bbh.boolean_expressions
```

## Results

Results are saved to `results/{base_model}_selection_{strategy}_n_{n_modules}_seed_{seed}/{dataset}/`:

| File | Description |
|---|---|
| `eval_results.json` | Per-instance evaluation scores (accuracy / BLEU / ROUGE) |
| `predictions.json` | Model-generated text for each instance |
| `selected_loras.json` | Which LoRA adapters were selected per instance |
| `weights.json` | Adapter merging weights derived from selection scores |
| `metric_dict.json` | Raw selection metric scores (norm or entropy) per instance |
| `time_dict.json` | Timing breakdown: model loading and inference duration |

## Citation

If you use this work, please cite:

```bibtex
@article{lee2025lora,
  title={LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging},
  author={Lee, Seungeon and Das, Soumi and Gupta, Manish and Gummadi, Krishna P.},
  journal={arXiv preprint arXiv:2511.07129},
  year={2025}
}
```
