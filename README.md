# Sparse Features Uncover a Readout Gap in VLMs for Harmful Meme Detection

Code accompanying the paper *Decodable but Misrouted: Sparse Features Uncover a Readout Gap in Vision-Language Models for Harmful Meme Detection*. The repository trains and evaluates residual sparse autoencoders (SAEs) on six harmful-meme benchmarks: Crisis_HateMM, HarMeme, Hateful_Memes_Dataset, MAMI, MMHS150K, and MultiOFF.

The two experiment families share dataset and evaluation utilities but use different vision-language models and pretrained base SAEs.

## Experiment families

| | Gemma experiments | Qwen experiments |
|---|---|---|
| Package | `gemma3_residual_sae` | `qwen3_residual_sae` |
| Model | `google/gemma-3-4b-it` | `Qwen/Qwen3.5-9B-Base` |
| Pretrained SAE | Gemma Scope 2 `resid_post` | Qwen residual-stream TopK SAE |
| Default layer | 22 | 20 |
| Residual SAE | TopK or JumpReLU | JumpReLU |
| Entrypoint prefix | `gemma3_residual_sae.launch.residual` | `qwen3_residual_sae.launch` |

Qwen reuses the dataset loaders and normalization utilities in `gemma3_residual_sae`; keep both packages together at the repository root.

## Repository layout

```text
gemma3_residual_sae/
  residual_sae/       Gemma model, caching, SAE training, and shared datasets
  vendor_sae/         Minimal vendored TopK SAE implementation
  launch/residual/    Gemma training, inference, probing, and causal analyses
  meme_safety_probe/  Frozen sparse-probe wrapper
qwen3_residual_sae/
  config.py            Qwen experiment configuration
  qwen.py              Qwen model and pretrained-SAE integration
  models.py            Residual SAE model
  launch/              Qwen training, inference, probing, and causal analyses
```

Plot-only appendix generators, presentation files, research notes, generated outputs, and cluster launch scripts are intentionally excluded.

## Setup

Use Python 3.12 and run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Model weights are downloaded from Hugging Face. Accept the relevant model licenses and authenticate with `huggingface-cli login` when required. The experiments require CUDA GPUs; cache size and memory needs depend on dataset and model size.

Place the six dataset directories under `./data`, or pass `--dataset.root /path/to/data` to commands that load samples. Dataset-specific parsing is implemented in `gemma3_residual_sae/residual_sae/datasets.py`. Datasets without official splits receive deterministic 60/20/20 train, validation, and test partitions.

## Gemma pipeline

Run the stages in order:

```bash
# 1. Cache Gemma hidden states.
python -m gemma3_residual_sae.launch.residual.cache_dense \
  --dataset.root /path/to/data \
  --output_dir ./outputs/gemma/cache/dense

# 2. Subtract the frozen Gemma Scope reconstruction.
python -m gemma3_residual_sae.launch.residual.compute_residuals \
  --dense_cache_dir ./outputs/gemma/cache/dense \
  --output_dir ./outputs/gemma/cache/residual

# 3. Train the residual SAE.
python -m gemma3_residual_sae.launch.residual.train \
  --residual_cache_dir ./outputs/gemma/cache/residual \
  --output_dir ./outputs/gemma/checkpoints \
  --sae_architecture jump_relu

# 4. Run the native or reconstructed readout.
python -m gemma3_residual_sae.launch.residual.joint_infer \
  --dataset.root /path/to/data \
  --dataset.splits validation \
  --residual_sae_path ./outputs/gemma/checkpoints/final \
  --output_path ./outputs/gemma/predictions.jsonl

# 5. Evaluate predictions.
python -m gemma3_residual_sae.launch.residual.evaluate_predictions \
  --predictions ./outputs/gemma/predictions.jsonl \
  --output_json ./outputs/gemma/metrics.json
```

Use `--apply_residual_sae false` in `joint_infer` for the matched native-model baseline. `cache_dense` and `compute_residuals` support `--resume true`. The main sparse-readout analysis is `probe_validate`; feature discovery and intervention use `discover_features`, `probe_sae_features`, `steering_sweep`, and `causal_feature_ablation`.

## Qwen pipeline

Qwen commands use a different module prefix and checkpoint directory:

```bash
# 1. Cache Qwen hidden states.
python -m qwen3_residual_sae.launch.cache_dense \
  --dataset.root /path/to/data \
  --output_dir ./outputs/qwen/cache/dense

# 2. Subtract the frozen Qwen SAE reconstruction.
python -m qwen3_residual_sae.launch.compute_residuals \
  --dense_cache_dir ./outputs/qwen/cache/dense \
  --output_dir ./outputs/qwen/cache/residual

# 3. Train the JumpReLU residual SAE.
python -m qwen3_residual_sae.launch.train \
  --residual_cache_dir ./outputs/qwen/cache/residual \
  --output_dir ./outputs/qwen/checkpoints

# 4. Run joint inference.
python -m qwen3_residual_sae.launch.joint_infer \
  --dataset.root /path/to/data \
  --dataset.splits validation \
  --residual_sae_path ./outputs/qwen/checkpoints/best_finite \
  --output_path ./outputs/qwen/predictions.jsonl
```

Use `--apply_residual_sae false` for the Qwen native-model baseline. The core sparse-readout entrypoints are `probe_base_sae`, `probe_sae_features`, and `probe_classify`; causal controls include `causal_mediation`, `layer_self_patching`, and `visual_credit_audit`.

Every entrypoint uses `simple-parsing`; append `--help` to inspect all options. Advanced scripts in each `launch` directory consume artifacts produced by the core pipeline, so paths should be supplied explicitly rather than relying on machine-specific defaults.
