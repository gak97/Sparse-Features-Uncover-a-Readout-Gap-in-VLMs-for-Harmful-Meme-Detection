

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.launch.residual.analyze_fhm_confounder_features import _build_pairs
from gemma3_experiments.launch.residual.probe_fhm_confounder_readout import LABELS, TASK_KEY, _pair_metrics
from gemma3_experiments.launch.residual.probe_validate import (
    _build_fhm_metadata_hard_negative_pairs,
    _calibrate_threshold,
    _load_fhm_confounder_metadata,
    _per_class_metrics,
    _read_jsonl,
    _row_gold_label,
    _row_sample_key,
)
from gemma3_experiments.residual_sae import ROLE_GENERATED, ROLE_IMAGE, ROLE_PROMPT, resolve_visible_device

logger = logging.getLogger(__name__)


@dataclass
class DenseLayerSpec:
    name: str = ""
    train_dense_cache_dir: str = ""
    val_dense_cache_dir: str = ""


@dataclass
class FHMCrosscoderProbeConfig:
    crosscoder_repo_id: str = "google/gemma-scope-2-4b-it"
    crosscoder_hookpoint: str = "crosscoder/layer_9_17_22_29_width_65k_l0_medium"
    layer9_name: str = "layer9"
    layer9_train_dense_cache_dir: str = ""
    layer9_val_dense_cache_dir: str = ""
    layer17_name: str = "layer17"
    layer17_train_dense_cache_dir: str = ""
    layer17_val_dense_cache_dir: str = ""
    layer22_name: str = "layer22"
    layer22_train_dense_cache_dir: str = ""
    layer22_val_dense_cache_dir: str = ""
    layer29_name: str = "layer29"
    layer29_train_dense_cache_dir: str = ""
    layer29_val_dense_cache_dir: str = ""
    baseline_predictions_path: str = ""
    fhm_confounder_metadata_path: str = ""
    output_dir: str = ""
    eval_splits: list[str] = field(default_factory=lambda: ["validation", "val", "dev"])

    token_selection: str = "all"
    aggregation: str = "max"
    batch_size: int = 64
    chunk_size: int = 256
    shard_cache_size: int = 16

    include_pair_delta_rows: bool = True
    pair_sources: list[str] = field(default_factory=lambda: ["pseudo_img", "pseudo_text"])

    classifier_type: str = "mlp"  # mlp | logistic
    C: float = 1.0
    max_iter: int = 5000
    class_weight: str = "balanced"
    mlp_hidden_size: int = 512
    mlp_alpha: float = 1e-4
    mlp_max_iter: int = 200

    save_feature_matrix: bool = False  # save X_val + sklearn probe/scaler for causal ablation

    device: str = "cuda:0"


def _layer_specs_from_config(cfg: FHMCrosscoderProbeConfig) -> list[DenseLayerSpec]:
    specs = [
        DenseLayerSpec(cfg.layer9_name, cfg.layer9_train_dense_cache_dir, cfg.layer9_val_dense_cache_dir),
        DenseLayerSpec(cfg.layer17_name, cfg.layer17_train_dense_cache_dir, cfg.layer17_val_dense_cache_dir),
        DenseLayerSpec(cfg.layer22_name, cfg.layer22_train_dense_cache_dir, cfg.layer22_val_dense_cache_dir),
        DenseLayerSpec(cfg.layer29_name, cfg.layer29_train_dense_cache_dir, cfg.layer29_val_dense_cache_dir),
    ]
    return [spec for spec in specs if spec.train_dense_cache_dir and spec.val_dense_cache_dir]


def _role_mask(roles: torch.Tensor, token_selection: str) -> torch.Tensor:
    if token_selection == "all":
        return torch.ones_like(roles, dtype=torch.bool)
    if token_selection == "image":
        return roles == ROLE_IMAGE
    if token_selection in {"input", "prompt_and_image"}:
        return (roles == ROLE_PROMPT) | (roles == ROLE_IMAGE)
    if token_selection == "generated":
        return roles == ROLE_GENERATED
    if token_selection == "image_and_generated":
        return (roles == ROLE_IMAGE) | (roles == ROLE_GENERATED)
    if token_selection == "prompt":
        return roles == ROLE_PROMPT
    raise ValueError(f"Unsupported token_selection: {token_selection}")


def _load_fhm_rows(cache_dir: Path, split_names: set[str]) -> list[dict[str, Any]]:
    rows = []
    for row in _read_jsonl(cache_dir / "manifest.jsonl"):
        if row.get("dataset_name") != "Hateful_Memes_Dataset":
            continue
        if row.get("task_name") != "hateful_memes_binary":
            continue
        if row.get("split") not in split_names:
            continue
        if _row_gold_label(row) not in set(LABELS):
            continue
        rows.append(row)
    return rows


def _load_baseline(path: Path) -> dict[str, dict[str, Any]]:
    baseline = {}
    for row in _read_jsonl(path):
        key = f"{row['dataset_name']}::{row['task_name']}::{row['sample_id']}"
        baseline[key] = row
    return baseline


def _baseline_metrics(keys: list[str], y_val: list[str], baseline: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gold = []
    pred = []
    for key, label in zip(keys, y_val):
        base_pred = baseline.get(key, {}).get("prediction_label")
        if base_pred in set(LABELS):
            gold.append(label)
            pred.append(base_pred)
    return _per_class_metrics(gold, pred) if gold else None


class GemmaCrosscoder:
    def __init__(self, state_dict: dict[str, torch.Tensor], n_layers: int, d_model: int, device: str):
        self.state_dict_keys = sorted(state_dict)
        self.n_layers = n_layers
        self.d_model = d_model
        self.device = device
        self.w_enc = self._get_weight(state_dict, ["w_enc", "W_enc", "encoder.weight", "W_E"])
        self.b_enc = self._get_optional(state_dict, ["b_enc", "b_e", "encoder.bias"])
        self.threshold = self._get_optional(state_dict, ["threshold", "thresholds", "jump_relu_threshold", "b_mag"])
        self.b_dec = self._get_optional(state_dict, ["b_dec", "b_d", "decoder.bias"])
        self.w_enc = self.w_enc.to(device)
        if self.b_enc is not None:
            self.b_enc = self.b_enc.to(device)
        if self.threshold is not None:
            self.threshold = self.threshold.to(device)
        if self.b_dec is not None:
            self.b_dec = self.b_dec.to(device)
        self.num_latents = self._infer_num_latents()
        logger.info("Loaded crosscoder: w_enc=%s num_latents=%d keys=%s", tuple(self.w_enc.shape), self.num_latents, self.state_dict_keys)

    @staticmethod
    def _get_weight(state_dict: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
        for name in names:
            if name in state_dict:
                return state_dict[name]
        candidates = [key for key in state_dict if "enc" in key.lower() and state_dict[key].ndim in {2, 3}]
        if len(candidates) == 1:
            return state_dict[candidates[0]]
        raise KeyError(f"Could not identify crosscoder encoder weight. Keys: {sorted(state_dict)}")

    @staticmethod
    def _get_optional(state_dict: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor | None:
        for name in names:
            if name in state_dict:
                return state_dict[name]
        return None

    @classmethod
    def load_from_hub(cls, repo_id: str, hookpoint: str, n_layers: int, d_model: int, device: str) -> "GemmaCrosscoder":
        repo_path = Path(
            snapshot_download(
                repo_id,
                allow_patterns=[
                    f"{hookpoint}/*.json",
                    f"{hookpoint}/*.safetensors",
                    f"{hookpoint}/*/*.safetensors",
                    f"{hookpoint}/*.safetensors.index.json",
                ],
            )
        ) / hookpoint

        safetensor_paths = sorted(repo_path.rglob("*.safetensors"))
        if not safetensor_paths:
            available_files = sorted(path.name for path in repo_path.iterdir()) if repo_path.exists() else []
            raise FileNotFoundError(
                f"No safetensors weights found for {repo_id}/{hookpoint}. "
                f"Available files: {available_files}"
            )

        if len(safetensor_paths) == 1:
            state_dict = load_file(str(safetensor_paths[0]), device="cpu")
        else:
            # Standard single-layer SAEs (e.g. Gemma Scope resid_post) ship
            # params.safetensors (weights) + examples.safetensors (exemplars).
            # The per-layer crosscoder loader cannot handle this; load only the
            # params file and ignore exemplars.
            params_paths = [p for p in safetensor_paths if p.name == "params.safetensors"]
            if params_paths:
                logger.info(
                    "Found %d safetensors files for %s; loading params.safetensors "
                    "and ignoring non-param files (%s).",
                    len(safetensor_paths),
                    hookpoint,
                    [p.name for p in safetensor_paths if p.name != "params.safetensors"],
                )
                state_dict = load_file(str(params_paths[0]), device="cpu")
            else:
                logger.info("Found %d safetensors files for %s; combining multi-file weights.", len(safetensor_paths), hookpoint)
                state_dict = cls._load_multifile_state_dict(safetensor_paths, n_layers=n_layers)

        return cls(state_dict, n_layers=n_layers, d_model=d_model, device=device)

    @staticmethod
    def _load_multifile_state_dict(safetensor_paths: list[Path], n_layers: int) -> dict[str, torch.Tensor]:
        indexed_paths: list[tuple[int, Path]] = []
        for fpath in safetensor_paths:
            match = re.search(r"(?:^|_)layer_(\d+)\.safetensors$", fpath.name)
            if match is None:
                raise ValueError(
                    "Multiple safetensors files found, but filenames do not match expected "
                    f"per-layer pattern '*layer_<idx>.safetensors': {[path.name for path in safetensor_paths]}"
                )
            indexed_paths.append((int(match.group(1)), fpath))
        indexed_paths.sort(key=lambda item: item[0])

        if len(indexed_paths) != n_layers:
            logger.warning(
                "Expected %d layer files but found %d: %s",
                n_layers,
                len(indexed_paths),
                [path.name for _, path in indexed_paths],
            )

        per_layer_state_dicts = [load_file(str(fpath), device="cpu") for _, fpath in indexed_paths]
        keys = sorted(set().union(*(state_dict.keys() for state_dict in per_layer_state_dicts)))

        stack_keys = {
            "w_enc",
            "W_enc",
            "encoder.weight",
            "W_E",
            "w_dec",
            "W_dec",
            "decoder.weight",
            "b_dec",
            "b_d",
            "decoder.bias",
            "b_enc",
            "b_e",
            "encoder.bias",
            "threshold",
            "thresholds",
            "jump_relu_threshold",
            "b_mag",
        }
        shared_keys: set[str] = set()

        combined: dict[str, torch.Tensor] = {}
        for key in keys:
            tensors = []
            for state_dict in per_layer_state_dicts:
                if key not in state_dict:
                    tensors = []
                    break
                tensors.append(state_dict[key])
            if not tensors:
                continue

            if key in stack_keys:
                combined[key] = torch.stack(tensors, dim=0)
                continue

            first = tensors[0]
            if any(tensor.shape != first.shape for tensor in tensors[1:]):
                raise ValueError(f"Key {key} has inconsistent shapes across per-layer files: {[tuple(t.shape) for t in tensors]}")
            if any(not torch.equal(tensor, first) for tensor in tensors[1:]):
                raise ValueError(
                    f"Key {key} has differing values across per-layer files and no combination rule is defined."
                )
            combined[key] = first

        return combined

    def _infer_num_latents(self) -> int:
        if self.w_enc.ndim == 3:
            return int(self.w_enc.shape[-1])
        if self.w_enc.ndim == 2:
            return int(self.w_enc.shape[-1])
        raise ValueError(f"Unsupported encoder shape: {tuple(self.w_enc.shape)}")

    def _subtract_b_dec(self, hidden_tld: torch.Tensor) -> torch.Tensor:
        if self.b_dec is None:
            return hidden_tld
        b_dec = self.b_dec
        if b_dec.ndim == 2 and b_dec.shape == (self.n_layers, self.d_model):
            return hidden_tld - b_dec[None, :, :]
        if b_dec.ndim == 1 and b_dec.numel() == self.n_layers * self.d_model:
            return hidden_tld - b_dec.reshape(self.n_layers, self.d_model)[None, :, :]
        if b_dec.ndim == 1 and b_dec.numel() == self.d_model:
            return hidden_tld - b_dec[None, None, :]
        logger.warning("Ignoring unsupported b_dec shape: %s", tuple(b_dec.shape))
        return hidden_tld

    def encode(self, hidden_tld: torch.Tensor) -> torch.Tensor:
        hidden_tld = self._subtract_b_dec(hidden_tld.to(self.device, dtype=self.w_enc.dtype))
        if self.w_enc.ndim == 3:
            per_layer_bias = self.b_enc is not None and self.b_enc.ndim == 2 and self.b_enc.shape[0] == self.n_layers
            per_layer_threshold = (
                self.threshold is not None and self.threshold.ndim == 2 and self.threshold.shape[0] == self.n_layers
            )
            if per_layer_bias or per_layer_threshold:
                if self.w_enc.shape[0] == self.n_layers and self.w_enc.shape[1] == self.d_model:
                    pre_lf = torch.einsum("tld,ldf->tlf", hidden_tld, self.w_enc)
                elif self.w_enc.shape[0] == self.d_model and self.w_enc.shape[1] == self.n_layers:
                    pre_lf = torch.einsum("tld,dlf->tlf", hidden_tld, self.w_enc)
                else:
                    raise ValueError(
                        f"Unsupported 3D w_enc shape for n_layers={self.n_layers}, d_model={self.d_model}: {tuple(self.w_enc.shape)}"
                    )
                if per_layer_bias:
                    pre_lf = pre_lf + self.b_enc[None, :, :]
                if per_layer_threshold:
                    acts_lf = pre_lf * (pre_lf > self.threshold[None, :, :]).to(pre_lf.dtype)
                else:
                    acts_lf = torch.relu(pre_lf)
                return acts_lf.sum(dim=1)

            if self.w_enc.shape[0] == self.n_layers and self.w_enc.shape[1] == self.d_model:
                pre = torch.einsum("tld,ldf->tf", hidden_tld, self.w_enc)
            elif self.w_enc.shape[0] == self.d_model and self.w_enc.shape[1] == self.n_layers:
                pre = torch.einsum("tld,dlf->tf", hidden_tld, self.w_enc)
            else:
                raise ValueError(f"Unsupported 3D w_enc shape for n_layers={self.n_layers}, d_model={self.d_model}: {tuple(self.w_enc.shape)}")
        elif self.w_enc.ndim == 2:
            flat = hidden_tld.reshape(hidden_tld.shape[0], self.n_layers * self.d_model)
            if self.w_enc.shape[0] != flat.shape[1]:
                raise ValueError(f"Unsupported 2D w_enc shape {tuple(self.w_enc.shape)} for flattened input {tuple(flat.shape)}")
            pre = flat @ self.w_enc
        else:
            raise ValueError(f"Unsupported w_enc ndim: {self.w_enc.ndim}")
        if self.b_enc is not None:
            pre = pre + self.b_enc
        if self.threshold is not None and self.threshold.shape[-1] == pre.shape[-1]:
            return pre * (pre > self.threshold).to(pre.dtype)
        return torch.relu(pre)


def _common_keys(key_lists: list[list[str]]) -> list[str]:
    common = set(key_lists[0])
    for keys in key_lists[1:]:
        common &= set(keys)
    return [key for key in key_lists[0] if key in common]


def _manifest_key(row: dict[str, Any]) -> str:
    return _row_sample_key(row)


class DenseShardReader:
    def __init__(self, layer_specs: list[DenseLayerSpec], max_cache_size: int, token_selection: str):
        self.layer_specs = layer_specs
        self.max_cache_size = max_cache_size
        self.token_selection = token_selection
        self._role_mismatch_warning_count = 0
        self.cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}

    def _load_shard(self, spec: DenseLayerSpec, cache_dir: str, shard_file: str) -> dict[str, torch.Tensor]:
        key = (spec.name, cache_dir, shard_file)
        if key in self.cache:
            return self.cache[key]
        dense_fpath = Path(cache_dir) / shard_file
        with safe_open(str(dense_fpath), framework="pt", device="cpu") as fd:
            payload = {
                "hidden": fd.get_tensor("hidden"),
                "token_roles": fd.get_tensor("token_roles"),
            }
        if len(self.cache) >= self.max_cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = payload
        return payload

    def get_sample(self, rows_by_layer: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        hiddens = []
        role_rows = []
        key = _manifest_key(rows_by_layer[0])
        for spec, row in zip(self.layer_specs, rows_by_layer, strict=True):
            cache_dir = row["_cache_dir"]
            shard = self._load_shard(spec, cache_dir, row["shard_file"])
            start = int(row["offset"])
            end = start + int(row["length"])
            hidden = shard["hidden"][start:end]
            layer_roles = shard["token_roles"][start:end]
            hiddens.append(hidden)
            role_rows.append(layer_roles)

        lengths = [hidden.shape[0] for hidden in hiddens]
        min_len = int(min(lengths))
        assert min_len > 0, f"No tokens available for sample {key}."
        if len(set(lengths)) != 1 and self._role_mismatch_warning_count < 20:
            logger.warning("Token length mismatch for %s across layers %s; truncating to %d.", key, lengths, min_len)
            self._role_mismatch_warning_count += 1

        cropped_hiddens = [hidden[:min_len] for hidden in hiddens]
        roles = role_rows[0][:min_len]
        for spec, layer_roles in zip(self.layer_specs[1:], role_rows[1:], strict=True):
            layer_roles = layer_roles[:min_len]
            if not torch.equal(roles, layer_roles):
                if self.token_selection != "all":
                    raise AssertionError(f"Token role mismatch for {key} at layer {spec.name}")
                if self._role_mismatch_warning_count < 20:
                    logger.warning(
                        "Token role mismatch for %s at layer %s; using roles from %s because token_selection=all.",
                        key,
                        spec.name,
                        self.layer_specs[0].name,
                    )
                    self._role_mismatch_warning_count += 1
        return torch.stack(cropped_hiddens, dim=1), roles


def _aggregate_acts(acts: torch.Tensor, aggregation: str) -> np.ndarray:
    if acts.numel() == 0:
        return np.zeros((acts.shape[-1],), dtype=np.float32)
    if aggregation == "max":
        return acts.max(dim=0).values.detach().float().cpu().numpy()
    if aggregation == "mean":
        return acts.mean(dim=0).detach().float().cpu().numpy()
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def _top_classifier_features(clf: Any, classes: list[str], top_k: int = 200) -> dict[str, list[dict[str, Any]]]:
    if hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_)
        if coef.shape[0] == 1 and len(classes) == 2:
            rows = [-coef[0], coef[0]]
        else:
            rows = [coef[i] for i in range(len(classes))]
    elif hasattr(clf, "coefs_") and len(getattr(clf, "coefs_")) >= 2:
        # First-order MLP saliency proxy: input->hidden weights composed with hidden->class weights.
        input_hidden = np.asarray(clf.coefs_[0])
        hidden_out = np.asarray(clf.coefs_[-1])
        # Binary sklearn MLP uses a single output unit (hidden_out shape: [hidden, 1]).
        # Treat class[1] as positive (weight = +col0) and class[0] as negative (weight = -col0).
        if hidden_out.shape[1] == 1 and len(classes) == 2:
            pos_weights = (input_hidden * hidden_out[:, 0][None, :]).sum(axis=1)
            rows = [-pos_weights, pos_weights]
        else:
            rows = [(input_hidden * hidden_out[:, i][None, :]).sum(axis=1) for i in range(len(classes))]
    else:
        return {}

    out = {}
    for class_name, weights in zip(classes, rows, strict=True):
        order = np.argsort(np.abs(weights))[::-1][:top_k]
        out[class_name] = [
            {"feature_idx": int(feature_i), "weight": round(float(weights[feature_i]), 6), "abs_weight": round(float(abs(weights[feature_i])), 6)}
            for feature_i in order
        ]
    return out


def _extract_crosscoder_features(
    rows_by_layer: list[list[dict[str, Any]]],
    key_lists: list[list[str]],
    layer_specs: list[DenseLayerSpec],
    crosscoder: GemmaCrosscoder,
    cfg: FHMCrosscoderProbeConfig,
) -> tuple[list[str], Any]:
    import scipy.sparse

    common_keys = _common_keys(key_lists)
    row_lookup_by_layer = [{_manifest_key(row): row for row in rows} for rows in rows_by_layer]
    reader = DenseShardReader(layer_specs, cfg.shard_cache_size, cfg.token_selection)
    feature_rows = []
    for batch_start in range(0, len(common_keys), cfg.batch_size):
        batch_keys = common_keys[batch_start : batch_start + cfg.batch_size]
        for key in batch_keys:
            sample_rows = [lookup[key] for lookup in row_lookup_by_layer]
            hidden_tld, roles = reader.get_sample(sample_rows)
            mask = _role_mask(roles, cfg.token_selection)
            selected = hidden_tld[mask]
            chunks = []
            for start in range(0, selected.shape[0], cfg.chunk_size):
                chunks.append(crosscoder.encode(selected[start : start + cfg.chunk_size]))
            if chunks:
                acts = torch.cat(chunks, dim=0)
                vec = _aggregate_acts(acts, cfg.aggregation)
            else:
                vec = np.zeros((crosscoder.num_latents,), dtype=np.float32)
            feature_rows.append(scipy.sparse.csr_matrix(vec.reshape(1, -1)))
        logger.info("Extracted crosscoder features: %d/%d", min(batch_start + cfg.batch_size, len(common_keys)), len(common_keys))
    return common_keys, scipy.sparse.vstack(feature_rows, format="csr", dtype=np.float32)


def run_crosscoder_probe(cfg: FHMCrosscoderProbeConfig) -> None:
    import scipy.sparse
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder, MaxAbsScaler

    layer_specs = _layer_specs_from_config(cfg)
    assert len(layer_specs) >= 1, (
        "Expected at least one configured dense layer. "
        "For single-layer resid_post probes supply only the layer22_* fields."
    )
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = Path(cfg.baseline_predictions_path) if cfg.baseline_predictions_path else None
    metadata_path = Path(cfg.fhm_confounder_metadata_path)
    if baseline_path is not None:
        assert baseline_path.exists(), f"Baseline predictions missing: {baseline_path}"
    assert metadata_path.exists(), f"FHM metadata missing: {metadata_path}"

    train_rows_by_layer = []
    val_rows_by_layer = []
    train_key_lists = []
    val_key_lists = []
    for spec in layer_specs:
        train_cache_dir = Path(spec.train_dense_cache_dir)
        val_cache_dir = Path(spec.val_dense_cache_dir)
        assert train_cache_dir.exists(), f"Train dense cache missing for {spec.name}: {train_cache_dir}"
        assert val_cache_dir.exists(), f"Validation dense cache missing for {spec.name}: {val_cache_dir}"
        train_rows = _load_fhm_rows(train_cache_dir, {"train"})
        val_rows = _load_fhm_rows(val_cache_dir, set(cfg.eval_splits))
        for row in train_rows:
            row["_cache_dir"] = str(train_cache_dir)
        for row in val_rows:
            row["_cache_dir"] = str(val_cache_dir)
        assert train_rows and val_rows, (
            f"Missing FHM train/val rows for {spec.name}: train={train_cache_dir} val={val_cache_dir}"
        )
        train_rows_by_layer.append(train_rows)
        val_rows_by_layer.append(val_rows)
        train_key_lists.append([_manifest_key(row) for row in train_rows])
        val_key_lists.append([_manifest_key(row) for row in val_rows])
        logger.info("%s rows: train=%d val=%d", spec.name, len(train_rows), len(val_rows))

    first_shard = Path(train_rows_by_layer[0][0]["_cache_dir"]) / train_rows_by_layer[0][0]["shard_file"]
    with safe_open(str(first_shard), framework="pt", device="cpu") as fd:
        d_model = int(fd.get_tensor("hidden").shape[1])
    device = resolve_visible_device(cfg.device)
    crosscoder = GemmaCrosscoder.load_from_hub(
        cfg.crosscoder_repo_id,
        cfg.crosscoder_hookpoint,
        n_layers=len(layer_specs),
        d_model=d_model,
        device=device,
    )

    train_keys, X_train_all = _extract_crosscoder_features(train_rows_by_layer, train_key_lists, layer_specs, crosscoder, cfg)
    val_keys, X_val = _extract_crosscoder_features(val_rows_by_layer, val_key_lists, layer_specs, crosscoder, cfg)
    train_row_by_key = {_manifest_key(row): row for row in train_rows_by_layer[0]}
    val_row_by_key = {_manifest_key(row): row for row in val_rows_by_layer[0]}
    y_train = [_row_gold_label(train_row_by_key[key]) for key in train_keys]
    y_val = [_row_gold_label(val_row_by_key[key]) for key in val_keys]

    X_train = X_train_all
    hard_negative_stats = None
    if cfg.include_pair_delta_rows:
        train_key_to_idx = {key: idx for idx, key in enumerate(train_keys)}
        delta_csr, delta_labels, hard_negative_stats = _build_fhm_metadata_hard_negative_pairs(
            train_rows_by_layer[0],
            X_train_all,
            train_key_to_idx,
            metadata_path,
            cfg.pair_sources,
        )
        if delta_csr.shape[0]:
            X_train = scipy.sparse.vstack([X_train_all, delta_csr], format="csr")
            y_train = list(y_train) + delta_labels
            logger.info("Pair delta rows: train %d -> %d", len(train_keys), X_train.shape[0])

    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    classes = le.classes_.tolist()
    assert set(classes) == set(LABELS), f"Unexpected classes: {classes}"

    if cfg.classifier_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(cfg.mlp_hidden_size,),
            activation="relu",
            alpha=cfg.mlp_alpha,
            max_iter=cfg.mlp_max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
        )
        clf.fit(X_train_scaled.toarray(), y_train_enc)
        y_pred_enc = clf.predict(X_val_scaled.toarray())
        y_proba = clf.predict_proba(X_val_scaled.toarray())
    else:
        clf = LogisticRegression(
            C=cfg.C,
            max_iter=cfg.max_iter,
            solver="liblinear",
            class_weight=cfg.class_weight if cfg.class_weight != "None" else None,
        )
        clf.fit(X_train_scaled, y_train_enc)
        y_pred_enc = clf.predict(X_val_scaled)
        y_proba = clf.predict_proba(X_val_scaled)

    y_pred = le.inverse_transform(y_pred_enc).tolist()
    raw_metrics = _per_class_metrics(y_val, y_pred)
    best_t, _, best_cal_preds = _calibrate_threshold(y_val, y_proba, classes, "hateful")
    calibrated_metrics = _per_class_metrics(y_val, best_cal_preds)
    calibrated_metrics["threshold"] = round(best_t, 2)

    threshold_sweep = []
    pos_idx = classes.index("hateful")
    for t_int in range(10, 91):
        threshold = t_int / 100.0
        preds = ["hateful" if prob[pos_idx] >= threshold else "not hateful" for prob in y_proba]
        metrics = _per_class_metrics(y_val, preds)
        threshold_sweep.append({"threshold": threshold, "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"]})

    proba_rows = [
        {classes[col_i]: round(float(y_proba[row_i, col_i]), 6) for col_i in range(len(classes))}
        for row_i in range(len(val_keys))
    ]
    baseline = {} if baseline_path is None else _load_baseline(baseline_path)
    baseline_metrics = _baseline_metrics(val_keys, y_val, baseline)

    metadata = _load_fhm_confounder_metadata(metadata_path)
    val_key_set = set(val_keys)
    val_pairs_by_source = {
        source: _build_pairs(val_rows_by_layer[0], metadata, val_key_set, source)
        for source in cfg.pair_sources
    }
    pred_by_key = dict(zip(val_keys, y_pred))
    proba_by_key = dict(zip(val_keys, proba_rows))
    pair_metrics = {
        source: _pair_metrics(val_pairs_by_source[source], pred_by_key, proba_by_key)
        for source in cfg.pair_sources
    }
    top_classifier_features = _top_classifier_features(clf, classes)

    metrics = {
        "config": {
            **vars(cfg),
            "layer_specs": [vars(spec) for spec in layer_specs],
        },
        "task_key": TASK_KEY,
        "n_train_real": len(train_keys),
        "n_train_after_delta": int(X_train.shape[0]),
        "n_val": len(val_keys),
        "n_features_total": int(X_train_all.shape[1]),
        "class_counts_train": dict(Counter(y_train)),
        "class_counts_val": dict(Counter(y_val)),
        "hard_negative_stats": hard_negative_stats,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
        "baseline_metrics": baseline_metrics,
        "threshold_sweep": threshold_sweep,
        "pair_metrics": {
            source: {key: value for key, value in source_metrics.items() if key != "rows"}
            for source, source_metrics in pair_metrics.items()
        },
        "crosscoder_state_keys": crosscoder.state_dict_keys,
        "top_classifier_features": top_classifier_features,
    }
    (output_dir / "crosscoder_probe_metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "crosscoder_top_features.json").write_text(json.dumps(top_classifier_features, indent=2))
    with (output_dir / "probe_val_predictions.jsonl").open("w") as fd:
        for key, gold, pred, proba in zip(val_keys, y_val, y_pred, proba_rows):
            base_row = baseline.get(key)
            fd.write(json.dumps({
                "sample_key": key,
                "task_key": TASK_KEY,
                "gold_label": gold,
                "probe_pred": pred,
                "probe_proba": proba,
                "baseline_pred": base_row.get("prediction_label") if base_row else None,
                "baseline_label_scores": base_row.get("label_scores") if base_row else None,
                "probe_correct": pred == gold,
                "baseline_correct": (base_row.get("prediction_label") == gold) if base_row else None,
            }) + "\n")
    with (output_dir / "validation_pair_metrics.jsonl").open("w") as fd:
        for source, source_metrics in pair_metrics.items():
            for row in source_metrics.get("rows", []):
                fd.write(json.dumps({"source": source, **row}) + "\n")

    if cfg.save_feature_matrix:
        import joblib
        import scipy.sparse
        # Save unscaled X_val (consistent with causal_feature_ablation.py which expects unscaled).
        mat_path = output_dir / "feature_matrix.npz"
        scipy.sparse.save_npz(str(mat_path), X_val.tocsr())
        # Also save keys array for row-to-sample alignment.
        np.save(str(output_dir / "feature_matrix_keys.npy"), np.array(val_keys, dtype=object))
        joblib.dump(clf,    output_dir / "probe.joblib")
        joblib.dump(scaler, output_dir / "scaler.joblib")
        joblib.dump(le,     output_dir / "label_encoder.joblib")
        logger.info("Saved feature_matrix.npz + probe/scaler/label_encoder joblib to %s", output_dir)

    logger.info(
        "FHM crosscoder probe: raw_f1=%.4f calibrated_f1=%.4f baseline_f1=%.4f",
        raw_metrics["macro_f1"], calibrated_metrics["macro_f1"], baseline_metrics["macro_f1"] if baseline_metrics else float("nan"),
    )
    logger.info("Wrote outputs to %s", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(FHMCrosscoderProbeConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run_crosscoder_probe(cfg)


if __name__ == "__main__":
    main()
