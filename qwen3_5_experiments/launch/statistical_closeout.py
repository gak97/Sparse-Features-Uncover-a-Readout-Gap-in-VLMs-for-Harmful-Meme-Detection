

from __future__ import annotations

import gc
import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from simple_parsing import Serializable, list_field, parse

from ..qwen import QwenSae
from ..task_labels import (
    BINARY_TASK_DATASET,
    BINARY_TASK_FIELD,
    BINARY_TASKS,
    MULTICLASS_TASKS,
    fit_scaled_logreg,
)
from .direct_logit_projection import _extract_features_with_ids, _read_jsonl

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskSpec:
    dataset_name: str
    task_name: str
    field: str
    positive_label: str
    train_cache: str
    eval_cache: str
    train_split: str
    eval_split: str
    positive_group: set[str] | None = None


def _specs(root: str) -> list[TaskSpec]:
    dense = f"{root}/qwen_residual_sae_cache"
    specs = [
        TaskSpec(ds, task, BINARY_TASK_FIELD[task], positive, f"{dense}/dense", f"{dense}/dense_val_test", "train", "validation" if ds == "Crisis_HateMM" else "test")
        for task, positive in BINARY_TASKS.items()
        for ds in [BINARY_TASK_DATASET[task]]
    ]
    for task, spec in MULTICLASS_TASKS.items():
        specs.append(TaskSpec(
            spec.dataset_name, task, spec.field, spec.binary_positive,
            f"{dense}/dense", f"{dense}/dense_val_test", "train", "test",
            {label for label in spec.classes if spec.collapse(label) == spec.binary_positive},
        ))
    return specs


@dataclass
class StatisticalCloseoutConfig(Serializable):
    activations_root: str = "./outputs"
    baseline_jsonl: str = "./outputs/qwen_infer/baseline_proper.jsonl"
    hook_jsonl: str = "./outputs/qwen_infer/hook_l0_0001_alpha0_35_proper.jsonl"
    fhm_metadata_path: str = "./data/ivl_plus_gemini_captions_complete.json"
    output_dir: str = "./outputs/qwen_statistical_closeout"
    sae_repo_id: str = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
    sae_layer: int = 20
    device: str = "cuda:0"
    probe_seeds: list[int] = list_field(42, 43, 44, 45, 46)
    bilinear_seeds: list[int] = list_field(42, 43, 44, 45, 46)
    bootstrap_replicates: int = 2000
    causal_bootstrap_replicates: int = 2000
    seed: int = 20260816


def _binary_label(value: str, spec: TaskSpec) -> int:
    if spec.positive_group is not None:
        return int(value in spec.positive_group)
    return int(value.lower() == spec.positive_label.lower())


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _prediction_score(row: dict, spec: TaskSpec) -> float | None:
    if row.get("decode_strategy") == "yes_no":
        probability = row.get("conf_gap")
        if probability is None:
            return None
        probability = min(max(float(probability), 1e-6), 1 - 1e-6)
        return math.log(probability / (1 - probability))
    scores = row.get("label_scores") or {}
    if not scores:
        return None
    if spec.positive_group is None:
        positive = [value for label, value in scores.items() if label.lower() == spec.positive_label.lower()]
        negative = [value for label, value in scores.items() if label.lower() != spec.positive_label.lower()]
    else:
        positive = [value for label, value in scores.items() if label in spec.positive_group]
        negative = [value for label, value in scores.items() if label not in spec.positive_group]
    return _logsumexp(positive) - _logsumexp(negative) if positive and negative else None


def _locked_predictions(rows: list[dict], spec: TaskSpec) -> dict[str, tuple[int, int, float]]:
    output = {}
    for row in rows:
        if row.get("dataset_name") != spec.dataset_name or row.get("task_name") != spec.task_name:
            continue
        gold_value = (row.get("gold_fields") or {}).get(spec.field)
        prediction_value = row.get("prediction_label")
        score = _prediction_score(row, spec)
        if gold_value is None or prediction_value is None or score is None:
            continue
        gold = _binary_label(str(gold_value), spec)
        output[str(row["sample_id"])] = (gold, _binary_label(str(prediction_value), spec), score)
    return output


def _f1(gold: np.ndarray, prediction: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(gold, prediction, average="macro", zero_division=0))


def _bootstrap_delta(gold: np.ndarray, left: np.ndarray, right: np.ndarray, replicates: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(gold == label) for label in np.unique(gold)]
    deltas = np.empty(replicates)
    for replicate in range(replicates):
        indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        deltas[replicate] = _f1(gold[indices], right[indices]) - _f1(gold[indices], left[indices])
    observed = _f1(gold, right) - _f1(gold, left)
    return {
        "observed_delta_macro_f1": round(float(observed), 6),
        "ci95": [round(float(value), 6) for value in np.quantile(deltas, [0.025, 0.975])],
        "p_bootstrap_two_sided": round(float(min(1.0, 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0)))), 6),
    }


def _mcnemar(gold: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict:
    from scipy.stats import binomtest
    left_correct, right_correct = left == gold, right == gold
    left_only = int(np.logical_and(left_correct, ~right_correct).sum())
    right_only = int(np.logical_and(~left_correct, right_correct).sum())
    p_value = float(binomtest(min(left_only, right_only), n=left_only + right_only, p=0.5).pvalue) if left_only + right_only else 1.0
    return {"left_only_correct": left_only, "right_only_correct": right_only, "mcnemar_exact_p": round(p_value, 8)}


def _classwise(gold: np.ndarray, prediction: np.ndarray) -> dict:
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(gold, prediction, labels=[0, 1], zero_division=0)
    return {
        str(label): {"precision": round(float(precision[label]), 6), "recall": round(float(recall[label]), 6), "f1": round(float(f1[label]), 6), "support": int(support[label])}
        for label in (0, 1)
    }


def _calibration(gold: np.ndarray, score: np.ndarray, n_bins: int = 10) -> dict:
    probability = 1 / (1 + np.exp(-score))
    bins = np.linspace(0, 1, n_bins + 1)
    values = []
    for index in range(n_bins):
        mask = (probability >= bins[index]) & ((probability < bins[index + 1]) if index + 1 < n_bins else (probability <= bins[index + 1]))
        if mask.any():
            values.append({"lower": round(float(bins[index]), 3), "upper": round(float(bins[index + 1]), 3), "n": int(mask.sum()), "mean_probability": round(float(probability[mask].mean()), 6), "empirical_positive_rate": round(float(gold[mask].mean()), 6)})
    return {"brier": round(float(np.mean((probability - gold) ** 2)), 6), "bins": values}


def _fit_probe_seeds(sae: QwenSae, spec: TaskSpec, seeds: list[int], device: str) -> tuple[dict[int, dict[str, tuple[int, int, float]]], dict]:
    train_rows = [row for row in _read_jsonl(Path(spec.train_cache) / "manifest.jsonl") if row.get("dataset_name") == spec.dataset_name and row.get("task_name") == spec.task_name and row.get("split") == spec.train_split]
    eval_rows = [row for row in _read_jsonl(Path(spec.eval_cache) / "manifest.jsonl") if row.get("dataset_name") == spec.dataset_name and row.get("task_name") == spec.task_name and row.get("split") == spec.eval_split]
    train_x, train_raw, _ = _extract_features_with_ids(sae, Path(spec.train_cache), train_rows, spec.field, "max", device)
    eval_x, eval_raw, eval_ids = _extract_features_with_ids(sae, Path(spec.eval_cache), eval_rows, spec.field, "max", device)
    train_y = np.asarray([_binary_label(value, spec) for value in train_raw])
    eval_y = np.asarray([_binary_label(value, spec) for value in eval_raw])
    output, seed_report = {}, {}
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(len(train_y))
        classifier, scaler = fit_scaled_logreg(train_x[order], train_y[order])
        probability = classifier.predict_proba(scaler.transform(eval_x))[:, list(classifier.classes_).index(1)]
        score = classifier.decision_function(scaler.transform(eval_x))
        prediction = (probability >= 0.5).astype(np.int32)
        output[seed] = {str(sample_id): (int(gold), int(pred), float(logit)) for sample_id, gold, pred, logit in zip(eval_ids, eval_y, prediction, score)}
        seed_report[str(seed)] = {"macro_f1": round(_f1(eval_y, prediction), 6), "classwise": _classwise(eval_y, prediction), "calibration": _calibration(eval_y, score)}
    return output, {"n_train": int(len(train_y)), "n_eval": int(len(eval_y)), "seeds": seed_report}


def _normalise_fhm_id(value: object) -> str:
    return str(value).strip().lower().removesuffix(".png").removesuffix(".jpg")


def _fhm_pairs(predictions: dict[str, tuple[int, int, float]], metadata_path: Path, source: str, replicates: int, seed: int) -> dict:
    payload = json.loads(metadata_path.read_text())
    entries = payload.values() if isinstance(payload, dict) else payload
    metadata = {_normalise_fhm_id(row.get("id", row.get("img", ""))): row for row in entries if isinstance(row, dict)}
    column = f"{source}_idx"
    groups: dict[str, list[str]] = defaultdict(list)
    for sample_id in predictions:
        value = metadata.get(_normalise_fhm_id(sample_id), {}).get(column)
        if value is not None and str(value) not in {"", "None", "nan"}:
            groups[str(value)].append(sample_id)
    pairs = []
    for sample_ids in groups.values():
        positive = [sample_id for sample_id in sample_ids if predictions[sample_id][0] == 1]
        negative = [sample_id for sample_id in sample_ids if predictions[sample_id][0] == 0]
        pairs.extend((pos, neg) for pos in positive for neg in negative)
    if not pairs:
        return {"n_pairs": 0}
    def metric(selection: list[tuple[str, str]]) -> np.ndarray:
        both = [predictions[pos][1] == 1 and predictions[neg][1] == 0 for pos, neg in selection]
        separated = [predictions[pos][1] != predictions[neg][1] for pos, neg in selection]
        ranked = [predictions[pos][2] > predictions[neg][2] for pos, neg in selection]
        return np.asarray([np.mean(both), np.mean(separated), np.mean(ranked)])
    observed = metric(pairs)
    rng = np.random.default_rng(seed)
    boot = np.stack([metric([pairs[index] for index in rng.choice(len(pairs), len(pairs), replace=True)]) for _ in range(replicates)])
    names = ("both_correct", "separated", "positive_margin")
    return {"n_pairs": len(pairs), **{name: {"estimate": round(float(observed[index]), 6), "ci95": [round(float(value), 6) for value in np.quantile(boot[:, index], [0.025, 0.975])]} for index, name in enumerate(names)}}


def _bootstrap_examples(rows: list[dict], value_key: str, replicates: int, seed: int) -> dict | None:
    by_sample: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(value_key) is not None:
            by_sample[str(row["sample_id"])].append(float(row[value_key]))
    values = np.asarray([np.mean(item) for item in by_sample.values()])
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(replicates)])
    return {"n_examples": int(len(values)), "mean": round(float(values.mean()), 6), "ci95": [round(float(value), 6) for value in np.quantile(bootstrap, [0.025, 0.975])]} 


def _causal_bootstrap(root: Path, replicates: int, seed: int) -> dict:
    sources = [
        root / "qwen_causal_mediation/causal_mediation_results.jsonl",
        root / "multibully/causal_raw_ocr_v2/causal_mediation_results.jsonl",
        root / "exist_multilingual/causal_raw_ocr/causal_mediation_results.jsonl",
        root / "layer_self_patching_live_v1/fhm/results.jsonl",
        root / "layer_self_patching_live_v1/mami/results.jsonl",
        root / "layer_self_patching_v3/multibully/results.jsonl",
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in sources:
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            if "delta_gold_margin" in row:
                key = f"{path.parent.name}::{row.get('dataset', row.get('task_key', 'unknown'))}::{row.get('condition', row.get('intervention', 'unknown'))}"
            else:
                key = f"{path.parent.name}::{row.get('task_key', 'multibully')}::{row.get('intervention', 'knockout')}"
            grouped[key].append(row)
    output = {}
    for index, (key, rows) in enumerate(sorted(grouped.items())):
        effects = {}
        for value_key in ("delta_probe", "delta_logit", "delta_gold_margin"):
            result = _bootstrap_examples(rows, value_key, replicates, seed + index)
            if result is not None:
                effects[value_key] = result
        if effects:
            output[key] = effects
    return output


def _bilinear_seeds(sae: QwenSae, spec: TaskSpec, seeds: list[int], device: str) -> dict:
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    from .probe_fhm_bilinear import _train_bilinear
    from .probe_fhm_bilinear_base_sae import _extract_role_features_base_sae
    train_rows = [row for row in _read_jsonl(Path(spec.train_cache) / "manifest.jsonl") if row.get("dataset_name") == spec.dataset_name and row.get("task_name") == spec.task_name and row.get("split") == spec.train_split]
    eval_rows = [row for row in _read_jsonl(Path(spec.eval_cache) / "manifest.jsonl") if row.get("dataset_name") == spec.dataset_name and row.get("task_name") == spec.task_name and row.get("split") == spec.eval_split]
    image_train, prompt_train, y_train = _extract_role_features_base_sae(sae, Path(spec.train_cache), train_rows, None, None, device)
    image_eval, prompt_eval, y_eval = _extract_role_features_base_sae(sae, Path(spec.eval_cache), eval_rows, None, None, device)
    image_index = np.argsort(-image_train.var(axis=0))[:256]
    prompt_index = np.argsort(-prompt_train.var(axis=0))[:256]
    image_scale, prompt_scale = StandardScaler(), StandardScaler()
    image_train = image_scale.fit_transform(image_train[:, image_index]); image_eval = image_scale.transform(image_eval[:, image_index])
    prompt_train = prompt_scale.fit_transform(prompt_train[:, prompt_index]); prompt_eval = prompt_scale.transform(prompt_eval[:, prompt_index])
    result = {}
    for seed in seeds:
        torch.manual_seed(seed)
        model, internal_accuracy = _train_bilinear(image_train, prompt_train, y_train, 16, 200, 1e-3, 1e-3, 20, device, seed=seed)
        with torch.no_grad():
            logits = model(torch.tensor(image_eval, dtype=torch.float32, device=device), torch.tensor(prompt_eval, dtype=torch.float32, device=device)).cpu().numpy()
        prediction = (logits >= 0).astype(np.int32)
        result[str(seed)] = {"macro_f1": round(float(f1_score(y_eval, prediction, average="macro", zero_division=0)), 6), "internal_val_accuracy": round(float(internal_accuracy), 6)}
    return {"task": "FHM image_x_prompt bilinear", "seeds": result, "threshold": "fixed zero logit; no test threshold tuning", "internal_validation": "stratified 15% train split per seed; early stopping"}


def _plot_calibration(calibration: dict, output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(6, 5))
        for name, values in calibration.items():
            bins = values["bins"]
            axis.plot([item["mean_probability"] for item in bins], [item["empirical_positive_rate"] for item in bins], marker="o", label=name)
        axis.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
        axis.set(xlabel="Predicted positive probability", ylabel="Empirical positive rate", xlim=(0, 1), ylim=(0, 1))
        axis.legend(fontsize=7); figure.tight_layout(); figure.savefig(output, dpi=150); plt.close(figure)
    except ImportError:
        logger.warning("matplotlib unavailable; calibration figure omitted")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(StatisticalCloseoutConfig)
    output = Path(cfg.output_dir); output.mkdir(parents=True, exist_ok=True)
    baseline_rows, hook_rows = _read_jsonl(Path(cfg.baseline_jsonl)), _read_jsonl(Path(cfg.hook_jsonl))
    sae = QwenSae.load_from_hub(cfg.sae_repo_id, cfg.sae_layer, device=cfg.device)
    comparisons, seed_report, per_sample = {}, {}, []
    fhm_systems: dict[str, dict[str, tuple[int, int, float]]] = {}
    for task_index, spec in enumerate(_specs(cfg.activations_root)):
        base = _locked_predictions(baseline_rows, spec); hook = _locked_predictions(hook_rows, spec)
        common = sorted(set(base) & set(hook))
        probe_by_seed, report = _fit_probe_seeds(sae, spec, cfg.probe_seeds, cfg.device)
        seed_report[f"{spec.dataset_name}::{spec.task_name}"] = {"probe": report, "probe_seeds": cfg.probe_seeds}
        for seed, probe in probe_by_seed.items():
            common_seed = sorted(set(common) & set(probe))
            gold = np.asarray([base[sample_id][0] for sample_id in common_seed])
            base_pred = np.asarray([base[sample_id][1] for sample_id in common_seed])
            hook_pred = np.asarray([hook[sample_id][1] for sample_id in common_seed])
            probe_pred = np.asarray([probe[sample_id][1] for sample_id in common_seed])
            if seed == cfg.probe_seeds[0]:
                comparisons[f"{spec.dataset_name}::{spec.task_name}::hook_minus_base"] = {**_bootstrap_delta(gold, base_pred, hook_pred, cfg.bootstrap_replicates, cfg.seed + task_index), **_mcnemar(gold, base_pred, hook_pred), "n": len(gold)}
                comparisons[f"{spec.dataset_name}::{spec.task_name}::probe_minus_base"] = {**_bootstrap_delta(gold, base_pred, probe_pred, cfg.bootstrap_replicates, cfg.seed + 100 + task_index), **_mcnemar(gold, base_pred, probe_pred), "n": len(gold)}
                calibration = {"base": _calibration(gold, np.asarray([base[sample_id][2] for sample_id in common_seed])), "hook": _calibration(gold, np.asarray([hook[sample_id][2] for sample_id in common_seed])), "probe_seed42": _calibration(gold, np.asarray([probe[sample_id][2] for sample_id in common_seed]))}
                diagnostics = {
                    "calibration": calibration,
                    "classwise": {"base": _classwise(gold, base_pred), "hook": _classwise(gold, hook_pred), "probe_seed42": _classwise(gold, probe_pred)},
                }
                (output / f"calibration_{spec.dataset_name}_{spec.task_name}.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
                _plot_calibration(calibration, output / f"calibration_{spec.dataset_name}_{spec.task_name}.png")
                if spec.dataset_name == "Hateful_Memes_Dataset":
                    fhm_systems = {"base": {sample_id: base[sample_id] for sample_id in common_seed}, "hook": {sample_id: hook[sample_id] for sample_id in common_seed}, "probe_seed42": {sample_id: probe[sample_id] for sample_id in common_seed}}
            for sample_id in common_seed:
                per_sample.append({"dataset_name": spec.dataset_name, "task_name": spec.task_name, "sample_id": sample_id, "seed": seed, "gold": base[sample_id][0], "base_prediction": base[sample_id][1], "base_score": base[sample_id][2], "hook_prediction": hook[sample_id][1], "hook_score": hook[sample_id][2], "probe_prediction": probe[sample_id][1], "probe_score": probe[sample_id][2]})
        del probe_by_seed; gc.collect(); torch.cuda.empty_cache()
    fhm_pairs = {source: {system: _fhm_pairs(predictions, Path(cfg.fhm_metadata_path), source, cfg.bootstrap_replicates, cfg.seed) for system, predictions in fhm_systems.items()} for source in ("pseudo_img", "pseudo_text")}
    bilinear = _bilinear_seeds(sae, next(spec for spec in _specs(cfg.activations_root) if spec.dataset_name == "Hateful_Memes_Dataset"), cfg.bilinear_seeds, cfg.device)
    causal = _causal_bootstrap(Path(cfg.activations_root), cfg.causal_bootstrap_replicates, cfg.seed)
    protocol = {
        "probe_seeds": cfg.probe_seeds, "bilinear_seeds": cfg.bilinear_seeds,
        "locked_base_hook_threshold": "argmax constrained decode; no held-out threshold tuning",
        "probe_threshold": "fixed zero logistic decision score; no held-out threshold tuning",
        "router_calibration": "Not a locked-table system. Existing router analyses choose alpha and z-score moments on disjoint stratified calibration partitions, documented in their output summaries.",
        "multibully": "Single deterministic, label-stratified 1,000-example holdout (seed=20260803); reported as a supporting single-split result, not robust cross-model Hindi replication.",
    }
    with (output / "per_sample_locked_predictions.jsonl").open("w") as fd:
        for row in per_sample: fd.write(json.dumps(row) + "\n")
    (output / "paired_tests_and_bootstrap.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    (output / "seed_report.json").write_text(json.dumps({"probe": seed_report, "bilinear": bilinear, "protocol": protocol}, indent=2) + "\n")
    (output / "fhm_pair_bootstrap.json").write_text(json.dumps(fhm_pairs, indent=2) + "\n")
    (output / "causal_example_bootstrap.json").write_text(json.dumps(causal, indent=2) + "\n")
    logger.info("Wrote statistical close-out outputs to %s", output)


if __name__ == "__main__":
    main()