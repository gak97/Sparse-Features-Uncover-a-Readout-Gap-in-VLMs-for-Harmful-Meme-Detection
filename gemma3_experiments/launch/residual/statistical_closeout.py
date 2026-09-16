

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from simple_parsing import ArgumentGenerationMode, parse

ACT = Path("./outputs")
FHM_TASK = "Hateful_Memes_Dataset::hateful_memes_binary::label"


@dataclass
class StatisticalCloseoutConfig:
    output_dir: str = str(ACT / "residual_sae_statistical_closeout")
    n_bootstrap: int = 10_000
    n_bins: int = 10
    seed: int = 42

    fhm_12b_base_predictions_path: str = str(
        ACT / "residual_sae_predictions.12b_gemma.fhm_mami_mmhs.test.label_score.jsonl"
    )
    fhm_bilinear_test_predictions_path: str = str(
        ACT / "residual_sae_fhm_bilinear_r32_test" / "bilinear_test_predictions.jsonl"
    )
    fhm_bilinear_test_metrics_path: str = str(
        ACT / "residual_sae_fhm_bilinear_r32_test" / "bilinear_test_metrics.json"
    )
    fhm_bilinear_seed_root: str = str(ACT)
    generated_probe_predictions_path: str = str(ACT / "residual_sae_probe_test_generated" / "probe_val_predictions.jsonl")
    generated_probe_metrics_path: str = str(ACT / "residual_sae_probe_test_generated" / "probe_val_metrics.json")
    image_probe_predictions_path: str = str(ACT / "residual_sae_probe_test_image" / "probe_val_predictions.jsonl")
    image_probe_metrics_path: str = str(ACT / "residual_sae_probe_test_image" / "probe_val_metrics.json")
    fhm_self_patching_path: str = str(
        ACT / "fhm_layer_self_patching_12b_l31_to_l41_test_seen" / "per_sample.jsonl"
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as fd:
        return [json.loads(line) for line in fd if line.strip()]


def _macro_f1(gold: np.ndarray, pred: np.ndarray) -> float:
    labels = np.unique(np.concatenate([gold, pred]))
    f1_values = []
    for label in labels:
        true_positive = np.sum((gold == label) & (pred == label))
        false_positive = np.sum((gold != label) & (pred == label))
        false_negative = np.sum((gold == label) & (pred != label))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1_values))


def _class_metrics(gold: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    metrics = {}
    for label in np.unique(np.concatenate([gold, pred])):
        true_positive = int(np.sum((gold == label) & (pred == label)))
        false_positive = int(np.sum((gold != label) & (pred == label)))
        false_negative = int(np.sum((gold == label) & (pred != label)))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        metrics[str(label)] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(2 * precision * recall / (precision + recall) if precision + recall else 0.0, 6),
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
        }
    return metrics


def _stratified_indices(gold: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    indices = []
    for label in np.unique(gold):
        label_indices = np.flatnonzero(gold == label)
        indices.append(rng.choice(label_indices, size=len(label_indices), replace=True))
    return np.concatenate(indices)


def _paired_f1_bootstrap(
    gold: np.ndarray, candidate: np.ndarray, baseline: np.ndarray, n_bootstrap: int, rng: np.random.Generator,
) -> dict[str, Any]:
    differences = np.empty(n_bootstrap, dtype=np.float64)
    for bootstrap_i in range(n_bootstrap):
        indices = _stratified_indices(gold, rng)
        differences[bootstrap_i] = _macro_f1(gold[indices], candidate[indices]) - _macro_f1(gold[indices], baseline[indices])
    return {
        "candidate_minus_baseline": round(_macro_f1(gold, candidate) - _macro_f1(gold, baseline), 6),
        "ci_95_percentile": [round(float(value), 6) for value in np.quantile(differences, [0.025, 0.975])],
        "n_bootstrap": n_bootstrap,
        "resampling": "paired, label-stratified examples",
    }


def _mcnemar_exact(candidate_correct: np.ndarray, baseline_correct: np.ndarray) -> dict[str, Any]:
    candidate_only = int(np.sum(candidate_correct & ~baseline_correct))
    baseline_only = int(np.sum(~candidate_correct & baseline_correct))
    discordant = candidate_only + baseline_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower_tail = sum(math.comb(discordant, value) for value in range(min(candidate_only, baseline_only) + 1))
        p_value = min(1.0, 2.0 * lower_tail / (2**discordant))
    return {
        "test": "two-sided exact McNemar test on paired correctness",
        "candidate_only_correct": candidate_only,
        "baseline_only_correct": baseline_only,
        "discordant_pairs": discordant,
        "p_value": p_value,
    }


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    labels = list(scores)
    values = np.array([scores[label] for label in labels], dtype=np.float64)
    values -= values.max()
    probabilities = np.exp(values)
    probabilities /= probabilities.sum()
    return {label: float(probability) for label, probability in zip(labels, probabilities)}


def _calibration_curve(gold: np.ndarray, probabilities: np.ndarray, n_bins: int) -> dict[str, Any]:
    bins = []
    for bin_i in range(n_bins):
        lower = bin_i / n_bins
        upper = (bin_i + 1) / n_bins
        mask = (probabilities >= lower) & ((probabilities < upper) if bin_i < n_bins - 1 else (probabilities <= upper))
        if not mask.any():
            continue
        bins.append({
            "lower": lower,
            "upper": upper,
            "n": int(mask.sum()),
            "mean_prediction": round(float(probabilities[mask].mean()), 6),
            "observed_frequency": round(float(gold[mask].mean()), 6),
        })
    brier = float(np.mean((probabilities - gold) ** 2))
    ece = sum(bin_row["n"] / len(gold) * abs(bin_row["mean_prediction"] - bin_row["observed_frequency"]) for bin_row in bins)
    return {"n": len(gold), "brier": round(brier, 6), "ece_equal_width": round(ece, 6), "bins": bins}


def _base_fhm_rows(path: str) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in _read_jsonl(path):
        if row.get("dataset_name") != "Hateful_Memes_Dataset" or row.get("task_name") != "hateful_memes_binary":
            continue
        sample_id = str(row["sample_id"])
        scores = row.get("label_scores")
        if not isinstance(scores, dict):
            continue
        rows[sample_id] = {
            "gold": row.get("gold_fields", {}).get("label"),
            "pred": row.get("prediction_label"),
            "probabilities": _softmax({label: float(score) for label, score in scores.items()}),
        }
    return rows


def _bilinear_fhm_rows(path: str) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in _read_jsonl(path):
        sample_id = str(row["sample_key"]).split("::")[-1]
        hateful_probability = float(row["probe_score"])
        rows[sample_id] = {
            "gold": row["gold_label"],
            "pred": row["probe_pred"],
            "probabilities": {"hateful": hateful_probability, "not hateful": 1.0 - hateful_probability},
        }
    return rows


def _paired_fhm_report(cfg: StatisticalCloseoutConfig, rng: np.random.Generator) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    base = _base_fhm_rows(cfg.fhm_12b_base_predictions_path)
    bilinear = _bilinear_fhm_rows(cfg.fhm_bilinear_test_predictions_path)
    common_ids = sorted(set(base) & set(bilinear))
    assert common_ids, "No shared FHM IDs between locked 12B base and bilinear predictions."
    assert all(base[sample_id]["gold"] == bilinear[sample_id]["gold"] for sample_id in common_ids)
    gold = np.array([base[sample_id]["gold"] for sample_id in common_ids])
    base_pred = np.array([base[sample_id]["pred"] for sample_id in common_ids])
    bilinear_pred = np.array([bilinear[sample_id]["pred"] for sample_id in common_ids])
    report = {
        "n_paired": len(common_ids),
        "candidate": "12B FHM bilinear r=32 seed 0",
        "baseline": "12B native LVLM label_score",
        "candidate_macro_f1": round(_macro_f1(gold, bilinear_pred), 6),
        "baseline_macro_f1": round(_macro_f1(gold, base_pred), 6),
        "paired_macro_f1": _paired_f1_bootstrap(gold, bilinear_pred, base_pred, cfg.n_bootstrap, rng),
        "paired_correctness": _mcnemar_exact(bilinear_pred == gold, base_pred == gold),
        "class_wise_precision_recall": {
            "candidate": _class_metrics(gold, bilinear_pred),
            "baseline": _class_metrics(gold, base_pred),
        },
    }
    calibration_rows = {
        "12b_native_fhm": {
            "gold": gold,
            "probabilities": np.array([base[sample_id]["probabilities"]["hateful"] for sample_id in common_ids]),
        },
        "12b_bilinear_fhm_r32_seed0": {
            "gold": gold,
            "probabilities": np.array([bilinear[sample_id]["probabilities"]["hateful"] for sample_id in common_ids]),
        },
    }
    return report, calibration_rows


def _seed_closeout(cfg: StatisticalCloseoutConfig, rng: np.random.Generator) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for rank in (8, 32):
        seed_rows = []
        for seed in range(5):
            directory = Path(cfg.fhm_bilinear_seed_root) / f"residual_sae_probe_validation_fhm_bilinear_r{rank}_seed{seed}"
            metrics = _read_json(directory / "pairwise_probe_metrics.json")
            if not metrics:
                seed_rows.append({"seed": seed, "missing": True})
                continue
            calibrated = metrics.get("calibrated_metrics") or {}
            row: dict[str, Any] = {
                "seed": seed,
                "training_seed": (metrics.get("config") or {}).get("training_seed"),
                "calibration_seed": (metrics.get("config") or {}).get("calibration_seed"),
                "calibration_fraction": (metrics.get("config") or {}).get("calibration_frac"),
                "n_calibration": metrics.get("n_val_calib"),
                "n_held_out_validation": metrics.get("n_val_held"),
                "threshold": calibrated.get("threshold"),
                "held_out_macro_f1": calibrated.get("macro_f1"),
                "raw_macro_f1": (metrics.get("raw_metrics") or {}).get("macro_f1"),
                "pair_bootstrap": {},
            }
            by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for pair_row in _read_jsonl(directory / "validation_pair_metrics.jsonl"):
                by_source[str(pair_row["source"])].append(pair_row)
            for source, pairs in by_source.items():
                row["pair_bootstrap"][source] = {}
                for metric in ("both_correct", "positive_margin", "prob_gap"):
                    values = np.array([float(pair[metric]) for pair in pairs], dtype=np.float64)
                    estimates = np.empty(cfg.n_bootstrap, dtype=np.float64)
                    for bootstrap_i in range(cfg.n_bootstrap):
                        estimates[bootstrap_i] = float(rng.choice(values, size=len(values), replace=True).mean())
                    row["pair_bootstrap"][source][metric] = {
                        "n_pairs": len(values),
                        "estimate": round(float(values.mean()), 6),
                        "ci_95_percentile": [round(float(value), 6) for value in np.quantile(estimates, [0.025, 0.975])],
                        "resampling": "paired pseudo-pairs within source",
                    }
            seed_rows.append(row)
        available = [row["held_out_macro_f1"] for row in seed_rows if row.get("held_out_macro_f1") is not None]
        result[f"rank_{rank}"] = {
            "selection_protocol": "threshold selected on the fixed 20% calibration subset; F1 reported on the disjoint 80% held-out validation subset",
            "seeds": seed_rows,
            "held_out_macro_f1_mean": round(float(np.mean(available)), 6) if available else None,
            "held_out_macro_f1_sample_sd": round(float(np.std(available, ddof=1)), 6) if len(available) > 1 else None,
        }
    return result


def _probe_seed_report() -> dict[str, Any]:
    """Enumerate every completed non-bilinear probe robustness split.

    These seed values choose deterministic held-out partitions, rather than
    classifier initializations. Keep them separate from the FHM bilinear
    training-seed sweep above.
    """
    seed_values = (13, 29, 42, 73, 101)
    families = {
        "multibully_all_tokens": "residual_sae_probe_multibully_all",
        "exist_en_to_es_all_tokens": "residual_sae_probe_exist_en_to_es_all",
        "exist_es_to_en_generated": "residual_sae_probe_exist_es_to_en_generated",
        "exist_bilingual_generated": "residual_sae_probe_exist_bilingual_generated",
    }
    report: dict[str, Any] = {}
    for name, stem in families.items():
        rows = []
        for seed in seed_values:
            directory = ACT / (stem if seed == 42 else f"{stem}_seed{seed}")
            metrics = _read_json(directory / "probe_val_metrics.json")
            if not metrics:
                rows.append({"split_seed": seed, "missing": True, "path": str(directory)})
                continue
            task_key, task_result = next(iter(metrics.items()))
            metric = task_result.get("probe_metrics") or task_result.get("stage1_binary_metrics") or {}
            rows.append({
                "split_seed": seed,
                "task_key": task_key,
                "n": metric.get("n"),
                "macro_f1": metric.get("macro_f1"),
                "path": str(directory),
            })
        f1_values = [row["macro_f1"] for row in rows if row.get("macro_f1") is not None]
        report[name] = {
            "seed_kind": "deterministic label-stratified held-out split seed",
            "seeds": rows,
            "macro_f1_mean": round(float(np.mean(f1_values)), 6) if f1_values else None,
            "macro_f1_sample_sd": round(float(np.std(f1_values, ddof=1)), 6) if len(f1_values) > 1 else None,
        }
    return report


def _probe_calibration_and_class_metrics(cfg: StatisticalCloseoutConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration: dict[str, Any] = {}
    class_metrics: dict[str, Any] = {}
    for source, path in {
        "generated_probe": cfg.generated_probe_predictions_path,
        "image_probe": cfg.image_probe_predictions_path,
    }.items():
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in _read_jsonl(path):
            if isinstance(row.get("probe_proba"), dict) and isinstance(row.get("gold_label"), str):
                by_task[str(row["task_key"])].append(row)
        for task_key, rows in by_task.items():
            gold = np.array([row["gold_label"] for row in rows])
            pred = np.array([row["probe_pred"] for row in rows])
            key = f"{source}::{task_key}"
            class_metrics[key] = {"n": len(rows), "macro_f1": round(_macro_f1(gold, pred), 6), "per_class": _class_metrics(gold, pred)}
            calibration[key] = {}
            for label in sorted({label for row in rows for label in row["probe_proba"]}):
                probabilities = np.array([float(row["probe_proba"].get(label, 0.0)) for row in rows])
                calibration[key][label] = _calibration_curve((gold == label).astype(float), probabilities, cfg.n_bins)
    return calibration, class_metrics


def _causal_bootstrap(cfg: StatisticalCloseoutConfig, rng: np.random.Generator) -> dict[str, Any]:
    rows = _read_jsonl(cfg.fhm_self_patching_path)
    if not rows:
        return {"missing": True, "path": cfg.fhm_self_patching_path}
    condition_names = rows[0]["margin_delta_from_baseline"].keys()
    result = {
        "n_examples": len(rows),
        "resampling_unit": "examples; feature interventions are not treated as independent observations",
        "conditions": {},
    }
    for condition in condition_names:
        values = np.array([float(row["margin_delta_from_baseline"][condition]) for row in rows])
        estimates = np.empty(cfg.n_bootstrap, dtype=np.float64)
        for bootstrap_i in range(cfg.n_bootstrap):
            estimates[bootstrap_i] = float(rng.choice(values, size=len(values), replace=True).mean())
        result["conditions"][condition] = {
            "mean_margin_delta": round(float(values.mean()), 6),
            "ci_95_percentile": [round(float(value), 6) for value in np.quantile(estimates, [0.025, 0.975])],
        }
    superadditivity = np.array([float(row["joint_superadditivity"]) for row in rows])
    estimates = np.empty(cfg.n_bootstrap, dtype=np.float64)
    for bootstrap_i in range(cfg.n_bootstrap):
        estimates[bootstrap_i] = float(rng.choice(superadditivity, size=len(superadditivity), replace=True).mean())
    result["joint_superadditivity"] = {
        "mean": round(float(superadditivity.mean()), 6),
        "ci_95_percentile": [round(float(value), 6) for value in np.quantile(estimates, [0.025, 0.975])],
    }
    return result


def _write_fhm_calibration_plot(curves: dict[str, Any], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axis = plt.subplots(figsize=(5.5, 5.0))
    axis.plot([0, 1], [0, 1], color="0.5", linestyle="--", label="perfect calibration")
    for name, curve in curves.items():
        bins = curve["hateful"]["bins"]
        axis.plot([row["mean_prediction"] for row in bins], [row["observed_frequency"] for row in bins], marker="o", label=name)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted P(hateful)", ylabel="Observed hateful frequency")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(cfg: StatisticalCloseoutConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    fhm_comparison, fhm_calibration_rows = _paired_fhm_report(cfg, rng)
    seed_report = _seed_closeout(cfg, rng)
    probe_calibration, probe_class_metrics = _probe_calibration_and_class_metrics(cfg)
    calibration = {"fhm_locked_models": {}}
    for name, rows in fhm_calibration_rows.items():
        calibration["fhm_locked_models"][name] = {"hateful": _calibration_curve((rows["gold"] == "hateful").astype(float), rows["probabilities"], cfg.n_bins)}
    calibration["probe_test_artifacts"] = probe_calibration
    report = {
        "protocol": {
            "locked_results_only": True,
            "n_bootstrap": cfg.n_bootstrap,
            "bootstrap_ci": "two-sided 95% percentile confidence interval",
            "f1_difference_resampling": "paired stratified bootstrap over examples",
            "paired_test": "two-sided exact McNemar test on paired correctness",
        },
        "fhm_locked_comparison": fhm_comparison,
        "bilinear_seed_report": seed_report,
        "probe_robustness_seed_report": _probe_seed_report(),
        "threshold_selection": {
            "fhm_bilinear_test": _read_json(cfg.fhm_bilinear_test_metrics_path),
            "note": "The FHM test threshold is loaded from the selected seed-0 validation calibration subset; it is not tuned on test_seen.",
        },
        "class_wise_precision_recall": {"fhm_locked_comparison": fhm_comparison["class_wise_precision_recall"], "probe_test_artifacts": probe_class_metrics},
        "calibration_curves": calibration,
        "fhm_pair_bootstrap": {key: value["seeds"] for key, value in seed_report.items()},
        "causal_example_bootstrap": _causal_bootstrap(cfg, rng),
    }
    (out_dir / "statistical_closeout.json").write_text(json.dumps(report, indent=2) + "\n")
    _write_fhm_calibration_plot(calibration["fhm_locked_models"], out_dir / "fhm_calibration_curve.png")
    (out_dir / "README.txt").write_text(
        "Locked statistical close-out. FHM CI/test compare matched 12B base vs seed-0 bilinear test_seen rows. "
        "Bilinear seed and pair bootstraps use validation artifacts; causal bootstraps resample examples from the self-patching run.\n"
    )
    print(json.dumps({
        "output": str(out_dir),
        "fhm_delta": fhm_comparison["paired_macro_f1"],
        "fhm_mcnemar": fhm_comparison["paired_correctness"],
        "seed_means": {key: value["held_out_macro_f1_mean"] for key, value in seed_report.items()},
    }, indent=2))


def main() -> None:
    run(parse(StatisticalCloseoutConfig, argument_generation_mode=ArgumentGenerationMode.BOTH))


if __name__ == "__main__":
    main()