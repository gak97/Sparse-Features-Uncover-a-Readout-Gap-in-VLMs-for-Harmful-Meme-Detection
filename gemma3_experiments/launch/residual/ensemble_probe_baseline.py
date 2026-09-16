

import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from simple_parsing import ArgumentGenerationMode, parse

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    probe_predictions_path: str = ""   # probe_val_predictions.jsonl from probe_validate.py
    baseline_predictions_path: str = ""  # original baseline jsonl
    output_dir: str = ""

    # Grid of probe weights to sweep. Best weight selected per task by macro F1.
    probe_weights: list[float] = field(default_factory=lambda: [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    ])

    # Single fixed weight to use for ensemble_predictions.jsonl output (in addition
    # to per-task best weight). Set to -1 to skip.
    fixed_weight: float = 0.5

    plot: bool = True


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    """Softmax over label_scores dict (log-prob or raw logits)."""
    vals = np.array(list(scores.values()), dtype=float)
    vals = vals - vals.max()
    exps = np.exp(vals)
    exps /= exps.sum()
    return {k: float(v) for k, v in zip(scores.keys(), exps)}


def _per_class_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    per_class: dict[str, Any] = {}
    macro_f1 = 0.0
    for lbl in labels:
        tp = sum(1 for g, p in zip(y_true, y_pred) if g == lbl and p == lbl)
        fp = sum(1 for g, p in zip(y_true, y_pred) if g != lbl and p == lbl)
        fn = sum(1 for g, p in zip(y_true, y_pred) if g == lbl and p != lbl)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per_class[lbl] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}
        macro_f1 += f
    n = len(labels)
    correct = sum(1 for g, p in zip(y_true, y_pred) if g == p)
    return {
        "n": len(y_true),
        "accuracy": round(correct / len(y_true), 4) if y_true else 0.0,
        "macro_f1": round(macro_f1 / n, 4) if n else 0.0,
        "per_class": per_class,
        "gold_counts": dict(Counter(y_true).most_common()),
    }


def run_ensemble(cfg: EnsembleConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_path = Path(cfg.probe_predictions_path)
    baseline_path = Path(cfg.baseline_predictions_path)
    assert probe_path.exists(), f"Probe predictions not found: {probe_path}"
    assert baseline_path.exists(), f"Baseline predictions not found: {baseline_path}"

    # ── Load probe predictions ────────────────────────────────────────────────
    logger.info("Loading probe predictions from %s", probe_path)
    probe_rows = []
    with probe_path.open() as fd:
        for line in fd:
            line = line.strip()
            if line:
                probe_rows.append(json.loads(line))
    logger.info("Probe predictions: %d samples", len(probe_rows))

    # ── Load baseline predictions ─────────────────────────────────────────────
    logger.info("Loading baseline predictions from %s", baseline_path)
    baseline_by_key: dict[str, dict[str, Any]] = {}
    with baseline_path.open() as fd:
        for line in fd:
            line = line.strip()
            if not line:
                continue
            br = json.loads(line)
            key = f"{br['dataset_name']}::{br['task_name']}::{br['sample_id']}"
            baseline_by_key[key] = br
    logger.info("Baseline predictions: %d entries", len(baseline_by_key))

    # ── Group probe rows by task ──────────────────────────────────────────────
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in probe_rows:
        by_task[row["task_key"]].append(row)

    all_metrics: dict[str, Any] = {}
    best_ensemble_preds: list[dict[str, Any]] = []

    for task_key in sorted(by_task.keys()):
        rows = by_task[task_key]

        # Filter rows that have both probe_proba and baseline label_scores.
        # For yes_no decode strategy (MAMI, Hateful_Memes_Dataset, MultiOFF), label_scores is
        # None but is_positive + conf_gap are available. Construct synthetic scores from these.
        valid = []
        for row in rows:
            proba = row.get("probe_proba")
            sk = row["sample_key"]
            br = baseline_by_key.get(sk)
            base_scores = row.get("baseline_label_scores") or (br.get("label_scores") if br else None)

            if base_scores is None and br is not None and br.get("decode_strategy") == "yes_no":
                schema = br.get("schema") or {}
                primary_labels = (schema.get("primary_field") or {}).get("labels") or []
                if len(primary_labels) == 2:
                    pos_label, neg_label = primary_labels[-1], primary_labels[0]
                    # Use a small epsilon floor so that gap=0.0 still breaks ties correctly
                    # (without it, -0.0 == 0.0 in floating point → equal softmax → alphabetical
                    # tie-break picks wrong class for is_positive=False samples).
                    _EPS = 1e-9
                    gap = max(float(br.get("conf_gap") or _EPS), _EPS)
                    if br.get("is_positive"):
                        base_scores = {pos_label: gap, neg_label: -gap}
                    else:
                        base_scores = {pos_label: -gap, neg_label: gap}

            if proba and base_scores:
                valid.append((row, proba, base_scores))

        if len(valid) < 10:
            logger.warning("Task %s: only %d samples with both probe_proba and baseline scores — skipping", task_key, len(valid))
            continue

        gold_labels = [r["gold_label"] for r, _, _ in valid]
        probe_classes = sorted(set(gold_labels) | set(r["probe_pred"] for r, _, _ in valid))

        logger.info("Task %s: %d samples, classes=%s", task_key, len(valid), probe_classes)

        # ── Weight sweep ─────────────────────────────────────────────────────
        sweep_results: list[dict[str, Any]] = []
        for w in cfg.probe_weights:
            preds = []
            for row, proba, base_scores in valid:
                base_proba = _softmax(base_scores)
                # Align class sets: use probe's known classes as reference
                combined: dict[str, float] = {}
                for cls in probe_classes:
                    p_prob = proba.get(cls, 0.0)
                    b_prob = base_proba.get(cls, 0.0)
                    combined[cls] = w * p_prob + (1 - w) * b_prob
                preds.append(max(combined, key=combined.__getitem__))
            m = _per_class_metrics(gold_labels, preds)
            sweep_results.append({"probe_weight": w, "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]})

        best = max(sweep_results, key=lambda x: x["macro_f1"])
        logger.info("Task %s: best weight=%.1f  macro_f1=%.4f  (probe_only=%.4f, baseline_only=%.4f)",
                    task_key, best["probe_weight"], best["macro_f1"],
                    sweep_results[-1]["macro_f1"],  # w=1.0 = probe only
                    sweep_results[0]["macro_f1"])   # w=0.0 = baseline only

        # ── Metrics at best weight ────────────────────────────────────────────
        best_preds = []
        for row, proba, base_scores in valid:
            base_proba = _softmax(base_scores)
            combined = {
                cls: best["probe_weight"] * proba.get(cls, 0.0) + (1 - best["probe_weight"]) * base_proba.get(cls, 0.0)
                for cls in probe_classes
            }
            best_preds.append(max(combined, key=combined.__getitem__))
        best_metrics = _per_class_metrics(gold_labels, best_preds)

        # ── Metrics at fixed weight ───────────────────────────────────────────
        fixed_metrics = None
        fixed_preds_for_task = []
        if 0.0 <= cfg.fixed_weight <= 1.0:
            fixed_preds_for_task = []
            for row, proba, base_scores in valid:
                base_proba = _softmax(base_scores)
                combined = {
                    cls: cfg.fixed_weight * proba.get(cls, 0.0) + (1 - cfg.fixed_weight) * base_proba.get(cls, 0.0)
                    for cls in probe_classes
                }
                fixed_preds_for_task.append(max(combined, key=combined.__getitem__))
            fixed_metrics = _per_class_metrics(gold_labels, fixed_preds_for_task)

        all_metrics[task_key] = {
            "task_key": task_key,
            "n": len(valid),
            "classes": probe_classes,
            "probe_only_f1": sweep_results[-1]["macro_f1"],
            "baseline_only_f1": sweep_results[0]["macro_f1"],
            "best_weight": best["probe_weight"],
            "best_ensemble_f1": best["macro_f1"],
            "fixed_weight": cfg.fixed_weight,
            "fixed_ensemble_f1": fixed_metrics["macro_f1"] if fixed_metrics else None,
            "sweep": sweep_results,
            "best_metrics": best_metrics,
            "fixed_metrics": fixed_metrics,
        }

        # Collect per-sample ensemble predictions (at best weight)
        for (row, proba, base_scores), pred in zip(valid, best_preds):
            best_ensemble_preds.append({
                "sample_key": row["sample_key"],
                "task_key": task_key,
                "gold_label": row["gold_label"],
                "probe_pred": row["probe_pred"],
                "baseline_pred": row.get("baseline_pred"),
                "ensemble_pred": pred,
                "ensemble_weight": best["probe_weight"],
                "ensemble_correct": pred == row["gold_label"],
                "probe_correct": row["probe_pred"] == row["gold_label"],
                "baseline_correct": row.get("baseline_correct"),
            })

    # ── Write outputs ─────────────────────────────────────────────────────────
    (output_dir / "ensemble_metrics.json").write_text(json.dumps(all_metrics, indent=2))
    logger.info("Ensemble metrics written to %s/ensemble_metrics.json", output_dir)

    with (output_dir / "ensemble_predictions.jsonl").open("w") as fd:
        for row in best_ensemble_preds:
            fd.write(json.dumps(row) + "\n")
    logger.info("Ensemble predictions written to %s/ensemble_predictions.jsonl", output_dir)

    # ── Summary table ─────────────────────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("%-50s  %6s  %6s  %6s  %5s", "Task", "Probe", "Base", "Ensemble", "Wt")
    logger.info("-" * 80)
    for task_key, r in sorted(all_metrics.items(), key=lambda x: -x[1]["best_ensemble_f1"]):
        logger.info("%-50s  %.4f  %.4f  %.4f   %.1f",
                    task_key[:50],
                    r["probe_only_f1"], r["baseline_only_f1"],
                    r["best_ensemble_f1"], r["best_weight"])
    logger.info("=" * 80)

    if cfg.plot:
        _plot_ensemble(all_metrics, output_dir)


def _plot_ensemble(all_metrics: dict[str, Any], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")
        return

    # ── 1. Weight sweep: macro F1 vs probe_weight per task ───────────────────
    n_tasks = len(all_metrics)
    if n_tasks == 0:
        return

    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 4), squeeze=False)
    for ax, (task_key, r) in zip(axes[0], sorted(all_metrics.items())):
        weights = [s["probe_weight"] for s in r["sweep"]]
        f1s = [s["macro_f1"] for s in r["sweep"]]
        ax.plot(weights, f1s, "o-", color="steelblue", linewidth=1.5)
        ax.axhline(r["probe_only_f1"], color="steelblue", linestyle="--", alpha=0.5, label=f"Probe {r['probe_only_f1']:.3f}")
        ax.axhline(r["baseline_only_f1"], color="darkorange", linestyle="--", alpha=0.5, label=f"Base {r['baseline_only_f1']:.3f}")
        ax.axvline(r["best_weight"], color="green", linestyle=":", alpha=0.7, label=f"Best w={r['best_weight']:.1f}")
        ax.set_xlabel("Probe weight")
        ax.set_ylabel("Macro F1")
        short = task_key.split("::")[0].replace("_Dataset", "") + "\n" + task_key.split("::")[-1][:20]
        ax.set_title(short, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 1.0)
    fig.suptitle("Ensemble Weight Sweep: Macro F1 vs Probe Weight", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "ensemble_weight_sweep.png", dpi=150)
    plt.close(fig)
    logger.info("Saved ensemble_weight_sweep.png")

    # ── 2. Bar chart: ensemble vs probe vs baseline per task ─────────────────
    tasks_sorted = sorted(all_metrics.items(), key=lambda x: -x[1]["best_ensemble_f1"])
    labels = [t.split("::")[0].replace("_Dataset", "") + "\n" + t.split("::")[-1][:15]
              for t, _ in tasks_sorted]
    probe_f1s = [r["probe_only_f1"] for _, r in tasks_sorted]
    base_f1s = [r["baseline_only_f1"] for _, r in tasks_sorted]
    ens_f1s = [r["best_ensemble_f1"] for _, r in tasks_sorted]
    best_ws = [r["best_weight"] for _, r in tasks_sorted]

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 2.2), 5))
    b1 = ax.bar(x - width, probe_f1s, width, label="SAE Probe", color="steelblue")
    b2 = ax.bar(x, base_f1s, width, label="Baseline", color="darkorange")
    b3 = ax.bar(x + width, ens_f1s, width, label="Ensemble (best weight)", color="forestgreen")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7)
    for bar, val, w in zip(b3, ens_f1s, best_ws):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}\n(w={w:.1f})", ha="center", va="bottom", fontsize=7, color="forestgreen")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("Ensemble vs Probe vs Baseline: Macro F1 on Validation Set")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "ensemble_vs_components.png", dpi=150)
    plt.close(fig)
    logger.info("Saved ensemble_vs_components.png")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse(
        EnsembleConfig,
        argument_generation_mode=ArgumentGenerationMode.FLAT,
    )
    run_ensemble(cfg)


if __name__ == "__main__":
    main()
