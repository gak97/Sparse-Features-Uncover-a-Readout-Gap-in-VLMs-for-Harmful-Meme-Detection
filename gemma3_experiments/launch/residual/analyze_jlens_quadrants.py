

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simple_parsing import ArgumentGenerationMode, parse

logger = logging.getLogger(__name__)


@dataclass
class JlensQuadrantConfig:
    jlens_per_sample_path: str = ""     # jlens_per_sample.jsonl from compute_jacobian_lens.py
    probe_predictions_path: str = ""    # probe_val_predictions.jsonl
    task_key: str = ""                  # e.g. "MAMI::misogyny_and_subtypes::label"
    layer_idx: int = 22

    # Optional override for base-model correctness. Some probe_val_predictions.jsonl
    # files have a null baseline_correct (e.g. residual_sae_probe_crisis_heldout_image,
    # which was run without --baseline_predictions_path). When set, base correctness
    # is instead computed from a raw joint_infer.py-style predictions JSONL, matched
    # by sample_id, using base_gold_field_name/base_dataset_name/base_task_name.
    base_predictions_path: str = ""
    base_gold_field_name: str = ""      # defaults to matching gold_field logic if empty
    base_dataset_name: str = ""
    base_task_name: str = ""

    positive_tokens: list[str] = field(default_factory=list)   # e.g. ["misogyny"]
    negative_tokens: list[str] = field(default_factory=list)   # e.g. ["safe", "not"]
    positive_label: str = ""            # gold label string treated as "positive"
    negative_label: str = ""            # gold label string treated as "negative"
    # For multi-class gold (e.g. HarMeme's 3-class harmfulness), any gold label NOT
    # equal to negative_label is treated as positive (binary collapse), mirroring
    # build_final_eval_table.py's _collapse_binary convention.
    collapse_multiclass: bool = True

    calibration_fraction: float = 0.2   # fraction of samples used ONLY to pick the threshold
    calibration_seed: int = 42

    output_dir: str = ""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _collapse(label: str, positive_label: str, negative_label: str, collapse_multiclass: bool) -> str:
    if not collapse_multiclass:
        return label
    return negative_label if label == negative_label else positive_label


def _load_base_correctness(
    path: str, dataset_name: str, task_name: str, gold_field_name: str,
) -> dict[str, bool]:
    """Match base-LVLM correctness from a raw joint_infer.py-style predictions JSONL.

    Used when probe_val_predictions.jsonl has a null baseline_correct (e.g. the
    Crisis_A held-out probe run, which was executed without
    --baseline_predictions_path). Returns {sample_id: is_correct}.
    """
    result: dict[str, bool] = {}
    for row in _read_jsonl(Path(path)):
        if row.get("dataset_name") != dataset_name or row.get("task_name") != task_name:
            continue
        gold = (row.get("gold_fields") or {}).get(gold_field_name)
        pred = row.get("prediction_label")
        if gold is None or pred is None:
            continue
        result[str(row.get("sample_id"))] = (str(gold) == str(pred))
    return result


def run(cfg: JlensQuadrantConfig) -> None:
    import random

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert cfg.positive_tokens and cfg.negative_tokens, "Need positive_tokens and negative_tokens"
    assert cfg.positive_label and cfg.negative_label, "Need positive_label and negative_label"

    jlens_rows = _read_jsonl(Path(cfg.jlens_per_sample_path))
    probe_rows = [r for r in _read_jsonl(Path(cfg.probe_predictions_path)) if r.get("task_key") == cfg.task_key]
    assert probe_rows, f"No probe rows found for task_key={cfg.task_key!r} in {cfg.probe_predictions_path}"

    base_correct_override: dict[str, bool] = {}
    if cfg.base_predictions_path:
        base_correct_override = _load_base_correctness(
            cfg.base_predictions_path, cfg.base_dataset_name, cfg.base_task_name, cfg.base_gold_field_name)
        logger.info("Loaded %d base-model correctness overrides from %s",
                    len(base_correct_override), cfg.base_predictions_path)

    # sample_key format: "<Dataset>::<task>::<sample_id>" -- match on the bare
    # sample_id suffix since jlens_per_sample.jsonl stores the raw manifest id.
    probe_by_sid: dict[str, dict[str, Any]] = {}
    for r in probe_rows:
        sid = r["sample_key"].split("::")[-1]
        probe_by_sid[sid] = r

    layer_key = str(cfg.layer_idx)
    samples: list[dict[str, Any]] = []
    for jr in jlens_rows:
        sid = jr.get("sample_id", "")
        probe_r = probe_by_sid.get(sid) or probe_by_sid.get(sid.split(":")[-1])
        if probe_r is None:
            continue
        layer_data = jr.get("layers", {}).get(layer_key)
        if layer_data is None:
            continue
        norms = layer_data.get("per_token_norm", {})
        pos_norms = [norms[t] for t in cfg.positive_tokens if t in norms]
        neg_norms = [norms[t] for t in cfg.negative_tokens if t in norms]
        if not pos_norms or not neg_norms:
            continue
        jlens_score = max(pos_norms) - max(neg_norms)
        gold_raw = probe_r.get("gold_label", "")
        gold = _collapse(gold_raw, cfg.positive_label, cfg.negative_label, cfg.collapse_multiclass)
        # Recompute probe correctness from the (possibly multi-class) probe_pred,
        # collapsed the SAME way as gold/jlens, so all three judgments are compared
        # on a consistent binary footing regardless of whether the underlying probe
        # task was itself binary or multi-class (e.g. MMHS150K's 6-class task_key).
        probe_pred_raw = probe_r.get("probe_pred", "")
        probe_pred = _collapse(probe_pred_raw, cfg.positive_label, cfg.negative_label, cfg.collapse_multiclass)
        probe_correct = probe_pred == gold
        if sid in base_correct_override:
            base_correct = base_correct_override[sid]
        elif probe_r.get("baseline_pred") is not None:
            base_pred = _collapse(probe_r.get("baseline_pred", ""), cfg.positive_label, cfg.negative_label, cfg.collapse_multiclass)
            base_correct = base_pred == gold
        else:
            base_correct = bool(probe_r.get("baseline_correct"))
        samples.append({
            "sample_id": sid,
            "gold_label": gold,
            "gold_label_raw": gold_raw,
            "jlens_score": jlens_score,
            "probe_correct": probe_correct,
            "base_correct": base_correct,
        })

    assert samples, "No samples matched between J-lens and probe predictions -- check sample_id alignment"
    logger.info("Matched %d samples (task=%s, layer=%d)", len(samples), cfg.task_key, cfg.layer_idx)

    # ── Calibrate a threshold on jlens_score using a held-out calibration slice ──
    rng = random.Random(cfg.calibration_seed)
    order = list(range(len(samples)))
    rng.shuffle(order)
    n_cal = max(1, int(len(order) * cfg.calibration_fraction))
    cal_idx, eval_idx = set(order[:n_cal]), set(order[n_cal:])

    cal_samples = [samples[i] for i in cal_idx]
    best_t, best_acc = 0.0, -1.0
    scores = sorted(s["jlens_score"] for s in cal_samples)
    candidates = scores + [scores[0] - 1e-6, scores[-1] + 1e-6] if scores else [0.0]
    for t in candidates:
        acc = sum(
            1 for s in cal_samples
            if ((s["jlens_score"] >= t) == (s["gold_label"] == cfg.positive_label))
        ) / len(cal_samples)
        if acc > best_acc:
            best_acc, best_t = acc, t
    logger.info("Calibrated jlens threshold=%.5f (calib_n=%d, calib_acc=%.4f)", best_t, len(cal_samples), best_acc)

    # ── Apply threshold on the held-out eval slice; classify quadrants ──────────
    eval_samples = [samples[i] for i in eval_idx]
    for s in eval_samples:
        jlens_pred = cfg.positive_label if s["jlens_score"] >= best_t else cfg.negative_label
        s["jlens_pred"] = jlens_pred
        s["jlens_correct"] = jlens_pred == s["gold_label"]
        if s["probe_correct"] and s["jlens_correct"]:
            s["quadrant"] = "both_correct"
        elif s["probe_correct"] and not s["jlens_correct"]:
            s["quadrant"] = "probe_only"
        elif not s["probe_correct"] and s["jlens_correct"]:
            s["quadrant"] = "jlens_only"
        else:
            s["quadrant"] = "both_miss"

    def _quadrant_counts(subset: list[dict[str, Any]]) -> dict[str, int]:
        counts = {"both_correct": 0, "probe_only": 0, "jlens_only": 0, "both_miss": 0}
        for s in subset:
            counts[s["quadrant"]] += 1
        return counts

    overall_counts = _quadrant_counts(eval_samples)
    base_wrong_samples = [s for s in eval_samples if not s["base_correct"]]
    base_wrong_counts = _quadrant_counts(base_wrong_samples)

    def _ratio(counts: dict[str, int]) -> float | None:
        return round(counts["probe_only"] / counts["jlens_only"], 3) if counts["jlens_only"] > 0 else (
            float("inf") if counts["probe_only"] > 0 else None
        )

    summary = {
        "task_key": cfg.task_key,
        "layer_idx": cfg.layer_idx,
        "positive_tokens": cfg.positive_tokens,
        "negative_tokens": cfg.negative_tokens,
        "positive_label": cfg.positive_label,
        "negative_label": cfg.negative_label,
        "n_matched": len(samples),
        "n_calibration": len(cal_samples),
        "n_eval": len(eval_samples),
        "calibrated_threshold": round(best_t, 5),
        "calibration_accuracy": round(best_acc, 4),
        "overall": {
            "counts": overall_counts,
            "probe_only_over_jlens_only": _ratio(overall_counts),
        },
        "base_model_wrong_subset": {
            "n": len(base_wrong_samples),
            "counts": base_wrong_counts,
            "probe_only_over_jlens_only": _ratio(base_wrong_counts),
        },
    }

    (out_dir / "jlens_quadrant_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "jlens_quadrant_per_sample.jsonl").write_text(
        "\n".join(json.dumps(s) for s in eval_samples)
    )
    logger.info("Wrote jlens_quadrant_summary.json + jlens_quadrant_per_sample.jsonl to %s", out_dir)

    print("\n" + "=" * 80)
    print(f"Probe vs J-lens Quadrants: {cfg.task_key}")
    print(f"Positive tokens: {cfg.positive_tokens}  Negative tokens: {cfg.negative_tokens}")
    print(f"Calibrated threshold={best_t:.5f} (calib n={len(cal_samples)}, acc={best_acc:.4f})")
    print("-" * 80)
    print(f"{'Subset':22s} {'n':>5s} {'both_correct':>13s} {'probe_only':>11s} "
          f"{'jlens_only':>11s} {'both_miss':>10s} {'probe/jlens':>12s}")
    for name, subset_counts, n in (
        ("overall", overall_counts, len(eval_samples)),
        ("base_model_wrong", base_wrong_counts, len(base_wrong_samples)),
    ):
        ratio = _ratio(subset_counts)
        ratio_str = f"{ratio:.2f}" if isinstance(ratio, float) and ratio != float("inf") else str(ratio)
        print(f"{name:22s} {n:>5d} {subset_counts['both_correct']:>13d} {subset_counts['probe_only']:>11d} "
              f"{subset_counts['jlens_only']:>11d} {subset_counts['both_miss']:>10d} {ratio_str:>12s}")
    print("=" * 80)
    print("Success criterion: probe_only >> jlens_only in the base_model_wrong row"
          " replicates the FHM pattern.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = parse(JlensQuadrantConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
