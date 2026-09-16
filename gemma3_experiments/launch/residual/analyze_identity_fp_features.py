

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from simple_parsing import ArgumentGenerationMode, parse


from gemma3_experiments.launch.residual.probe_validate import _read_jsonl
from gemma3_experiments.launch.residual.reward_monitor import RewardMonitor, RewardMonitorConfig, RewardWeights
from gemma3_experiments.meme_safety_probe.wrapper import HATEFUL_MEMES_SCHEMA
from gemma3_experiments.residual_sae import MemeDatasetConfig, TaskSchema, load_meme_samples
from gemma3_experiments.residual_sae.datasets import MAMI_SCHEMA

logger = logging.getLogger(__name__)

_SCHEMA_BY_TASK: dict[str, TaskSchema] = {
    "hateful_memes_binary": HATEFUL_MEMES_SCHEMA,
    "misogyny_and_subtypes": MAMI_SCHEMA,
}


@dataclass
class AnalyzeIdentityFPFeaturesConfig:
    reward_monitor: RewardMonitorConfig = field(default_factory=RewardMonitorConfig)
    identity_group_controls_path: str = ""       # full controls file (may span multiple datasets)
    identity_group_false_positives_path: str = ""  # from validate_reward_monitor.py, THIS dataset's run
    dataset_name: str = "Hateful_Memes_Dataset"
    task_name: str = "hateful_memes_binary"
    positive_label: str = "hateful"

    # Optional true-positive comparison group. Leave datasets empty to skip.
    comparison_dataset: MemeDatasetConfig = field(
        default_factory=lambda: MemeDatasetConfig(datasets=[], splits=[])
    )
    comparison_n_samples: int = 30

    top_n_group_features: int = 10   # size of the "group top feature set" used for Jaccard comparisons
    top_n_differential_features: int = 15  # size of the group-mean-difference ranking (the discriminative one)
    confounder_audit_path: str = ""  # FHM-only, optional cross-reference (confounder_feature_top_features.jsonl)
    output_dir: str = "./identity_fp_feature_analysis"


def _collect_feature_stats(monitor: RewardMonitor, schema: TaskSchema, samples: list[dict[str, Any]]):
    pass
    counts: Counter = Counter()
    activation_sum: dict[int, float] = defaultdict(float)
    vec_sum: torch.Tensor | None = None
    for s in samples:
        vec, _prompt_text, _generated_text = monitor.probe._extract_feature_vector(
            s["image_path"], s["meme_text"], schema,
        )
        vec_cpu = vec.detach().float().cpu()
        vec_sum = vec_cpu.clone() if vec_sum is None else vec_sum + vec_cpu
        nz = vec_cpu.nonzero(as_tuple=True)[0]
        if len(nz):
            order = torch.argsort(vec_cpu[nz].abs(), descending=True)[: monitor.cfg.probe.top_k_features]
            for i in order.tolist():
                idx = int(nz[i].item())
                counts[idx] += 1
                activation_sum[idx] += abs(float(vec_cpu[idx]))
    ranked = sorted(counts, key=lambda i: (-counts[i], -activation_sum[i]))
    mean_vec = vec_sum / len(samples) if (vec_sum is not None and samples) else vec_sum
    return ranked, counts, activation_sum, mean_vec


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _differential_ranking(
    mean_a: torch.Tensor, mean_b: torch.Tensor, descriptions: dict[str, str], top_n: int,
) -> dict[str, list[dict[str, Any]]]:
    """Rank features by (mean_a - mean_b) -- the group-mean ACTIVATION
    DIFFERENCE, not each sample's own top-K membership. This is what actually
    isolates features that differ BETWEEN groups (e.g. elevated specifically
    in false positives relative to true negatives), rather than features that
    are simply strong/frequent in every group.
    """
    diff = mean_a - mean_b
    top_a_over_b = torch.argsort(diff, descending=True)[:top_n].tolist()
    top_b_over_a = torch.argsort(diff, descending=False)[:top_n].tolist()

    def _rows(idxs: list[int]) -> list[dict[str, Any]]:
        return [
            {
                "feature_index": idx,
                "mean_diff": round(float(diff[idx]), 4),
                "mean_a": round(float(mean_a[idx]), 4),
                "mean_b": round(float(mean_b[idx]), 4),
                "description": descriptions.get(str(idx)),
            }
            for idx in idxs
        ]

    return {"elevated_in_a": _rows(top_a_over_b), "elevated_in_b": _rows(top_b_over_a)}


def _group_summary(
    label: str, ranked: list[int], counts: Counter, activation_sum: dict[int, float],
    n_samples: int, descriptions: dict[str, str], top_n: int,
) -> dict[str, Any]:
    top = ranked[:top_n]
    return {
        "group": label,
        "n_samples": n_samples,
        "top_features": [
            {
                "feature_index": idx,
                "n_samples_active": counts[idx],
                "frac_samples_active": round(counts[idx] / n_samples, 4) if n_samples else 0.0,
                "mean_abs_activation": round(activation_sum[idx] / counts[idx], 4) if counts[idx] else 0.0,
                "description": descriptions.get(str(idx)),
            }
            for idx in top
        ],
    }


def run(cfg: AnalyzeIdentityFPFeaturesConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = _SCHEMA_BY_TASK[cfg.task_name]

    monitor = RewardMonitor(cfg.reward_monitor, RewardWeights())
    descriptions = monitor.probe.feature_descriptions

    all_controls = _read_jsonl(Path(cfg.identity_group_controls_path))
    controls = [c for c in all_controls if c["dataset_name"] == cfg.dataset_name]
    assert controls, f"No controls found for dataset_name={cfg.dataset_name!r}"
    fp_rows = _read_jsonl(Path(cfg.identity_group_false_positives_path))
    fp_ids = {row["sample_id"] for row in fp_rows}

    fp_controls = [c for c in controls if c["sample_id"] in fp_ids]
    tn_controls = [c for c in controls if c["sample_id"] not in fp_ids]
    assert fp_controls, "No false-positive controls found -- did the gate actually fail for this dataset?"
    logger.info("dataset=%s: %d false-positive controls, %d true-negative controls", cfg.dataset_name, len(fp_controls), len(tn_controls))

    fp_ranked, fp_counts, fp_act, fp_mean_vec = _collect_feature_stats(monitor, schema, fp_controls)
    tn_ranked, tn_counts, tn_act, tn_mean_vec = _collect_feature_stats(monitor, schema, tn_controls) if tn_controls else ([], Counter(), {}, None)

    result: dict[str, Any] = {
        "dataset_name": cfg.dataset_name,
        "false_positives": _group_summary("false_positives", fp_ranked, fp_counts, fp_act, len(fp_controls), descriptions, cfg.top_n_group_features),
        "true_negatives": _group_summary("true_negatives", tn_ranked, tn_counts, tn_act, len(tn_controls), descriptions, cfg.top_n_group_features) if tn_controls else None,
    }

    fp_top_set = set(fp_ranked[: cfg.top_n_group_features])
    tn_top_set = set(tn_ranked[: cfg.top_n_group_features])
    result["fp_vs_tn_top_feature_jaccard"] = round(_jaccard(fp_top_set, tn_top_set), 4) if tn_controls else None
    logger.info("FP vs TN top-%d feature Jaccard: %s", cfg.top_n_group_features, result["fp_vs_tn_top_feature_jaccard"])

    if tn_mean_vec is not None:
        result["differential_fp_vs_tn"] = _differential_ranking(
            fp_mean_vec, tn_mean_vec, descriptions, cfg.top_n_differential_features,
        )
        logger.info(
            "Top FP-elevated-over-TN feature: f%d (mean_diff=%.4f)",
            result["differential_fp_vs_tn"]["elevated_in_a"][0]["feature_index"],
            result["differential_fp_vs_tn"]["elevated_in_a"][0]["mean_diff"],
        )

    # ── Optional true-positive comparison group ─────────────────────────
    if cfg.comparison_dataset.datasets:
        comp_samples = load_meme_samples(cfg.comparison_dataset)
        comp_positive = [
            {"image_path": s.image_path, "meme_text": s.meme_text, "sample_id": s.sample_id}
            for s in comp_samples
            if s.gold_fields.get(s.schema.primary_field.name) == cfg.positive_label
        ][: cfg.comparison_n_samples]
        assert comp_positive, f"No gold-positive comparison samples found for {cfg.comparison_dataset.datasets}"
        tp_ranked, tp_counts, tp_act, tp_mean_vec = _collect_feature_stats(monitor, schema, comp_positive)
        result["true_positives"] = _group_summary(
            "true_positives", tp_ranked, tp_counts, tp_act, len(comp_positive), descriptions, cfg.top_n_group_features,
        )
        tp_top_set = set(tp_ranked[: cfg.top_n_group_features])
        result["fp_vs_true_positive_top_feature_jaccard"] = round(_jaccard(fp_top_set, tp_top_set), 4)
        logger.info("FP vs true-positive top-%d feature Jaccard: %s", cfg.top_n_group_features, result["fp_vs_true_positive_top_feature_jaccard"])
        result["differential_fp_vs_true_positive"] = _differential_ranking(
            fp_mean_vec, tp_mean_vec, descriptions, cfg.top_n_differential_features,
        )

    # ── Optional FHM confounder-audit cross-reference ──────────────────
    if cfg.confounder_audit_path:
        audit_rows = _read_jsonl(Path(cfg.confounder_audit_path))
        audit_by_feature: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in audit_rows:
            audit_by_feature[row["feature_idx"]].append({
                "direction": row.get("direction"), "source": row.get("source"),
                "rank": row.get("rank"), "score": row.get("score"),
            })
        for entry in result["false_positives"]["top_features"]:
            hits = audit_by_feature.get(entry["feature_index"])
            if hits:
                entry["confounder_audit_hits"] = hits
        n_cross_referenced = sum(1 for e in result["false_positives"]["top_features"] if "confounder_audit_hits" in e)
        logger.info("%d/%d top FP features already appear in the confounder audit", n_cross_referenced, len(result["false_positives"]["top_features"]))
        for group_key in ("differential_fp_vs_tn", "differential_fp_vs_true_positive"):
            if group_key not in result:
                continue
            for entry in result[group_key]["elevated_in_a"] + result[group_key]["elevated_in_b"]:
                hits = audit_by_feature.get(entry["feature_index"])
                if hits:
                    entry["confounder_audit_hits"] = hits

    (out_dir / "identity_fp_feature_analysis.json").write_text(json.dumps(result, indent=2))

    lines = [f"Identity-group false-positive feature analysis: {cfg.dataset_name}", "=" * 60, ""]
    lines.append(f"False positives (n={len(fp_controls)}) top features:")
    for f in result["false_positives"]["top_features"]:
        desc = f"  -- {f['description']}" if f.get("description") else ""
        conf = f"  [confounder audit: {f['confounder_audit_hits']}]" if f.get("confounder_audit_hits") else ""
        lines.append(f"  f{f['feature_index']}: active in {f['n_samples_active']}/{len(fp_controls)} samples{desc}{conf}")
    if result["true_negatives"]:
        lines.append("")
        lines.append(f"True negatives (n={len(tn_controls)}) top features:")
        for f in result["true_negatives"]["top_features"]:
            desc = f"  -- {f['description']}" if f.get("description") else ""
            lines.append(f"  f{f['feature_index']}: active in {f['n_samples_active']}/{len(tn_controls)} samples{desc}")
    lines.append("")
    lines.append(f"FP vs TN top-feature Jaccard: {result['fp_vs_tn_top_feature_jaccard']}")
    if "fp_vs_true_positive_top_feature_jaccard" in result:
        lines.append(f"FP vs true-positive top-feature Jaccard: {result['fp_vs_true_positive_top_feature_jaccard']}")
    lines.append("")
    lines.append("NOTE: the two Jaccard/top-feature-frequency sections above are dominated by")
    lines.append("features that fire near-universally in EVERY group (confirm by checking")
    lines.append("frac_samples_active in the JSON) -- NOT discriminative. The sections below,")
    lines.append("ranked by group-MEAN-DIFFERENCE instead of per-sample top-K membership, are")
    lines.append("the ones that actually isolate features that differ BETWEEN groups.")

    def _write_differential(title: str, key: str, a_label: str, b_label: str) -> None:
        if key not in result:
            return
        lines.append("")
        lines.append(title)
        lines.append(f"  Features elevated in {a_label} (relative to {b_label}):")
        for f in result[key]["elevated_in_a"]:
            desc = f"  -- {f['description']}" if f.get("description") else ""
            conf = f"  [confounder audit: {f['confounder_audit_hits']}]" if f.get("confounder_audit_hits") else ""
            lines.append(f"    f{f['feature_index']}: mean_diff={f['mean_diff']} ({a_label}={f['mean_a']}, {b_label}={f['mean_b']}){desc}{conf}")
        lines.append(f"  Features elevated in {b_label} (relative to {a_label}):")
        for f in result[key]["elevated_in_b"]:
            desc = f"  -- {f['description']}" if f.get("description") else ""
            conf = f"  [confounder audit: {f['confounder_audit_hits']}]" if f.get("confounder_audit_hits") else ""
            lines.append(f"    f{f['feature_index']}: mean_diff={f['mean_diff']} ({a_label}={f['mean_a']}, {b_label}={f['mean_b']}){desc}{conf}")

    _write_differential(
        "DIFFERENTIAL: false positives vs true negatives (same identity-group control pool):",
        "differential_fp_vs_tn", "false_positives", "true_negatives",
    )
    _write_differential(
        "DIFFERENTIAL: false positives vs genuine true positives (gold-harmful comparison set):",
        "differential_fp_vs_true_positive", "false_positives", "true_positives",
    )
    (out_dir / "identity_fp_feature_report.txt").write_text("\n".join(lines) + "\n")
    logger.info("Wrote analysis to %s", out_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(AnalyzeIdentityFPFeaturesConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
