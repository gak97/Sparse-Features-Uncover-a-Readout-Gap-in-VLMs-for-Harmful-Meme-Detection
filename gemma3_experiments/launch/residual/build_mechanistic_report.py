

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any


ACT = Path("./outputs")


@dataclass
class ReportConfig:
    output_dir: Path = ACT / "mechanistic_analysis_gemma4b_it"
    top_k_features_per_class: int = 15
    top_k_global_features: int = 50
    exemplar_count: int = 6

    final_summary_path: Path = ACT / "residual_sae_probe_test_final_summary" / "final_test_summary.json"
    image_top_features_path: Path = ACT / "residual_sae_probe_results_harm_binary" / "top_features.json"
    generated_top_features_path: Path = ACT / "residual_sae_probe_results_mmhs_generated" / "top_features.json"
    image_predictions_path: Path = ACT / "residual_sae_probe_test_image" / "probe_val_predictions.jsonl"
    generated_predictions_path: Path = ACT / "residual_sae_probe_test_generated" / "probe_val_predictions.jsonl"
    fhm_final_predictions_path: Path = ACT / "residual_sae_probe_test_fhm_crosscoder_prompt_image_plus_l22_l29_fixed_blend" / "probe_val_predictions.jsonl"
    fhm_cross_predictions_path: Path = ACT / "residual_sae_probe_test_fhm_crosscoder_9_17_22_29_prompt_image_mlp_pseudo_hn" / "probe_val_predictions.jsonl"
    fhm_final_metrics_path: Path = ACT / "residual_sae_probe_test_fhm_crosscoder_prompt_image_plus_l22_l29_fixed_blend" / "fixed_blend_metrics.json"
    fhm_old_blend_metrics_path: Path = ACT / "residual_sae_probe_test_fhm_l22_l29_fixed_blend" / "fixed_blend_metrics.json"
    fhm_audit_summary_path: Path = ACT / "residual_sae_fhm_confounder_feature_audit" / "confounder_feature_audit_summary.json"
    fhm_audit_top_features_path: Path = ACT / "residual_sae_fhm_confounder_feature_audit" / "confounder_feature_top_features.jsonl"
    metadata_path: Path = Path("ivl_plus_gemini_captions_complete.json")


DATASET_FRIENDLY = {
    "Crisis_HateMM": "Crisis hate speech / target",
    "HarMeme": "COVID-era harmfulness",
    "Hateful_Memes_Dataset": "Facebook Hateful Memes",
    "MAMI": "Misogyny",
    "MMHS150K": "Twitter hate categories",
    "MultiOFF": "Offensive memes",
}


HATE_LIKE = {
    "hateful", "Hate", "Hate Speech", "harmful", "misogynous", "offensive",
    "somewhat harmful", "very harmful", "Racist", "Sexist", "Homophobe",
    "Religion", "OtherHate", "Community", "Individual", "Organization",
}


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as fd:
        for line in fd:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sample_key(row: dict[str, Any]) -> str:
    if row.get("sample_key"):
        return row["sample_key"]
    return f"{row['dataset_name']}::{row['task_name']}::{row['sample_id']}"


def _metadata_by_key(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_json(path, [])
    if isinstance(rows, dict):
        rows = list(rows.values())
    out = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sample_id = row.get("id")
        if sample_id is None and row.get("img"):
            sample_id = Path(str(row["img"])).stem
        if sample_id is None:
            continue
        sample_id_str = str(int(sample_id)) if isinstance(sample_id, float) and sample_id.is_integer() else str(sample_id)
        out[f"Hateful_Memes_Dataset::hateful_memes_binary::{sample_id_str}"] = row
    return out


def _is_hate_like(label: str) -> bool:
    return label in HATE_LIKE


def _feature_atlas(feature_sources: dict[str, dict[str, Any]], top_k: int) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    occurrences: dict[int, list[dict[str, Any]]] = defaultdict(list)
    class_cards: dict[str, list[dict[str, Any]]] = {}
    for source_name, top_features in feature_sources.items():
        for task_key, by_class in top_features.items():
            for class_name, feats in by_class.items():
                card_key = f"{source_name}::{task_key}::{class_name}"
                class_cards[card_key] = []
                for rank, feat in enumerate(feats[:top_k], start=1):
                    feature_i = int(feat["feature_idx"])
                    weight = float(feat["weight"])
                    row = {
                        "source": source_name,
                        "task_key": task_key,
                        "class_name": class_name,
                        "feature_idx": feature_i,
                        "weight": weight,
                        "abs_weight": abs(weight),
                        "rank": rank,
                        "class_is_hate_like": _is_hate_like(class_name),
                    }
                    occurrences[feature_i].append(row)
                    class_cards[card_key].append(row)
    atlas = []
    for feature_i, rows in occurrences.items():
        sources = sorted({row["source"] for row in rows})
        tasks = sorted({row["task_key"] for row in rows})
        classes = sorted({row["class_name"] for row in rows})
        hate_rows = [row for row in rows if row["class_is_hate_like"]]
        safe_rows = [row for row in rows if not row["class_is_hate_like"]]
        atlas.append({
            "feature_idx": feature_i,
            "n_occurrences": len(rows),
            "n_tasks": len(tasks),
            "sources": sources,
            "tasks": tasks,
            "classes": classes,
            "mean_abs_weight": round(mean(row["abs_weight"] for row in rows), 6),
            "max_abs_weight": round(max(row["abs_weight"] for row in rows), 6),
            "hate_like_weight_mean": round(mean(row["weight"] for row in hate_rows), 6) if hate_rows else None,
            "non_hate_like_weight_mean": round(mean(row["weight"] for row in safe_rows), 6) if safe_rows else None,
            "rows": sorted(rows, key=lambda row: row["abs_weight"], reverse=True),
        })
    atlas.sort(key=lambda row: (row["n_tasks"], row["n_occurrences"], row["max_abs_weight"]), reverse=True)
    return atlas, class_cards


def _prediction_slices(rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_task[row.get("task_key", "")].append(row)
    out = {}
    for task_key, task_rows in rows_by_task.items():
        categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in task_rows:
            gold = row.get("gold_label")
            pred = row.get("probe_pred")
            if gold == pred:
                cat = f"correct::{gold}"
            else:
                cat = f"{gold}-> {pred}"
            categories[cat].append(row)
        task_out = {}
        for cat, cat_rows in sorted(categories.items(), key=lambda item: (-len(item[1]), item[0])):
            examples = []
            for row in cat_rows[:6]:
                meta = metadata.get(row["sample_key"], {}) if metadata else {}
                examples.append({
                    "sample_key": row.get("sample_key"),
                    "gold": row.get("gold_label"),
                    "pred": row.get("probe_pred"),
                    "probe_proba": row.get("probe_proba"),
                    "text": meta.get("text"),
                    "gemini_caption": meta.get("gemini_caption"),
                    "split": meta.get("split"),
                })
            task_out[cat] = {"n": len(cat_rows), "examples": examples}
        out[task_key] = task_out
    return out


def _fhm_pair_delta_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return metrics.get("pair_metrics") or {}


def _audit_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    normalized = []
    for row in rows:
        if "top_features" in row:
            source = row.get("source") or row.get("pair_source") or row.get("name") or "unknown"
            for rank, feat in enumerate(row.get("top_features") or [], start=1):
                normalized.append({"source": source, "rank": rank, **feat})
        else:
            normalized.append(row)
    return normalized


def _overlap_with_audit(atlas: list[dict[str, Any]], audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audit_by_feature: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        if "feature_idx" in row:
            audit_by_feature[int(row["feature_idx"])].append(row)
    out = []
    for feature in atlas:
        audit = audit_by_feature.get(feature["feature_idx"], [])
        if audit:
            out.append({**{k: v for k, v in feature.items() if k != "rows"}, "audit_rows": audit})
    return out


def _format_metric(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Mechanistic Analysis Report — Gemma 4B-IT SAE/Crosscoder Probes")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("This report aggregates saved probe/crosscoder artifacts into a mechanistic map of what the internal features support across the six hate/meme datasets. It is not another validation sweep: it uses final test predictions, trained-probe top feature weights, FHM pseudo-confounder audits, and pair metrics.")
    lines.append("")
    lines.append("Key conclusions:")
    lines.append("- Generated-token residual SAE features carry the strongest cross-dataset decision signal for MAMI, MMHS, MultiOFF, and HarMeme 3-class.")
    lines.append("- Image-token features remain best for HarMeme harmful/not-harmful binary and provide important visual harm/symbol cues.")
    lines.append("- FHM remains the most confounder-sensitive task: pseudo-pair behavior improves with crosscoder blending, but global test F1 stays near 0.70.")
    lines.append("- Cross-dataset recurrent features indicate broad harm/safety axes, while MMHS and FHM require more specialized features and pair-sensitive reasoning.")
    lines.append("")

    lines.append("## Final Test Performance Snapshot")
    lines.append("")
    lines.append("| Task | Source | n | Accuracy | Macro-F1 |")
    lines.append("|------|--------|--:|---------:|---------:|")
    for row in report["final_summary"].get("rows", []):
        lines.append(f"| {row.get('task_key')} | {row.get('source')} | {row.get('n', '—')} | {_format_metric(row.get('accuracy'))} | {_format_metric(row.get('macro_f1'))} |")
    lines.append("")

    lines.append("## Cross-Dataset Feature Atlas")
    lines.append("")
    lines.append("Features below recur across task/class probes. Recurrence is evidence for broad semantic axes rather than one dataset artifact; signs are probe-label dependent and should be interpreted with the class label and encoder convention in mind.")
    lines.append("")
    lines.append("| Feature | Occurrences | Tasks | Max | Mean | Example classes |")
    lines.append("|---------|------------:|------:|----:|-----:|-----------------|")
    for feat in report["feature_atlas"][: report["config"]["top_k_global_features"]]:
        example_classes = "; ".join(f"{row['class_name']}@{row['source']}" for row in feat["rows"][:4])
        lines.append(f"| f{feat['feature_idx']} | {feat['n_occurrences']} | {feat['n_tasks']} | {feat['max_abs_weight']:.3f} | {feat['mean_abs_weight']:.3f} | {example_classes} |")
    lines.append("")

    lines.append("## FHM Benign/Pseudo-Confounder Analysis")
    lines.append("")
    fhm = report.get("fhm", {})
    lines.append("### Pair Metrics")
    lines.append("")
    lines.append("| Readout | Source | n pairs | Both-correct | Separated | Positive-margin | Median gap |")
    lines.append("|---------|--------|--------:|-------------:|----------:|----------------:|-----------:|")
    for readout, by_source in fhm.get("pair_metrics", {}).items():
        for source, vals in by_source.items():
            lines.append(f"| {readout} | {source} | {vals.get('n_pairs')} | {_format_metric(vals.get('both_correct_rate'))} | {_format_metric(vals.get('separated_rate'))} | {_format_metric(vals.get('positive_margin_rate'))} | {_format_metric(vals.get('median_prob_gap'))} |")
    lines.append("")

    if report.get("audit_feature_overlap"):
        lines.append("### Probe Features Overlapping FHM Confounder Audit")
        lines.append("")
        lines.append("| Feature | Tasks | Occurrences | Audit evidence |")
        lines.append("|---------|------:|------------:|----------------|")
        for row in report["audit_feature_overlap"][:25]:
            audit_evidence = "; ".join(
                f"{a.get('source','?')}:rank{a.get('rank','?')} score={a.get('score', a.get('effect', ''))}"
                for a in row.get("audit_rows", [])[:3]
            )
            lines.append(f"| f{row['feature_idx']} | {row['n_tasks']} | {row['n_occurrences']} | {audit_evidence} |")
        lines.append("")

    lines.append("## Error Slices And Exemplars")
    lines.append("")
    lines.append("The JSON report contains per-task error categories and up to six examples per category. These examples should be used for qualitative feature-card writing and causal patching target selection.")
    lines.append("")
    for task_key, slices in report.get("prediction_slices", {}).items():
        lines.append(f"### {task_key}")
        lines.append("")
        lines.append("| Slice | Count |")
        lines.append("|-------|------:|")
        for cat, payload in list(slices.items())[:12]:
            lines.append(f"| {cat} | {payload['n']} |")
        lines.append("")

    lines.append("## Mechanistic Follow-Up Experiments")
    lines.append("")
    lines.append("1. Build feature cards for the recurring atlas features and FHM-audit-overlap features: top positive/negative exemplars, activation histograms by label, and text/image/caption summaries.")
    lines.append("2. Run causal ablation/patching on FHM pseudo-pairs for the top crosscoder and residual-SAE features: patch hateful-pair feature activations into benign mates and vice versa.")
    lines.append("3. For MMHS, focus on generated-token features distinguishing Homophobe/Racist/OtherHate/Sexist; Religion has too few examples for high-confidence claims.")
    lines.append("4. For MAMI, investigate why generated features overpredict misogynous on test: compare false positives against training/validation exemplars and inspect whether high-weight features encode gender presence versus misogyny.")
    lines.append("5. For HarMeme, use the binary harm boundary as the mechanistic target; the 3-class very-harmful boundary is data-limited and not reliable enough for fine-grained causal claims.")
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n")


def build_report(cfg: ReportConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    final_summary = _read_json(cfg.final_summary_path, {"rows": []})
    image_top = _read_json(cfg.image_top_features_path, {})
    gen_top = _read_json(cfg.generated_top_features_path, {})
    feature_atlas, class_cards = _feature_atlas(
        {"image_or_harm_binary": image_top, "generated": gen_top},
        top_k=cfg.top_k_features_per_class,
    )

    metadata = _metadata_by_key(cfg.metadata_path)
    image_preds = _read_jsonl(cfg.image_predictions_path)
    gen_preds = _read_jsonl(cfg.generated_predictions_path)
    fhm_preds = _read_jsonl(cfg.fhm_final_predictions_path)
    prediction_slices = {}
    prediction_slices.update(_prediction_slices(image_preds, metadata))
    prediction_slices.update(_prediction_slices(gen_preds, metadata))
    prediction_slices.update(_prediction_slices(fhm_preds, metadata))

    fhm_final = _read_json(cfg.fhm_final_metrics_path, {})
    fhm_old = _read_json(cfg.fhm_old_blend_metrics_path, {})
    audit_summary = _read_json(cfg.fhm_audit_summary_path, {})
    audit_rows = _audit_rows(cfg.fhm_audit_top_features_path)
    audit_overlap = _overlap_with_audit(feature_atlas, audit_rows)

    report = {
        "config": {
            "top_k_features_per_class": cfg.top_k_features_per_class,
            "top_k_global_features": cfg.top_k_global_features,
            "exemplar_count": cfg.exemplar_count,
        },
        "inputs": {k: str(v) for k, v in cfg.__dict__.items() if k.endswith("path")},
        "final_summary": final_summary,
        "feature_atlas": feature_atlas,
        "class_feature_cards": class_cards,
        "fhm": {
            "pair_metrics": {
                "old_layer22_29_blend": _fhm_pair_delta_summary(fhm_old),
                "final_crosscoder_prompt_image_blend": _fhm_pair_delta_summary(fhm_final),
            },
            "audit_summary": audit_summary,
        },
        "audit_feature_overlap": audit_overlap,
        "prediction_slices": prediction_slices,
    }
    (cfg.output_dir / "mechanistic_report.json").write_text(json.dumps(report, indent=2))
    _write_markdown(report, cfg.output_dir / "MECHANISTIC_REPORT.md")
    print(f"Wrote {cfg.output_dir / 'MECHANISTIC_REPORT.md'}")
    print(f"Wrote {cfg.output_dir / 'mechanistic_report.json'}")


def main() -> None:
    build_report(ReportConfig())


if __name__ == "__main__":
    main()
