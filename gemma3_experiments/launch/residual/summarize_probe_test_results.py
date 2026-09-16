

import json
from pathlib import Path
from typing import Any

OUT_ROOT = Path("./outputs")

IMAGE_METRICS = OUT_ROOT / "residual_sae_probe_test_image" / "probe_val_metrics.json"
GENERATED_METRICS = OUT_ROOT / "residual_sae_probe_test_generated" / "probe_val_metrics.json"
FHM_FINAL = OUT_ROOT / "residual_sae_probe_test_fhm_crosscoder_prompt_image_plus_l22_l29_fixed_blend" / "fixed_blend_metrics.json"
FHM_OLD_BLEND = OUT_ROOT / "residual_sae_probe_test_fhm_l22_l29_fixed_blend" / "fixed_blend_metrics.json"
FHM_CROSS = OUT_ROOT / "residual_sae_probe_test_fhm_crosscoder_9_17_22_29_prompt_image_mlp_pseudo_hn" / "crosscoder_probe_metrics.json"

# Selection is frozen from validation observations, not chosen from test labels.
BEST_BY_TASK = {
    "Crisis_HateMM::subtask_a_hate_speech::label": "image",
    "Crisis_HateMM::subtask_b_hate_target::target": "generated",
    "HarMeme::harmfulness_and_target::harmfulness": "generated",
    "HarMeme::harmfulness_and_target::harmfulness [harm_binary]": "image",
    "MAMI::misogyny_and_subtypes::label": "generated",
    "MMHS150K::hate_category::label": "generated",
    "MMHS150K::hate_category::label [binary]": "generated",
    "MMHS150K::hate_category::label [hierarchical]": "generated",
    "MultiOFF::offensive_detection::label": "generated",
}

# Minimum test-set size to appear in the clean summary.
CRISIS_TEST_MIN_N = 20


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _metric_row(task_key: str, source: str, metrics_by_source: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    data = metrics_by_source.get(source, {})
    result = data.get(task_key)
    if result is None:
        return None
    metrics = result.get("probe_metrics") or result.get("stage1_binary_metrics") or {}
    n = metrics.get("n", result.get("n_val"))
    # Gate unreliable Crisis test rows: skip if the evaluated count is too small
    # to support a stable macro-F1 estimate.
    if task_key.startswith("Crisis_HateMM") and n is not None and n < CRISIS_TEST_MIN_N:
        return None
    return {
        "task_key": task_key,
        "source": source,
        "n": n,
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "gold_counts": metrics.get("gold_counts"),
        "pred_counts": metrics.get("pred_counts"),
        "per_class": metrics.get("per_class"),
    }


def main() -> None:
    image = _read_json(IMAGE_METRICS)
    generated = _read_json(GENERATED_METRICS)
    metrics_by_source = {"image": image, "generated": generated}

    rows = []
    for task_key, source in BEST_BY_TASK.items():
        row = _metric_row(task_key, source, metrics_by_source)
        if row is None:
            rows.append({"task_key": task_key, "source": source, "missing": True})
        else:
            rows.append(row)

    fhm_final = _read_json(FHM_FINAL)
    if fhm_final:
        rows.append({
            "task_key": "Hateful_Memes_Dataset::hateful_memes_binary::label",
            "source": "crosscoder_prompt_image_plus_l22_l29_fixed_blend",
            "n": fhm_final.get("metrics", {}).get("n"),
            "accuracy": fhm_final.get("metrics", {}).get("accuracy"),
            "macro_f1": fhm_final.get("metrics", {}).get("macro_f1"),
            "gold_counts": fhm_final.get("metrics", {}).get("gold_counts"),
            "pred_counts": fhm_final.get("metrics", {}).get("pred_counts"),
            "per_class": fhm_final.get("metrics", {}).get("per_class"),
            "pair_metrics": fhm_final.get("pair_metrics"),
        })
    else:
        rows.append({"task_key": "Hateful_Memes_Dataset::hateful_memes_binary::label", "source": "crosscoder_prompt_image_plus_l22_l29_fixed_blend", "missing": True})

    comparison = {
        "image_metrics_path": str(IMAGE_METRICS),
        "generated_metrics_path": str(GENERATED_METRICS),
        "fhm_final_path": str(FHM_FINAL),
        "fhm_old_blend_path": str(FHM_OLD_BLEND),
        "fhm_crosscoder_path": str(FHM_CROSS),
        "rows": rows,
    }
    out_dir = OUT_ROOT / "residual_sae_probe_test_final_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final_test_summary.json").write_text(json.dumps(comparison, indent=2))

    print("task_key\tsource\tn\taccuracy\tmacro_f1")
    for row in rows:
        print("\t".join(map(str, [
            row.get("task_key"),
            row.get("source"),
            row.get("n"),
            row.get("accuracy"),
            row.get("macro_f1"),
        ])))
    print(f"\nWrote {out_dir / 'final_test_summary.json'}")


if __name__ == "__main__":
    main()
