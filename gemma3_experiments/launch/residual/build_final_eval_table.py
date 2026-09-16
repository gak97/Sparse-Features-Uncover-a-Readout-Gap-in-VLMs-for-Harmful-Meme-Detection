

import json
import logging
from pathlib import Path
from typing import Any

from gemma3_experiments.launch.residual.probe_validate import _per_class_metrics

logger = logging.getLogger(__name__)

ACT = Path("./outputs")

# ── File paths (frozen; not parameterised to avoid accidental test-set tuning) ──
PROBE_TEST_GENERATED         = ACT / "residual_sae_probe_test_generated" / "probe_val_metrics.json"
PROBE_TEST_IMAGE             = ACT / "residual_sae_probe_test_image" / "probe_val_metrics.json"
PROBE_CRISIS_HELDOUT         = ACT / "residual_sae_probe_crisis_heldout_image" / "probe_val_metrics.json"
FHM_BILINEAR_TEST            = ACT / "residual_sae_fhm_bilinear_r32_test" / "bilinear_test_metrics.json"
BASE_LVLM_4B_TEST            = ACT / "residual_sae_predictions.4b_gemma.all.test.yes_no.jsonl"
BASE_LVLM_4B_MMHS_TEST       = ACT / "residual_sae_predictions.4b_gemma.mmhs150k.test.label_score.jsonl"
BASE_LVLM_4B_CRISIS_HELDOUT  = ACT / "residual_sae_predictions.4b_gemma.crisis_a.heldout_validation.yes_no.jsonl"
BASE_LVLM_12B_TEST           = ACT / "residual_sae_predictions.12b_gemma.fhm_mami_mmhs.test.label_score.jsonl"
FINAL_SUMMARY                = ACT / "residual_sae_probe_test_final_summary" / "final_test_summary.json"
OUTPUT_DIR                   = ACT / "residual_sae_final_eval_table"

# Token-selection choice per task (frozen from validation — do NOT update from test numbers)
BEST_BY_TASK = {
    "Crisis_HateMM::subtask_a_hate_speech::label":              "image",
    "HarMeme::harmfulness_and_target::harmfulness [harm_binary]":"image",
    "MAMI::misogyny_and_subtypes::label":                        "generated",
    "MMHS150K::hate_category::label [binary]":                  "generated",
    "MultiOFF::offensive_detection::label":                     "generated",
    # Fine-grained (weaker, reported for completeness)
    "HarMeme::harmfulness_and_target::harmfulness":             "generated",
    "MMHS150K::hate_category::label":                           "generated",
    "MMHS150K::hate_category::label [hierarchical]":            "generated",
    "Crisis_HateMM::subtask_b_hate_target::target":             "generated",
}

# Task display names, type classification, and the held-out split actually used.
TASK_META = {
    "Hateful_Memes_Dataset::hateful_memes_binary::label":
        ("FHM (hateful/not)", "binary", "test_seen"),
    "HarMeme::harmfulness_and_target::harmfulness [harm_binary]":
        ("HarMeme (harmful/not)", "binary", "test"),
    "MAMI::misogyny_and_subtypes::label":
        ("MAMI (misogynous/not)", "binary", "test"),
    "MMHS150K::hate_category::label [binary]":
        ("MMHS (hate/notHate)", "binary", "test"),
    "MultiOFF::offensive_detection::label":
        ("MultiOFF (offensive/not)", "binary", "test"),
    "Crisis_HateMM::subtask_a_hate_speech::label":
        ("Crisis_A (hate/no hate)", "binary", "validation (no labeled test split exists)"),
    # Fine-grained
    "HarMeme::harmfulness_and_target::harmfulness":
        ("HarMeme 3-class harmfulness", "fine_grained", "test"),
    "MMHS150K::hate_category::label":
        ("MMHS 6-class hate type", "fine_grained", "test"),
    "MMHS150K::hate_category::label [hierarchical]":
        ("MMHS hierarchical hate type", "fine_grained", "test"),
    "Crisis_HateMM::subtask_b_hate_target::target":
        ("Crisis_B (hate target)", "fine_grained", "test"),
}

# Raw manifest field name (schema.primary_field.name) per dataset::task, used to
# read gold_fields from joint_infer.py's raw prediction rows.
_FIELD_NAME_BY_DATASET_TASK = {
    ("Hateful_Memes_Dataset", "hateful_memes_binary"): "label",
    ("HarMeme", "harmfulness_and_target"): "harmfulness",
    ("MAMI", "misogyny_and_subtypes"): "label",
    ("MMHS150K", "hate_category"): "label",
    ("MultiOFF", "offensive_detection"): "label",
    ("Crisis_HateMM", "subtask_a_hate_speech"): "label",
}

# Datasets/files eligible per row for the 4B and 12B base-LVLM columns.
_BASE_4B_FILES_BY_DATASET = {
    "Hateful_Memes_Dataset": [BASE_LVLM_4B_TEST],
    "HarMeme": [BASE_LVLM_4B_TEST],
    "MAMI": [BASE_LVLM_4B_TEST],
    "MultiOFF": [BASE_LVLM_4B_TEST],
    "MMHS150K": [BASE_LVLM_4B_MMHS_TEST],
    "Crisis_HateMM": [BASE_LVLM_4B_CRISIS_HELDOUT],
}
_BASE_12B_FILES_BY_DATASET = {
    "Hateful_Memes_Dataset": [BASE_LVLM_12B_TEST],
    "MAMI": [BASE_LVLM_12B_TEST],
    "MMHS150K": [BASE_LVLM_12B_TEST],
}

# Minimum held-out n below which a macro-F1 should be treated as unreliable.
_MIN_RELIABLE_N = 30


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        logger.warning("Missing: %s", path)
        return {}
    return json.loads(path.read_text())


def _probe_f1(task_key: str, metrics_gen: dict, metrics_img: dict) -> float | None:
    sel = BEST_BY_TASK.get(task_key, "generated")
    m = metrics_gen if sel == "generated" else metrics_img
    result = m.get(task_key)
    if result is None:
        return None
    pm = result.get("probe_metrics") or result.get("stage1_binary_metrics") or {}
    return pm.get("macro_f1")


def _collapse_binary(label: str, dataset_name: str) -> str:
    """Collapse a raw multi-class label to its binary headline view."""
    if dataset_name == "MMHS150K":
        return "NotHate" if label == "NotHate" else "Hate"
    if dataset_name == "HarMeme":
        return "not harmful" if label == "not harmful" else "harmful"
    return label


def _base_f1_from_jsonl(task_key: str, jsonl_paths: list[Path]) -> tuple[float | None, int | None]:
    """Compute macro-F1 (+n) for the base LVLM from joint_infer.py prediction JSONLs.

    joint_infer.py rows use `sample.to_manifest_dict()`, i.e. gold labels live
    under `gold_fields[<field_name>]`, not a flat `gold_label` key, and there is
    no `primary_field_name` key -- both were bugs in the original version of
    this function (it silently never matched any row).
    """
    ds, tn, field_suffix = task_key.split("::")
    field_name = _FIELD_NAME_BY_DATASET_TASK.get((ds, tn), "label")
    is_binary_view = "[" in field_suffix  # "...[harm_binary]" / "...[binary]" suffix marks a collapsed view

    golds: list[str] = []
    preds: list[str] = []
    for jsonl_path in jsonl_paths:
        if not jsonl_path.exists():
            continue
        for line in open(jsonl_path):
            row = json.loads(line)
            if row.get("dataset_name") != ds or row.get("task_name") != tn:
                continue
            gold = (row.get("gold_fields") or {}).get(field_name)
            pred = row.get("prediction_label")
            if gold is None or pred is None:
                continue
            if is_binary_view:
                gold = _collapse_binary(gold, ds)
                pred = _collapse_binary(pred, ds)
            golds.append(gold)
            preds.append(pred)

    if not golds:
        return None, None
    m = _per_class_metrics(golds, preds)
    return m.get("macro_f1"), len(golds)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_gen = _load_json(PROBE_TEST_GENERATED)
    metrics_img = _load_json(PROBE_TEST_IMAGE)
    metrics_crisis_heldout = _load_json(PROBE_CRISIS_HELDOUT)
    fhm_bilinear = _load_json(FHM_BILINEAR_TEST)
    existing = _load_json(FINAL_SUMMARY)

    # Build a lookup from existing final summary rows
    existing_by_key: dict[str, float] = {}
    for row in existing.get("rows", []):
        tk = row.get("task_key", "")
        f1 = row.get("macro_f1")
        if f1 is not None:
            existing_by_key[tk] = f1

    rows = []
    for task_key, (display, task_type, split) in TASK_META.items():
        ds = task_key.split("::")[0]
        is_fhm = task_key == "Hateful_Memes_Dataset::hateful_memes_binary::label"
        is_crisis_a = task_key == "Crisis_HateMM::subtask_a_hate_speech::label"

        # 4B probe F1
        if is_fhm:
            probe_f1 = existing_by_key.get(task_key)
            # 12B bilinear overrides for FHM
            probe_12b = fhm_bilinear.get("macro_f1")
        elif is_crisis_a:
            # Fixed 2026-07-18: the combined test-split probe runs evaluated this on
            # a stale 5-sample cache (see NOTE above); use the dedicated held-out
            # (validation) probe run instead.
            crisis_result = metrics_crisis_heldout.get(task_key)
            crisis_pm = (crisis_result or {}).get("probe_metrics") or (crisis_result or {}).get("stage1_binary_metrics") or {}
            probe_f1 = crisis_pm.get("macro_f1")
            probe_12b = None
        else:
            probe_f1 = _probe_f1(task_key, metrics_gen, metrics_img) or existing_by_key.get(task_key)
            probe_12b = None

        n = None
        if is_fhm:
            n = fhm_bilinear.get("n") if fhm_bilinear else existing_by_key.get("n")
        elif is_crisis_a:
            crisis_result = metrics_crisis_heldout.get(task_key)
            crisis_pm = (crisis_result or {}).get("probe_metrics") or (crisis_result or {}).get("stage1_binary_metrics") or {}
            n = crisis_pm.get("n")
        else:
            sel = BEST_BY_TASK.get(task_key, "generated")
            m = metrics_gen if sel == "generated" else metrics_img
            result = m.get(task_key)
            if result:
                pm = result.get("probe_metrics") or result.get("stage1_binary_metrics") or {}
                n = pm.get("n")

        # Base LVLM (4B), sourced from joint_infer.py raw prediction JSONLs.
        base_f1, base_n = _base_f1_from_jsonl(task_key, _BASE_4B_FILES_BY_DATASET.get(ds, []))

        # Base LVLM (12B) — only wired for FHM/MAMI/MMHS per the required table.
        base_12b_f1, base_12b_n = _base_f1_from_jsonl(task_key, _BASE_12B_FILES_BY_DATASET.get(ds, []))

        low_n = bool(base_n is not None and base_n < _MIN_RELIABLE_N)

        rows.append({
            "task_key": task_key,
            "display": display,
            "type": task_type,
            "split": split,
            "n": n,
            "probe_4b_f1": probe_f1,
            "probe_12b_bilinear_f1": probe_12b,
            "base_lvlm_4b_f1": base_f1,
            "base_lvlm_4b_n": base_n,
            "base_lvlm_12b_f1": base_12b_f1,
            "base_lvlm_12b_n": base_12b_n,
            "token_selection": BEST_BY_TASK.get(task_key),
            "low_n_warning": low_n,
        })

    # ── Print table ────────────────────────────────────────────────────────────
    def _fmt(v): return f"{v:.4f}" if v is not None else "  N/A "

    print()
    print("=" * 100)
    print("Final Base-vs-Probe Evaluation Table")
    print("Metric: macro-F1  |  Calibration: frozen from val/calib split, never tuned on the reported split")
    print("=" * 100)

    binary_rows = [r for r in rows if r["type"] == "binary"]
    fine_rows   = [r for r in rows if r["type"] == "fine_grained"]

    print("\n### Binary harmfulness/hate/misogyny/offensiveness (headline results) ###")
    print(f"{'Task':30s} {'Split':30s} {'4B Base':>9} {'4B Probe':>9} {'12B Base':>9} {'12B Best':>9}")
    print("-" * 100)
    for r in binary_rows:
        is_fhm = "FHM" in r["display"]
        best_12b = r["probe_12b_bilinear_f1"] if is_fhm and r["probe_12b_bilinear_f1"] else r["base_lvlm_12b_f1"]
        warn = "  [n<%d, unreliable]" % _MIN_RELIABLE_N if r["low_n_warning"] else ""
        print(f"  {r['display']:28s} {r['split']:30s} "
              f"{_fmt(r['base_lvlm_4b_f1']):>9} {_fmt(r['probe_4b_f1']):>9} "
              f"{_fmt(r['base_lvlm_12b_f1']):>9} {_fmt(best_12b):>9}{warn}")
    print("\n  12B Best = FHM bilinear r=32 (threshold frozen from calib split, 0.52) for FHM;")
    print("             base LVLM label_score otherwise (no 12B probe cached yet for MAMI/MMHS).")

    print("\n### Fine-grained taxonomy (weaker — never mixed with the binary claim above) ###")
    print(f"{'Task':30s} {'Split':30s} {'4B Base':>9} {'4B Probe':>9}")
    print("-" * 85)
    for r in fine_rows:
        print(f"  {r['display']:28s} {r['split']:30s} "
              f"{_fmt(r['base_lvlm_4b_f1']):>9} {_fmt(r['probe_4b_f1']):>9}")

    print("\n  NOTE: Fine-grained results are substantially weaker than binary.")
    print("  The correct paper claim is:")
    print("  'LVLM internals contain strong decodable signal for binary harmfulness/")
    print("   hatefulness/misogyny/offensiveness, and partial but weaker signal for")
    print("   fine-grained hate-type taxonomy.' Do not mix the two in a single number.")
    print()

    # ── Missing / unreliable data warnings ──────────────────────────────────────
    missing_probe = [r["display"] for r in rows if r["probe_4b_f1"] is None and r["probe_12b_bilinear_f1"] is None]
    if missing_probe:
        print(f"[WARN] Missing probe results for: {missing_probe}")
    if not fhm_bilinear:
        print("[WARN] FHM 12B bilinear test result MISSING.")
        print("  -> Run: sbatch gemma3_residual_sae/slurm/evaluate_fhm_bilinear_r32_test.sub")
    base_missing_4b = [r["display"] for r in rows if r["base_lvlm_4b_f1"] is None]
    if base_missing_4b:
        print(f"[WARN] 4B base LVLM results MISSING for: {base_missing_4b}")
        print("  -> Run: sbatch gemma3_residual_sae/slurm/baseline_4b_test_infer.sub")
        print("  -> Run: sbatch gemma3_residual_sae/slurm/baseline_4b_crisis_heldout_infer.sub  (Crisis_A)")
        print("  -> Run: sbatch gemma3_residual_sae/slurm/baseline_4b_mmhs_test_infer.sub        (MMHS150K)")
    base_missing_12b = [r["display"] for r in rows if r["display"] in
                        ("FHM (hateful/not)", "MAMI (misogynous/not)", "MMHS (hate/notHate)")
                        and r["base_lvlm_12b_f1"] is None]
    if base_missing_12b:
        print(f"[WARN] 12B base LVLM results MISSING for: {base_missing_12b}")
        print("  -> Run: sbatch gemma3_residual_sae/slurm/baseline_12b_test_infer.sub")
    if not metrics_crisis_heldout:
        print("[WARN] Crisis_A held-out probe result MISSING (falling back to the stale test-split probe, if any).")
        print("  -> Run: sbatch gemma3_residual_sae/slurm/probe_validate_crisis_heldout.sub")
    low_n_rows = [r["display"] for r in rows if r["low_n_warning"]]
    if low_n_rows:
        print(f"[WARN] Held-out n < {_MIN_RELIABLE_N} for: {low_n_rows} — treat as unreliable, do not headline.")
    print()

    # ── Save ──────────────────────────────────────────────────────────────────
    (OUTPUT_DIR / "final_eval_table.json").write_text(json.dumps(rows, indent=2))
    logger.info("Saved final_eval_table.json to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
