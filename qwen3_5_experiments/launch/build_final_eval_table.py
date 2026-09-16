

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simple_parsing import Serializable, parse

from ..task_labels import (
    BINARY_TASK_DATASET,
    BINARY_TASK_FIELD,
    MULTICLASS_TASKS,
    compute_metrics,
)

logger = logging.getLogger(__name__)

# ── Headline binary table (macro-F1, one row per system) ─────────────────────
HEADLINE_TASKS: list[tuple[str, str, str]] = [
    ("Crisis_HateMM", "subtask_a_hate_speech", "Crisis_HateMM (hate/no-hate)"),
    ("Hateful_Memes_Dataset", "hateful_memes_binary", "Hateful Memes (hateful/not)"),
    ("MAMI", "misogyny_and_subtypes", "MAMI (misogynous/not)"),
    ("MultiOFF", "offensive_detection", "MultiOFF (offensive/not)"),
    ("HarMeme", "harmfulness_and_target", "HarMeme (harmful/not)"),
    ("MMHS150K", "hate_category", "MMHS150K (Hate/NotHate)"),
]

# ── Fine-grained taxonomy, reported SEPARATELY — never mixed into headline ──
FINE_GRAINED_TASKS: list[tuple[str, str, str]] = [
    ("HarMeme", "harmfulness_and_target", "HarMeme 3-class harmfulness"),
    ("MMHS150K", "hate_category", "MMHS150K 6-class hate taxonomy"),
]


@dataclass
class FinalTableConfig(Serializable):
    base_lvlm_jsonl: str = ""
    hook_jsonl: str = ""
    base_sae_probe_jsonl: str = ""
    residual_sae_probe_jsonl: str = ""
    output_dir: str = "./qwen_final_eval_table"


def _read_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not path or not p.exists():
        logger.warning("Missing input file: %s", path)
        return []
    rows = []
    with p.open() as fd:
        for line in fd:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _lvlm_metrics(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    """From a joint_infer.py predictions jsonl, compute macro-F1 for every
    headline binary task (native or collapsed) and every fine-grained task.
    Keyed by (dataset_name, base_task_name, view)."""
    by_task: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_task[(r.get("dataset_name"), r.get("task_name"))].append(r)

    out: dict[tuple[str, str, str], dict] = {}

    # Native binary tasks.
    for task_name, field in BINARY_TASK_FIELD.items():
        ds = BINARY_TASK_DATASET[task_name]
        trows = by_task.get((ds, task_name), [])
        golds, preds = [], []
        for r in trows:
            gv = (r.get("gold_fields") or {}).get(field)
            pv = r.get("prediction_label")
            if gv is None or pv is None:
                continue
            golds.append(gv)
            preds.append(pv)
        if golds:
            out[(ds, task_name, "binary")] = compute_metrics(golds, preds)

    # Multi-class tasks: fine-grained + binary-collapsed.
    for task_name, spec in MULTICLASS_TASKS.items():
        trows = by_task.get((spec.dataset_name, task_name), [])
        golds, preds = [], []
        for r in trows:
            gv = (r.get("gold_fields") or {}).get(spec.field)
            pv = r.get("prediction_label")
            if gv is None or pv is None:
                continue
            golds.append(gv)
            preds.append(pv)
        if golds:
            out[(spec.dataset_name, task_name, "fine_grained")] = compute_metrics(
                golds, preds, labels=list(spec.classes)
            )
            gb = [spec.collapse(v) for v in golds]
            pb = [spec.collapse(v) for v in preds]
            out[(spec.dataset_name, task_name, "binary")] = compute_metrics(
                gb, pb, labels=[spec.binary_negative, spec.binary_positive]
            )
    return out


def _probe_metrics(rows: list[dict], require_split: str | None = None) -> dict[tuple[str, str, str], dict]:
    """From a probe_base_sae.py / probe_sae_features.py results jsonl, build the
    same (dataset_name, base_task_name, view) -> {accuracy, macro_f1, n} lookup."""
    out: dict[tuple[str, str, str], dict] = {}
    for r in rows:
        if require_split is not None and r.get("split") not in (require_split, None):
            continue
        ds = r.get("dataset_name")
        task_name = r.get("task_name", "")
        view = r.get("view", "binary")
        base_task_name = task_name.replace(" [binary]", "")
        macro_f1 = r.get("macro_f1")
        if macro_f1 is None:
            continue
        out[(ds, base_task_name, view)] = {
            "accuracy": r.get("accuracy"),
            "macro_f1": macro_f1,
            "n": r.get("n_eval") or r.get("n"),
        }
    return out


def _fmt(v: Any) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "  N/A "


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = parse(FinalTableConfig)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_lvlm = _lvlm_metrics(_read_jsonl(cfg.base_lvlm_jsonl))
    hook = _lvlm_metrics(_read_jsonl(cfg.hook_jsonl))
    # Base SAE probe: use ONLY "proper" (train -> held-out val/test) rows, never
    # the 80/20 train_holdout diagnostic rows — no threshold leakage.
    base_sae_probe = _probe_metrics(_read_jsonl(cfg.base_sae_probe_jsonl), require_split="proper")
    # Residual SAE probe (ablation): train_holdout rows have split unset.
    residual_sae_probe = _probe_metrics(_read_jsonl(cfg.residual_sae_probe_jsonl), require_split=None)

    systems = [
        ("Qwen base LVLM", base_lvlm),
        ("Qwen hook", hook),
        ("Qwen base SAE probe", base_sae_probe),
        ("Qwen residual SAE probe (ABLATION ONLY)", residual_sae_probe),
    ]

    rows_out: list[dict] = []

    print()
    print("=" * 100)
    print("LOCKED FINAL EVALUATION TABLE — Qwen3.5-9B multimodal hate/harm detection")
    print("Metric: macro-F1  |  Protocol: train -> proper held-out (val for Crisis_HateMM, test/test_seen else)")
    print("No threshold tuning on eval split (argmax decode / argmax probe — no calibration surface).")
    print("=" * 100)

    print("\n### Binary hate/harm/misogyny/offensiveness detection (HEADLINE — main claim) ###\n")
    header = f"{'Task':<32}" + "".join(f"{name:>26}" for name, _ in systems)
    print(header)
    print("-" * len(header))
    for ds, task_name, display in HEADLINE_TASKS:
        line = f"{display:<32}"
        row_record = {"task": display, "dataset_name": ds, "task_name": task_name, "type": "binary"}
        for sys_name, metrics_map in systems:
            m = metrics_map.get((ds, task_name, "binary"))
            f1 = m["macro_f1"] if m else None
            line += f"{_fmt(f1):>26}"
            row_record[sys_name] = f1
        print(line)
        rows_out.append(row_record)

    # Mean macro-F1 across the 6 headline tasks, per system (ablation excluded
    # from being presented as "the" result — still computed for reference).
    print("-" * len(header))
    mean_line = f"{'Mean (6 tasks)':<32}"
    mean_record = {"task": "Mean (6 tasks)", "type": "binary_mean"}
    for sys_name, metrics_map in systems:
        vals = [metrics_map.get((ds, tn, "binary"), {}).get("macro_f1")
                for ds, tn, _ in HEADLINE_TASKS]
        vals = [v for v in vals if v is not None]
        mean_f1 = sum(vals) / len(vals) if vals else None
        mean_line += f"{_fmt(mean_f1):>26}"
        mean_record[sys_name] = mean_f1
    print(mean_line)
    rows_out.append(mean_record)

    print("\n### Fine-grained taxonomy (SEPARATE — not part of the binary headline claim) ###\n")
    print(header)
    print("-" * len(header))
    for ds, task_name, display in FINE_GRAINED_TASKS:
        line = f"{display:<32}"
        row_record = {"task": display, "dataset_name": ds, "task_name": task_name, "type": "fine_grained"}
        for sys_name, metrics_map in systems:
            m = metrics_map.get((ds, task_name, "fine_grained"))
            f1 = m["macro_f1"] if m else None
            line += f"{_fmt(f1):>26}"
            row_record[sys_name] = f1
        print(line)
        rows_out.append(row_record)

    print()
    print("NOTE: 'Qwen residual SAE probe' is an ABLATION column only. Per the locked")
    print("protocol it must not be quoted as the headline SAE-probe result — that is")
    print("'Qwen base SAE probe' (public TopK SAE, 65536 latents, k=100).")
    print()

    missing_by_system: dict[str, list[str]] = defaultdict(list)
    for r in rows_out:
        if r.get("type") not in ("binary", "fine_grained"):
            continue
        for name, _ in systems:
            if r.get(name) is None:
                missing_by_system[name].append(r["task"])
    for name, tasks in missing_by_system.items():
        print(f"WARNING: '{name}' is missing results for: {tasks}")
    if "Qwen base SAE probe" in missing_by_system:
        print("  -> Re-run: sbatch qwen3_residual_sae/slurm/probe_base_sae.sub")
    if "Qwen residual SAE probe (ABLATION ONLY)" in missing_by_system:
        print("  -> Re-run: sbatch qwen3_residual_sae/slurm/probe_sae_features.sub")
    if "Qwen base LVLM" in missing_by_system or "Qwen hook" in missing_by_system:
        print("  -> Re-run: sbatch qwen3_residual_sae/slurm/infer_test_splits.sub")

    (output_dir / "final_eval_table.json").write_text(json.dumps(rows_out, indent=2))
    logger.info("Saved final_eval_table.json to %s", output_dir)


if __name__ == "__main__":
    main()
