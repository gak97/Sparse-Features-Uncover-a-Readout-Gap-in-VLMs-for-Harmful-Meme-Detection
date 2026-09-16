

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simple_parsing import ArgumentGenerationMode, parse

logger = logging.getLogger(__name__)


@dataclass
class CausalMediationConfig:
    results_dir: str = ""              # dir with silent_knowledge_<task>[_<tag>].json
    routing_summary_path: str = ""     # routing_features_summary.json (for jaccard/overlap)
    tasks: list[str] = field(default_factory=lambda: ["MAMI", "HarMeme", "MMHS", "Crisis_A"])
    output_dir: str = ""


def _load_result(results_dir: Path, task: str, tag: str) -> dict[str, Any] | None:
    task_safe = task.lower()
    # New naming: silent_knowledge_<task>_<tag>.json. Fall back to the legacy
    # untagged filename (silent_knowledge_<task>.json) for the "silent" condition,
    # since the original 2026-07-09 runs predate the `result_tag` config field.
    candidates = [
        results_dir / f"silent_knowledge_{task_safe}_{tag}.json",
    ]
    if tag == "silent":
        candidates.append(results_dir / f"silent_knowledge_{task_safe}.json")
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text())
    return None


def _mean_abs(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(sum(abs(v) for v in clean) / len(clean), 4) if clean else None


def _fmt(v: Any) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"


def run(cfg: CausalMediationConfig) -> None:
    results_dir = Path(cfg.results_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    routing_summary: dict[str, Any] = {}
    if cfg.routing_summary_path and Path(cfg.routing_summary_path).exists():
        routing_summary = json.loads(Path(cfg.routing_summary_path).read_text())

    rows: list[dict[str, Any]] = []
    for task in cfg.tasks:
        silent = _load_result(results_dir, task, "silent")
        routed = _load_result(results_dir, task, "routed")
        overlap_info = routing_summary.get(task, {})

        for label, data, expected_role in (
            ("silent (probe-derived)", silent,
             "probe advantage: mediates probe score, base output largely unaffected"),
            ("routed (J-lens-derived)", routed,
             "model output bias: mediates base output/margin, not necessarily correct classification"),
        ):
            if data is None:
                rows.append({
                    "task": task, "feature_class": label,
                    "status": "MISSING - experiment not yet run",
                    "expected_role": expected_role,
                })
                continue
            agg = data["aggregate"]
            per_sample = data.get("per_sample", [])
            abl_stats = agg.get("ablation") or {}
            patch_stats = agg.get("patch") or {}
            probe_stats = agg.get("probe_score") or {}
            # Legacy files (pre-2026-07-18) don't have mean_abs_delta_margin/mean_abs_delta --
            # fall back to computing it from per_sample deltas so old and new results stay
            # directly comparable.
            ablation_abs = abl_stats.get("mean_abs_delta_margin")
            if ablation_abs is None:
                ablation_abs = _mean_abs([r.get("delta_ablation") for r in per_sample])
            patch_abs = patch_stats.get("mean_abs_delta_margin")
            if patch_abs is None:
                patch_abs = _mean_abs([r.get("delta_patch") for r in per_sample])
            probe_abs = probe_stats.get("mean_abs_delta")
            if probe_abs is None:
                probe_abs = _mean_abs([r.get("probe_score_delta") for r in per_sample])

            rows.append({
                "task": task,
                "feature_class": label,
                "n_targets": agg.get("n_targets"),
                "top_k_features": agg.get("top_k_features"),
                "ablation_mean_abs_delta_margin": ablation_abs,
                "ablation_frac_toward_correct": abl_stats.get("frac_margin_toward_correct"),
                "patch_mean_abs_delta_margin": patch_abs,
                "patch_frac_toward_correct": patch_stats.get("frac_margin_toward_correct"),
                "probe_score_mean_abs_delta": probe_abs,
                "expected_role": expected_role,
            })

        rows.append({
            "task": task, "feature_class": "overlap (silent vs routed)",
            "n_overlap": overlap_info.get("n_overlap"),
            "jaccard": overlap_info.get("jaccard_routing_vs_probe"),
            "expected_role": "should be near-zero if the two classes are genuinely distinct",
        })

    (out_dir / "causal_mediation_table.json").write_text(json.dumps(rows, indent=2))
    logger.info("Wrote causal_mediation_table.json to %s", out_dir)

    # ── Print human-readable table ──────────────────────────────────────────
    print("\n" + "=" * 110)
    print("Causal Mediation Table: Silent vs Routed Features")
    print("=" * 110)
    header = (f"{'Task':10s} {'Feature class':26s} {'AblΔ|margin|':>12s} {'PatchΔ|margin|':>14s} "
              f"{'ProbeΔ':>10s}")
    print(header)
    print("-" * 110)
    for task in cfg.tasks:
        task_rows = [r for r in rows if r["task"] == task]
        for r in task_rows:
            if r["feature_class"].startswith("overlap"):
                print(f"{task:10s} {'overlap (silent vs routed)':26s} "
                      f"n_overlap={r.get('n_overlap')}  jaccard={r.get('jaccard')}")
                continue
            if r.get("status"):
                print(f"{task:10s} {r['feature_class']:26s}  {r['status']}")
                continue
            print(f"{task:10s} {r['feature_class']:26s} "
                  f"{_fmt(r['ablation_mean_abs_delta_margin']):>12s} {_fmt(r['patch_mean_abs_delta_margin']):>14s} "
                  f"{_fmt(r['probe_score_mean_abs_delta']):>10s} ")
    print("=" * 110)
    print("\nExpected pattern (per feedback):")
    print("  silent:  probe_score delta LARGE, margin delta SMALL relative to routed")
    print("  routed:  margin delta LARGER than silent, but base label correctness need not improve")
    print("  overlap: near-zero jaccard confirms the two feature classes are largely disjoint")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = parse(CausalMediationConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
