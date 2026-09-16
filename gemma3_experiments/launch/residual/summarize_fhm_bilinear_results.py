

import json
import statistics
from pathlib import Path
from typing import Any

ACT = Path("./outputs")

# ── Fixed reference numbers ────────────────────────────────────────────────────
# These are stable results from earlier experiments.  Update if a re-run changes
# them, but do not tune them on test_seen.
FIXED_REFS: list[dict[str, Any]] = [
    {
        "label": "12B LVLM baseline (label_score)",
        "cal_f1": 0.6603,
        "pair_img_both_correct": None,
        "pair_txt_both_correct": None,
        "n_seeds": None,
        "note": "label_score constrained decode, no probe",
    },
    {
        "label": "Linear pairwise readout (token-pair, 4B)",
        "cal_f1": 0.6927,
        "pair_img_both_correct": None,
        "pair_txt_both_correct": None,
        "n_seeds": None,
        "note": "probe_fhm_token_pair_readout, calibrated threshold",
    },
    {
        "label": "12B resid31 MLP + IVL pseudo-HN (1 seed)",
        "cal_f1": 0.7633,
        "pair_img_both_correct": None,
        "pair_txt_both_correct": None,
        "n_seeds": 1,
        "note": "probe_fhm_crosscoder_12b_resid31 calibrated; threshold tuned on full val (not calib-split)",
    },
]

# ── Directories for seed-sweep results ────────────────────────────────────────
SEED_SWEEP_DIRS: dict[str, list[Path]] = {
    "Bilinear r=8  (5 seeds)": [
        ACT / f"residual_sae_probe_validation_fhm_bilinear_r8_seed{s}" for s in range(5)
    ],
    "Bilinear r=32 (5 seeds)": [
        ACT / f"residual_sae_probe_validation_fhm_bilinear_r32_seed{s}" for s in range(5)
    ],
}

# ── Single-run reference directories ──────────────────────────────────────────
SINGLE_RUN_DIRS: dict[str, Path] = {
    "Bilinear r=128 (full-rank, 1 seed)": (
        ACT / "residual_sae_probe_validation_fhm_crosscoder_pairwise_12b_bilinear"
    ),
    "Linear pairwise gate (12B resid31, 1 seed)": (
        ACT / "residual_sae_probe_validation_fhm_crosscoder_pairwise_12b_resid31"
    ),
}


def _load_metrics(path: Path) -> dict[str, Any] | None:
    f = path / "pairwise_probe_metrics.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def _extract_cal_f1(m: dict[str, Any]) -> float | None:
    return (m.get("calibrated_metrics") or {}).get("macro_f1")


def _extract_pair_both_correct(m: dict[str, Any], src: str) -> float | None:
    pm = (m.get("pair_metrics") or {}).get(src)
    if pm is None:
        return None
    return pm.get("both_correct_rate") or pm.get("both_correct")


def _fmt(val: float | None, decimals: int = 4) -> str:
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def _fmt_mean_std(vals: list[float], decimals: int = 4) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.{decimals}f}"
    mu = statistics.mean(vals)
    sd = statistics.stdev(vals)
    return f"{mu:.{decimals}f} ± {sd:.{decimals}f}"


def _col_width(col: list[str], header: str) -> int:
    return max(len(header), max((len(c) for c in col), default=0))


def main() -> None:
    rows: list[dict[str, Any]] = []

    # Fixed reference rows
    for ref in FIXED_REFS:
        rows.append({
            "label": ref["label"],
            "cal_f1_str": _fmt(ref["cal_f1"]),
            "pair_img_str": _fmt(ref.get("pair_img_both_correct"), 3),
            "pair_txt_str": _fmt(ref.get("pair_txt_both_correct"), 3),
            "n_seeds": ref.get("n_seeds"),
            "note": ref.get("note", ""),
            # raw values for best-seed reporting
            "cal_f1_vals": [ref["cal_f1"]] if ref["cal_f1"] is not None else [],
        })

    # Seed-sweep rows
    for label, dirs in SEED_SWEEP_DIRS.items():
        loaded = [(d, _load_metrics(d)) for d in dirs]
        missing = [d for d, m in loaded if m is None]
        available = [(d, m) for d, m in loaded if m is not None]

        cal_f1_vals = [v for _, m in available for v in [_extract_cal_f1(m)] if v is not None]
        pair_img_vals = [v for _, m in available for v in [_extract_pair_both_correct(m, "pseudo_img")] if v is not None]
        pair_txt_vals = [v for _, m in available for v in [_extract_pair_both_correct(m, "pseudo_text")] if v is not None]

        best_seed = None
        best_f1 = -1.0
        for i, (d, m) in enumerate(loaded):
            if m is None:
                continue
            f1 = _extract_cal_f1(m) or 0.0
            if f1 > best_f1:
                best_f1, best_seed = f1, i

        note_parts = []
        if missing:
            note_parts.append(f"{len(missing)} seed(s) missing: {[str(d.name) for d in missing]}")
        if best_seed is not None:
            note_parts.append(f"best seed={best_seed} cal_f1={best_f1:.4f}")

        rows.append({
            "label": label,
            "cal_f1_str": _fmt_mean_std(cal_f1_vals),
            "pair_img_str": _fmt_mean_std(pair_img_vals, 3),
            "pair_txt_str": _fmt_mean_std(pair_txt_vals, 3),
            "n_seeds": len(available),
            "note": "; ".join(note_parts),
            "cal_f1_vals": cal_f1_vals,
            "best_seed": best_seed,
        })

    # Single-run rows
    for label, d in SINGLE_RUN_DIRS.items():
        m = _load_metrics(d)
        if m is None:
            rows.append({
                "label": label, "cal_f1_str": "MISSING",
                "pair_img_str": "—", "pair_txt_str": "—",
                "n_seeds": 0, "note": f"not found: {d.name}", "cal_f1_vals": [],
            })
            continue
        cal_f1 = _extract_cal_f1(m)
        rows.append({
            "label": label,
            "cal_f1_str": _fmt(cal_f1),
            "pair_img_str": _fmt(_extract_pair_both_correct(m, "pseudo_img"), 3),
            "pair_txt_str": _fmt(_extract_pair_both_correct(m, "pseudo_text"), 3),
            "n_seeds": 1,
            "note": f"best_epoch={m.get('best_epoch')}",
            "cal_f1_vals": [cal_f1] if cal_f1 is not None else [],
        })

    # ── Print table ────────────────────────────────────────────────────────────
    H_MODEL = "Model"
    H_F1 = "Cal F1 (mean±std)"
    H_IMG = "Pair-img both-correct"
    H_TXT = "Pair-txt both-correct"
    H_N = "N seeds"

    labels  = [r["label"] for r in rows]
    f1_strs = [r["cal_f1_str"] for r in rows]
    img_strs = [r["pair_img_str"] for r in rows]
    txt_strs = [r["pair_txt_str"] for r in rows]
    n_strs  = [str(r["n_seeds"]) if r["n_seeds"] is not None else "—" for r in rows]

    w_model = _col_width(labels, H_MODEL)
    w_f1    = _col_width(f1_strs, H_F1)
    w_img   = _col_width(img_strs, H_IMG)
    w_txt   = _col_width(txt_strs, H_TXT)
    w_n     = _col_width(n_strs, H_N)

    sep = f"+-{'-'*w_model}-+-{'-'*w_f1}-+-{'-'*w_img}-+-{'-'*w_txt}-+-{'-'*w_n}-+"
    header = f"| {H_MODEL:<{w_model}} | {H_F1:<{w_f1}} | {H_IMG:<{w_img}} | {H_TXT:<{w_txt}} | {H_N:<{w_n}} |"

    print()
    print("FHM Bilinear Probe — Publication Comparison Table")
    print("=" * len(sep))
    print(sep)
    print(header)
    print(sep)
    for r, lbl, f1, img, txt, n in zip(rows, labels, f1_strs, img_strs, txt_strs, n_strs):
        print(f"| {lbl:<{w_model}} | {f1:<{w_f1}} | {img:<{w_img}} | {txt:<{w_txt}} | {n:<{w_n}} |")
        if r.get("note"):
            note = r["note"]
            print(f"|   {'↳ ' + note:<{w_model+w_f1+w_img+w_txt+w_n+11}} |")
    print(sep)

    # ── Best seed per rank ─────────────────────────────────────────────────────
    print()
    print("Best seed per rank (select by cal_f1 on calib split; report test_seen ONCE):")
    for r in rows:
        if r.get("best_seed") is not None and r["cal_f1_vals"]:
            best_idx = r["best_seed"]
            best_f1  = max(r["cal_f1_vals"])
            print(f"  {r['label']}: seed={best_idx}  cal_f1={best_f1:.4f}")

    # ── Readiness check ────────────────────────────────────────────────────────
    n_r8 = next((r["n_seeds"] for r in rows if "r=8" in r["label"]), 0)
    n_r32 = next((r["n_seeds"] for r in rows if "r=32" in r["label"]), 0)
    print()
    if n_r8 >= 3 and n_r32 >= 3:
        print("✓ Seed sweep complete (≥3 seeds for r=8 and r=32). Ready to select best model")
        print("  by calibration-set F1 and report test_seen ONCE.")
    else:
        missing_r8  = max(0, 3 - (n_r8  or 0))
        missing_r32 = max(0, 3 - (n_r32 or 0))
        print(f"⚠ Seed sweep incomplete: r=8 has {n_r8}/5 seeds, r=32 has {n_r32}/5 seeds.")
        if missing_r8 > 0:
            print(f"  → Run: sbatch gemma3_residual_sae/slurm/probe_fhm_bilinear_seed_sweep_r8.sub")
        if missing_r32 > 0:
            print(f"  → Run: sbatch gemma3_residual_sae/slurm/probe_fhm_bilinear_seed_sweep_r32.sub")
        print("  Do NOT report test_seen until at least 3 seeds are complete.")


if __name__ == "__main__":
    main()
