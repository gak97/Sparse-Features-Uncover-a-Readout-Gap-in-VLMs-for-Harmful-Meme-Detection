

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from simple_parsing import ArgumentGenerationMode, parse

logger = logging.getLogger(__name__)

TARGET_SEP_TOKENS = ["yes", "no", "hateful", "hate", "offensive", "harmful", "racist", "slur"]


@dataclass
class JlensPerSampleConfig:
    # J-lens output dir (from compute_jacobian_lens.py)
    jlens_dir: str = ""
    # Layers to analyse (must match files in jlens_dir)
    layer_indices: list[int] = field(default_factory=lambda: [22, 29])

    # IVL FHM metadata for pseudo-pair structure
    fhm_meta_path: str = ""

    # Base-model 12B yes/no predictions for error correlation
    # (residual_sae_predictions.12b_gemma.*.yes_no.jsonl)
    baseline_predictions_path: str = ""

    # Optional: 4B baseline predictions for comparison
    baseline_4b_predictions_path: str = ""

    output_dir: str = ""
    make_plots: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_per_sample(jlens_dir: Path) -> list[dict[str, Any]]:
    p = jlens_dir / "jlens_per_sample.jsonl"
    assert p.exists(), f"Missing {p}  — run compute_jacobian_lens.py first"
    rows = _read_jsonl(p)
    logger.info("Loaded %d per-sample J-lens records from %s", len(rows), p)
    return rows


def _normalise_id(sid: str) -> str:
    return sid.split("/")[-1].split(".")[0]


def _load_pseudo_pairs(fhm_meta_path: str) -> tuple[dict, dict]:
    """Return (img_groups, txt_groups): group_id → list of {sample_id, gold_label}."""
    with open(fhm_meta_path) as f:
        meta_list = json.load(f)

    img_groups: dict[Any, list[dict]] = {}
    txt_groups: dict[Any, list[dict]] = {}
    for m in meta_list:
        sid = str(m.get("id", ""))
        label = str(m.get("label", ""))
        img_grp = m.get("pseudo_img_idx")
        txt_grp = m.get("pseudo_text_idx")
        if img_grp is not None:
            img_groups.setdefault(img_grp, []).append({"sample_id": sid, "gold_label": label})
        if txt_grp is not None:
            txt_groups.setdefault(txt_grp, []).append({"sample_id": sid, "gold_label": label})

    return img_groups, txt_groups


def _build_cross_label_pairs(groups: dict) -> list[tuple[dict, dict]]:
    """For each group with ≥2 labels: return (hateful_member, benign_member) pairs."""
    pairs: list[tuple[dict, dict]] = []
    for grp_members in groups.values():
        hateful = [m for m in grp_members if m["gold_label"] in ("hateful", "1", "hate speech")]
        benign = [m for m in grp_members if m["gold_label"] not in ("hateful", "1", "hate speech")]
        if hateful and benign:
            for h in hateful:
                for b in benign:
                    pairs.append((h, b))
    return pairs


def _load_baseline_predictions(path: str) -> dict[str, dict]:
    """Load baseline predictions. Returns {sample_id_normalised: row}."""
    if not path:
        return {}
    result: dict[str, dict] = {}
    for row in _read_jsonl(Path(path)):
        if row.get("dataset_name") not in ("Hateful_Memes_Dataset", None):
            continue
        if row.get("dataset_name") and row["dataset_name"] != "Hateful_Memes_Dataset":
            continue
        sid = _normalise_id(str(row.get("sample_id", "")))
        result[sid] = row
    logger.info("Loaded %d baseline predictions from %s", len(result), path)
    return result


def _get_jnorm(rec: dict | None, layer_idx: int, token: str) -> float | None:
    if rec is None:
        return None
    ldata = rec.get("layers", {}).get(str(layer_idx))
    if ldata is None:
        return None
    return ldata.get("per_token_norm", {}).get(token)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 1: Pair separation by J-norm
# ─────────────────────────────────────────────────────────────────────────────

def _pair_separation_analysis(
    per_sample: list[dict],
    img_groups: dict,
    txt_groups: dict,
    layer_indices: list[int],
    sep_tokens: list[str],
) -> dict[str, Any]:
    # Build sample_id → record map
    by_id: dict[str, dict] = {_normalise_id(r["sample_id"]): r for r in per_sample}

    results: dict[str, Any] = {}
    for layer_idx in layer_indices:
        layer_res: dict[str, Any] = {}
        for source_name, groups in [("pseudo_img", img_groups), ("pseudo_text", txt_groups)]:
            pairs = _build_cross_label_pairs(groups)
            token_stats: dict[str, dict] = {}
            for tok in sep_tokens:
                margins = []
                positive = 0
                for h_member, b_member in pairs:
                    h_sid = _normalise_id(h_member["sample_id"])
                    b_sid = _normalise_id(b_member["sample_id"])
                    h_rec = by_id.get(h_sid)
                    b_rec = by_id.get(b_sid)
                    if h_rec is None or b_rec is None:
                        continue
                    h_norm = _get_jnorm(h_rec, layer_idx, tok)
                    b_norm = _get_jnorm(b_rec, layer_idx, tok)
                    if h_norm is None or b_norm is None:
                        continue
                    margin = h_norm - b_norm
                    margins.append(margin)
                    if margin > 0:
                        positive += 1

                if margins:
                    token_stats[tok] = {
                        "n_pairs": len(margins),
                        "positive_margin_rate": positive / len(margins),
                        "mean_margin": float(np.mean(margins)),
                        "median_margin": float(np.median(margins)),
                    }

            # Best discriminative token by positive_margin_rate
            best_tok = max(
                token_stats.items(),
                key=lambda x: x[1]["positive_margin_rate"],
                default=(None, {}),
            )
            layer_res[source_name] = {
                "per_token": token_stats,
                "best_discriminative_token": best_tok[0],
                "best_positive_margin_rate": best_tok[1].get("positive_margin_rate", 0.0),
            }
        results[str(layer_idx)] = layer_res

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 2: J-norm vs base-model error
# ─────────────────────────────────────────────────────────────────────────────

def _error_correlation_analysis(
    per_sample: list[dict],
    baseline_preds: dict[str, dict],
    layer_indices: list[int],
    sep_tokens: list[str],
) -> dict[str, Any]:
    if not baseline_preds:
        return {"skipped": True, "reason": "No baseline predictions provided"}

    results: dict[str, Any] = {}
    for layer_idx in layer_indices:
        layer_res: dict[str, Any] = {}
        for tok in sep_tokens:
            # Groups: hateful gold, base correct vs wrong; benign gold, base correct vs wrong
            buckets: dict[str, list[float]] = {
                "hateful_base_correct": [],
                "hateful_base_wrong": [],
                "benign_base_correct": [],
                "benign_base_wrong": [],
            }
            for rec in per_sample:
                sid = _normalise_id(rec["sample_id"])
                gold = rec.get("gold_label", "")
                pred_row = baseline_preds.get(sid)
                if pred_row is None:
                    continue

                j_norm = _get_jnorm(rec, layer_idx, tok)
                if j_norm is None:
                    continue

                # Base model prediction
                pred_label = (
                    pred_row.get("prediction_label")
                    or pred_row.get("is_positive")
                )
                if isinstance(pred_label, bool):
                    pred_label = "hateful" if pred_label else "not hateful"

                is_hateful_gold = gold in ("hateful", "1")
                is_hateful_pred = str(pred_label).lower() in ("hateful", "true", "1", "yes")
                base_correct = is_hateful_gold == is_hateful_pred

                if is_hateful_gold:
                    buckets["hateful_base_correct" if base_correct else "hateful_base_wrong"].append(j_norm)
                else:
                    buckets["benign_base_correct" if base_correct else "benign_base_wrong"].append(j_norm)

            tok_stats: dict[str, Any] = {}
            for bucket, vals in buckets.items():
                tok_stats[bucket] = {
                    "n": len(vals),
                    "mean": float(np.mean(vals)) if vals else None,
                    "median": float(np.median(vals)) if vals else None,
                }

            # Delta: hateful_correct - hateful_wrong (positive = higher J when model is right)
            h_correct = [v for v in buckets["hateful_base_correct"]]
            h_wrong = [v for v in buckets["hateful_base_wrong"]]
            delta = (
                (float(np.mean(h_correct)) - float(np.mean(h_wrong)))
                if h_correct and h_wrong else None
            )
            tok_stats["delta_hateful_correct_minus_wrong"] = delta
            tok_stats["interpretation"] = (
                "higher J-norm when base model is correct (supports causal connection)"
                if delta is not None and delta > 0
                else "lower J-norm when base model is correct (or no signal)"
                if delta is not None else "insufficient data"
            )
            layer_res[tok] = tok_stats

        results[str(layer_idx)] = layer_res

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _plot_pair_scatter(
    per_sample: list[dict],
    img_groups: dict,
    txt_groups: dict,
    layer_idx: int,
    token: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    by_id = {_normalise_id(r["sample_id"]): r for r in per_sample}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (src_name, groups) in zip(axes, [("pseudo_img", img_groups), ("pseudo_text", txt_groups)]):
        pairs = _build_cross_label_pairs(groups)
        h_vals, b_vals = [], []
        for h_m, b_m in pairs:
            h_n = _get_jnorm(by_id.get(_normalise_id(h_m["sample_id"])), layer_idx, token)
            b_n = _get_jnorm(by_id.get(_normalise_id(b_m["sample_id"])), layer_idx, token)
            if h_n is not None and b_n is not None:
                h_vals.append(h_n)
                b_vals.append(b_n)

        h_vals_np = np.array(h_vals)
        b_vals_np = np.array(b_vals)
        pos_rate = float((h_vals_np > b_vals_np).mean()) if len(h_vals) else 0.0

        ax.scatter(b_vals_np, h_vals_np, s=10, alpha=0.5, c="steelblue")
        lim = max(h_vals_np.max(), b_vals_np.max()) * 1.05 if len(h_vals) else 1
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel(f"J-norm({token}) — benign member")
        ax.set_ylabel(f"J-norm({token}) — hateful member")
        ax.set_title(f"{src_name}  layer={layer_idx}\nPMR={pos_rate:.3f}  n={len(h_vals)}")

    fig.suptitle(f"J-lens pair separation: token='{token}' layer={layer_idx}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _plot_error_jnorm(
    per_sample: list[dict],
    baseline_preds: dict[str, dict],
    layer_idx: int,
    token: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    buckets: dict[str, list[float]] = {
        "hateful correct": [], "hateful wrong": [],
        "benign correct": [], "benign wrong": [],
    }
    for rec in per_sample:
        sid = _normalise_id(rec["sample_id"])
        gold = rec.get("gold_label", "")
        pred_row = baseline_preds.get(sid)
        if pred_row is None:
            continue
        j_norm = _get_jnorm(rec, layer_idx, token)
        if j_norm is None:
            continue
        pred_label = pred_row.get("prediction_label") or pred_row.get("is_positive")
        if isinstance(pred_label, bool):
            pred_label = "hateful" if pred_label else "not hateful"
        is_h_gold = gold in ("hateful", "1")
        is_h_pred = str(pred_label).lower() in ("hateful", "true", "1", "yes")
        correct = is_h_gold == is_h_pred
        key = f"{'hateful' if is_h_gold else 'benign'} {'correct' if correct else 'wrong'}"
        buckets[key].append(j_norm)

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(buckets.keys())
    data = [buckets[l] for l in labels]
    ax.boxplot(data, labels=[f"{l}\n(n={len(d)})" for l, d in zip(labels, data)])
    ax.set_ylabel(f"J-norm({token}) at layer {layer_idx}")
    ax.set_title(f"J-norm vs base-model error\ntoken='{token}' layer={layer_idx}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _run(cfg: JlensPerSampleConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jlens_dir = Path(cfg.jlens_dir)

    per_sample = _load_per_sample(jlens_dir)

    # Load pseudo-pair structure from IVL metadata
    img_groups: dict = {}
    txt_groups: dict = {}
    if cfg.fhm_meta_path:
        img_groups, txt_groups = _load_pseudo_pairs(cfg.fhm_meta_path)
        img_pairs = _build_cross_label_pairs(img_groups)
        txt_pairs = _build_cross_label_pairs(txt_groups)
        logger.info(
            "IVL pseudo-pairs: %d img pairs, %d text pairs",
            len(img_pairs), len(txt_pairs),
        )

    # Load baseline predictions
    baseline_preds = _load_baseline_predictions(cfg.baseline_predictions_path)
    baseline_4b_preds = _load_baseline_predictions(cfg.baseline_4b_predictions_path)

    # Determine which layers are available
    summary = json.loads((jlens_dir / "jlens_summary.json").read_text())
    available_layers = [int(k) for k in summary.get("per_layer", {}).keys()]
    layer_indices = [l for l in cfg.layer_indices if l in available_layers]
    if not layer_indices:
        layer_indices = available_layers
    logger.info("Analysing layers: %s", layer_indices)

    # ── Analysis 1: pair separation ──────────────────────────────────────────
    pair_sep = {}
    if img_groups or txt_groups:
        pair_sep = _pair_separation_analysis(
            per_sample, img_groups, txt_groups, layer_indices, TARGET_SEP_TOKENS
        )
        (out_dir / "jlens_pair_separation.json").write_text(json.dumps(pair_sep, indent=2))
        logger.info("Pair separation written to %s", out_dir / "jlens_pair_separation.json")

        if cfg.make_plots:
            for layer_idx in layer_indices:
                # Use "yes" as the primary separation token (clearest for constrained decode)
                _plot_pair_scatter(
                    per_sample, img_groups, txt_groups, layer_idx,
                    token="yes",
                    out_path=out_dir / f"scatter_pair_jnorm_yes_l{layer_idx}.png",
                )
                # Also plot best discriminative token if different
                for src in ("pseudo_img", "pseudo_text"):
                    best = pair_sep.get(str(layer_idx), {}).get(src, {}).get("best_discriminative_token")
                    if best and best != "yes":
                        _plot_pair_scatter(
                            per_sample, img_groups, txt_groups, layer_idx,
                            token=best,
                            out_path=out_dir / f"scatter_pair_jnorm_{best}_l{layer_idx}.png",
                        )

    # ── Analysis 2: error correlation ────────────────────────────────────────
    error_corr: dict[str, Any] = {}
    for tag, preds in [("12b", baseline_preds), ("4b", baseline_4b_preds)]:
        if not preds:
            continue
        corr = _error_correlation_analysis(per_sample, preds, layer_indices, TARGET_SEP_TOKENS)
        error_corr[tag] = corr
        if cfg.make_plots:
            for layer_idx in layer_indices:
                _plot_error_jnorm(
                    per_sample, preds, layer_idx, "yes",
                    out_path=out_dir / f"scatter_error_jnorm_yes_{tag}_l{layer_idx}.png",
                )

    (out_dir / "jlens_error_correlation.json").write_text(json.dumps(error_corr, indent=2))

    # ── Summary per layer ────────────────────────────────────────────────────
    overall_summary: dict[str, Any] = {
        "jlens_dir": str(jlens_dir),
        "n_samples": len(per_sample),
        "layers_analysed": layer_indices,
        "pair_separation_highlights": {},
        "error_correlation_highlights": {},
    }

    for layer_idx in layer_indices:
        ps_layer = pair_sep.get(str(layer_idx), {})
        best_img = ps_layer.get("pseudo_img", {}).get("best_positive_margin_rate", 0.0)
        best_txt = ps_layer.get("pseudo_text", {}).get("best_positive_margin_rate", 0.0)
        yes_img = ps_layer.get("pseudo_img", {}).get("per_token", {}).get("yes", {}).get("positive_margin_rate")
        yes_txt = ps_layer.get("pseudo_text", {}).get("per_token", {}).get("yes", {}).get("positive_margin_rate")

        overall_summary["pair_separation_highlights"][str(layer_idx)] = {
            "best_img_pmr": best_img,
            "best_txt_pmr": best_txt,
            "yes_img_pmr": yes_img,
            "yes_txt_pmr": yes_txt,
        }

        for tag in ("12b", "4b"):
            if tag not in error_corr:
                continue
            yes_corr = error_corr[tag].get(str(layer_idx), {}).get("yes", {})
            delta = yes_corr.get("delta_hateful_correct_minus_wrong")
            overall_summary["error_correlation_highlights"].setdefault(tag, {})[str(layer_idx)] = {
                "yes_delta_correct_minus_wrong": delta,
                "interpretation": yes_corr.get("interpretation", ""),
            }

    (out_dir / "jlens_per_sample_summary.json").write_text(
        json.dumps(overall_summary, indent=2)
    )
    logger.info("Done. Outputs written to %s", out_dir)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse(JlensPerSampleConfig, argument_generation_mode=ArgumentGenerationMode.FLAT)
    _run(cfg)


if __name__ == "__main__":
    main()
