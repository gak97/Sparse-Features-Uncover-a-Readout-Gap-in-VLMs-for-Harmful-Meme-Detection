

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from simple_parsing import ArgumentGenerationMode, parse

logger = logging.getLogger(__name__)


@dataclass
class ProbeVsJlensConfig:
    # Per-sample J-lens dir (from compute_jacobian_lens.py)
    jlens_dir: str = ""
    # Layer to use for J-norm
    layer_idx: int = 22
    # J-lens token(s) to compute margin for
    jlens_tokens: list[str] = field(default_factory=lambda: ["yes", "hate"])

    # Probe validation predictions JSONL (must have probe_proba field)
    probe_predictions_path: str = ""

    # IVL FHM metadata for pseudo-pair structure
    fhm_meta_path: str = ""

    output_dir: str = ""
    make_plots: bool = True


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalise_id(sid: str) -> str:
    return str(sid).split("/")[-1].split(".")[0]


def _load_pseudo_pairs(fhm_meta_path: str) -> dict[str, list[tuple[dict, dict]]]:
    """Return {source_name: [(hateful_member, benign_member)]} for cross-label pairs."""
    with open(fhm_meta_path) as f:
        meta_list = json.load(f)

    img_groups: dict[Any, list[dict]] = {}
    txt_groups: dict[Any, list[dict]] = {}
    for m in meta_list:
        sid = str(m.get("id", ""))
        label = str(m.get("label", ""))
        if m.get("pseudo_img_idx") is not None:
            img_groups.setdefault(m["pseudo_img_idx"], []).append(
                {"sample_id": sid, "gold_label": label}
            )
        if m.get("pseudo_text_idx") is not None:
            txt_groups.setdefault(m["pseudo_text_idx"], []).append(
                {"sample_id": sid, "gold_label": label}
            )

    result: dict[str, list[tuple[dict, dict]]] = {}
    for src_name, groups in [("pseudo_img", img_groups), ("pseudo_text", txt_groups)]:
        pairs: list[tuple[dict, dict]] = []
        for members in groups.values():
            hateful = [m for m in members if m["gold_label"] in ("hateful", "1")]
            benign = [m for m in members if m["gold_label"] not in ("hateful", "1")]
            for h in hateful:
                for b in benign:
                    pairs.append((h, b))
        result[src_name] = pairs
    return result


def _load_jlens_by_id(jlens_dir: Path, layer_idx: int) -> dict[str, dict[str, float]]:
    """Return {normalised_sample_id: {token: jnorm}} for the given layer."""
    rows = _read_jsonl(jlens_dir / "jlens_per_sample.jsonl")
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        sid = _normalise_id(row["sample_id"])
        ldata = row.get("layers", {}).get(str(layer_idx))
        if ldata is None:
            continue
        result[sid] = ldata.get("per_token_norm", {})
    return result


def _load_probe_proba_by_id(probe_path: Path) -> dict[str, dict[str, float]]:
    """Return {normalised_sample_id: {class_label: probability}}."""
    rows = _read_jsonl(probe_path)
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        # Derive sample_id from sample_key if available
        key = row.get("sample_key", row.get("sample_id", ""))
        # sample_key format: "Hateful_Memes_Dataset::task::id"
        sid = _normalise_id(key.split("::")[-1] if "::" in str(key) else str(key))
        proba = row.get("probe_proba")
        if isinstance(proba, dict):
            result[sid] = proba
        elif isinstance(proba, (float, int)):
            # Binary scalar — interpret as P(hateful)
            result[sid] = {"hateful": float(proba), "not hateful": 1.0 - float(proba)}
    return result


def _analyse_pairs(
    pairs: list[tuple[dict, dict]],
    jlens_by_id: dict[str, dict[str, float]],
    probe_by_id: dict[str, dict[str, float]],
    jlens_tokens: list[str],
) -> list[dict[str, Any]]:
    """Compare probe margin vs J-norm margin for each matched pair."""
    records: list[dict[str, Any]] = []
    for h_member, b_member in pairs:
        h_sid = _normalise_id(h_member["sample_id"])
        b_sid = _normalise_id(b_member["sample_id"])

        h_jlens = jlens_by_id.get(h_sid)
        b_jlens = jlens_by_id.get(b_sid)
        h_probe = probe_by_id.get(h_sid)
        b_probe = probe_by_id.get(b_sid)

        if h_jlens is None or b_jlens is None:
            continue  # pair not in J-lens sample (only 200 sampled)
        if h_probe is None or b_probe is None:
            continue  # pair not in probe predictions

        # Probe margin: P(hateful | hateful_member) − P(hateful | benign_member)
        h_p = h_probe.get("hateful", 0.5)
        b_p = b_probe.get("hateful", 0.5)
        probe_margin = float(h_p - b_p)

        rec: dict[str, Any] = {
            "hateful_id": h_sid,
            "benign_id": b_sid,
            "probe_margin": probe_margin,
            "probe_correct": probe_margin > 0,
            "h_probe_proba": h_p,
            "b_probe_proba": b_p,
        }

        for tok in jlens_tokens:
            h_jn = h_jlens.get(tok)
            b_jn = b_jlens.get(tok)
            if h_jn is not None and b_jn is not None:
                jlens_margin = float(h_jn - b_jn)
                rec[f"jlens_margin_{tok}"] = jlens_margin
                rec[f"jlens_correct_{tok}"] = jlens_margin > 0

        records.append(rec)
    return records


def _quadrant_summary(
    records: list[dict[str, Any]],
    primary_token: str = "yes",
) -> dict[str, Any]:
    """Compute quadrant counts for probe vs J-lens."""
    margin_key = f"jlens_margin_{primary_token}"
    matched = [r for r in records if margin_key in r]
    if not matched:
        return {"n_matched": 0, "error": f"No records with {margin_key}"}

    n = len(matched)
    both_correct = sum(1 for r in matched if r["probe_correct"] and r.get(f"jlens_correct_{primary_token}"))
    probe_only = sum(1 for r in matched if r["probe_correct"] and not r.get(f"jlens_correct_{primary_token}"))
    jlens_only = sum(1 for r in matched if not r["probe_correct"] and r.get(f"jlens_correct_{primary_token}"))
    both_miss = sum(1 for r in matched if not r["probe_correct"] and not r.get(f"jlens_correct_{primary_token}"))

    probe_margins = np.array([r["probe_margin"] for r in matched])
    jlens_margins = np.array([r[margin_key] for r in matched])
    correlation = float(np.corrcoef(probe_margins, jlens_margins)[0, 1]) if n > 2 else float("nan")

    return {
        "n_matched_pairs": n,
        "primary_token": primary_token,
        "quadrant_counts": {
            "both_correct": both_correct,
            "probe_only": probe_only,
            "jlens_only": jlens_only,
            "both_miss": both_miss,
        },
        "quadrant_rates": {
            "both_correct": both_correct / n,
            "probe_only": probe_only / n,
            "jlens_only": jlens_only / n,
            "both_miss": both_miss / n,
        },
        "probe_pmr": sum(1 for r in matched if r["probe_correct"]) / n,
        "jlens_pmr": sum(1 for r in matched if r.get(f"jlens_correct_{primary_token}")) / n,
        "pearson_correlation_probe_jlens_margin": correlation,
        "interpretation": (
            "probe_only >> jlens_only → silent knowledge confirmed (probe exploits "
            "directions model does not route)" if probe_only > jlens_only * 1.5
            else "jlens_only ≥ probe_only → J-lens as discriminative as probe"
            if jlens_only >= probe_only else "roughly symmetric complementarity"
        ),
    }


def _plot_scatter(
    records: list[dict[str, Any]],
    primary_token: str,
    source_name: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    margin_key = f"jlens_margin_{primary_token}"
    matched = [r for r in records if margin_key in r]
    if not matched:
        return

    xs = np.array([r[margin_key] for r in matched])
    ys = np.array([r["probe_margin"] for r in matched])
    colors = ["green" if r["probe_correct"] and r.get(f"jlens_correct_{primary_token}")
              else "red" if r["probe_correct"] else "blue" if r.get(f"jlens_correct_{primary_token}")
              else "gray" for r in matched]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(xs, ys, c=colors, s=20, alpha=0.7)
    ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel(f"J-norm margin ({primary_token}) — hateful minus benign")
    ax.set_ylabel("Probe margin — P(hateful|h) minus P(hateful|b)")
    corr = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else float("nan")
    ax.set_title(
        f"{source_name} / token={primary_token}  n={len(matched)}\n"
        f"r={corr:.3f}  "
        f"green=both  red=probe_only  blue=jlens_only  gray=both_miss"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _plot_quadrant_bar(
    summary_by_source: dict[str, dict[str, Any]],
    primary_token: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    sources = list(summary_by_source.keys())
    quadrants = ["both_correct", "probe_only", "jlens_only", "both_miss"]
    colors = ["#2ca02c", "#d62728", "#1f77b4", "#7f7f7f"]
    x = np.arange(len(sources))
    width = 0.2

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (q, col) in enumerate(zip(quadrants, colors)):
        vals = [summary_by_source[s].get("quadrant_rates", {}).get(q, 0.0) for s in sources]
        ax.bar(x + i * width, vals, width, label=q, color=col, alpha=0.8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(sources)
    ax.set_ylabel("Pair fraction")
    ax.set_title(f"Probe vs J-lens pair quadrants (token={primary_token})")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _run(cfg: ProbeVsJlensConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jlens_dir = Path(cfg.jlens_dir)
    assert jlens_dir.exists(), f"J-lens dir not found: {jlens_dir}"

    # Load inputs
    jlens_by_id = _load_jlens_by_id(jlens_dir, cfg.layer_idx)
    logger.info("J-lens: %d samples at layer %d", len(jlens_by_id), cfg.layer_idx)

    probe_by_id = _load_probe_proba_by_id(Path(cfg.probe_predictions_path))
    logger.info("Probe: %d sample probabilities loaded", len(probe_by_id))

    pairs_by_source = _load_pseudo_pairs(cfg.fhm_meta_path)

    # Per-source analysis
    all_records: dict[str, list[dict]] = {}
    all_summaries: dict[str, dict[str, Any]] = {}

    primary_token = cfg.jlens_tokens[0] if cfg.jlens_tokens else "yes"

    for src_name, pairs in pairs_by_source.items():
        records = _analyse_pairs(pairs, jlens_by_id, probe_by_id, cfg.jlens_tokens)
        all_records[src_name] = records

        for tok in cfg.jlens_tokens:
            key = f"{src_name}/{tok}"
            summary = _quadrant_summary(records, primary_token=tok)
            all_summaries[key] = summary
            logger.info(
                "%s/%s: n=%d  probe_pmr=%.3f  jlens_pmr=%.3f  "
                "both_correct=%d  probe_only=%d  jlens_only=%d  both_miss=%d  r=%.3f",
                src_name, tok,
                summary.get("n_matched_pairs", 0),
                summary.get("probe_pmr", 0),
                summary.get("jlens_pmr", 0),
                summary.get("quadrant_counts", {}).get("both_correct", 0),
                summary.get("quadrant_counts", {}).get("probe_only", 0),
                summary.get("quadrant_counts", {}).get("jlens_only", 0),
                summary.get("quadrant_counts", {}).get("both_miss", 0),
                summary.get("pearson_correlation_probe_jlens_margin", float("nan")),
            )

        if cfg.make_plots:
            for tok in cfg.jlens_tokens:
                _plot_scatter(
                    records, tok, src_name,
                    out_dir / f"scatter_probe_vs_jlens_{src_name}_{tok}.png",
                )

    # Combined summary
    if cfg.make_plots:
        for tok in cfg.jlens_tokens:
            src_summaries = {
                src: all_summaries.get(f"{src}/{tok}", {})
                for src in pairs_by_source
            }
            _plot_quadrant_bar(
                src_summaries, tok,
                out_dir / f"quadrant_bar_{tok}.png",
            )

    # Write outputs
    (out_dir / "probe_vs_jlens_quadrant_summary.json").write_text(
        json.dumps(all_summaries, indent=2)
    )
    # Write all matched pair records per source
    pair_output: dict[str, Any] = {}
    for src_name, records in all_records.items():
        pair_output[src_name] = records
    (out_dir / "probe_vs_jlens_pair_comparison.json").write_text(
        json.dumps(pair_output, indent=2)
    )
    logger.info("Done — outputs written to %s", out_dir)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse(ProbeVsJlensConfig, argument_generation_mode=ArgumentGenerationMode.FLAT)
    _run(cfg)


if __name__ == "__main__":
    main()
