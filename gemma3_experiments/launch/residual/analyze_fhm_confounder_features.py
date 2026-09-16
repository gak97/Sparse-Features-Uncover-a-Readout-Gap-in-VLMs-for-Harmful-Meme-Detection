

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.launch.residual.probe_validate import (
    _extract_features,
    _extract_features_multi,
    _load_fhm_confounder_metadata,
    _normalise_fhm_id,
    _normalise_group_value,
    _read_jsonl,
    _row_gold_label,
    _row_sample_key,
)
from gemma3_experiments.residual_sae import (
    load_residual_sae,
    load_residual_stats,
    resolve_visible_device,
)

logger = logging.getLogger(__name__)


@dataclass
class FHMConfounderFeatureAuditConfig:
    residual_sae_path: str = ""
    train_cache_dir: str = ""
    val_cache_dir: str = ""
    output_dir: str = ""
    fhm_confounder_metadata_path: str = ""

    token_selections: list[str] = field(default_factory=lambda: ["all", "image", "prompt", "generated"])
    aggregation: str = "max"
    batch_size: int = 256
    top_k_features: int = 50
    transfer_top_k: list[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    compare_top_features_json: str = "./outputs/residual_sae_probe_results_harm_binary/top_features.json"
    device: str = "cuda:0"


def _load_fhm_rows(cache_dir: Path, split_names: set[str], *, cache_tag: bool = False) -> list[dict[str, Any]]:
    rows = []
    for row in _read_jsonl(cache_dir / "manifest.jsonl"):
        if row.get("dataset_name") != "Hateful_Memes_Dataset":
            continue
        if row.get("task_name") != "hateful_memes_binary":
            continue
        if row.get("split") not in split_names:
            continue
        if _row_gold_label(row) not in {"hateful", "not hateful"}:
            continue
        if cache_tag:
            row = dict(row)
            row["_cache_dir"] = str(cache_dir)
        rows.append(row)
    return rows


def _source_to_col(source: str) -> str:
    if source == "pseudo_img":
        return "pseudo_img_idx"
    if source == "pseudo_text":
        return "pseudo_text_idx"
    raise ValueError(f"Unsupported source: {source}")


def _build_pairs(
    rows: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    available_keys: set[str],
    source: str,
) -> list[tuple[str, str]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    col = _source_to_col(source)
    for row in rows:
        sample_key = _row_sample_key(row)
        if sample_key not in available_keys:
            continue
        meta = metadata.get(_normalise_fhm_id(row.get("sample_id")))
        if meta is None:
            continue
        group_value = _normalise_group_value(meta.get(col))
        if group_value is not None:
            groups[group_value].append(row)

    pairs = []
    seen_pairs: set[tuple[str, str]] = set()
    for group_rows in groups.values():
        hateful_rows = [row for row in group_rows if _row_gold_label(row) == "hateful"]
        benign_rows = [row for row in group_rows if _row_gold_label(row) == "not hateful"]
        for hateful_row in hateful_rows:
            hateful_key = _row_sample_key(hateful_row)
            for benign_row in benign_rows:
                benign_key = _row_sample_key(benign_row)
                pair_key = (hateful_key, benign_key)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                pairs.append(pair_key)
    return pairs


def _feature_stats(csr: Any, key_to_idx: dict[str, int], pairs: list[tuple[str, str]], top_k: int) -> dict[str, Any]:
    if not pairs:
        return {
            "n_pairs": 0,
            "top_features": [],
            "arrays": {},
        }

    hateful_i = np.array([key_to_idx[hateful_key] for hateful_key, _ in pairs], dtype=np.int32)
    benign_i = np.array([key_to_idx[benign_key] for _, benign_key in pairs], dtype=np.int32)
    delta = (csr[hateful_i] - csr[benign_i]).tocsr().astype(np.float32)
    n_pairs = delta.shape[0]

    mean_delta = np.asarray(delta.mean(axis=0)).ravel().astype(np.float64)
    mean_sq_delta = np.asarray(delta.multiply(delta).mean(axis=0)).ravel().astype(np.float64)
    var_delta = np.maximum(mean_sq_delta - np.square(mean_delta), 0.0)
    std_delta = np.sqrt(var_delta)
    effect = mean_delta / (std_delta + 1e-8)

    pos_counts = np.asarray((delta > 0).sum(axis=0)).ravel().astype(np.float64)
    neg_counts = np.asarray((delta < 0).sum(axis=0)).ravel().astype(np.float64)
    nonzero_counts = pos_counts + neg_counts
    pos_frac = pos_counts / n_pairs
    neg_frac = neg_counts / n_pairs
    sign_consistency = np.maximum(pos_frac, neg_frac)
    coverage = nonzero_counts / n_pairs
    score = np.abs(effect) * sign_consistency * np.sqrt(np.maximum(coverage, 1e-8))

    order = np.argsort(score)[::-1]
    top_features = []
    for feature_i in order[:top_k]:
        direction = "hateful_up" if mean_delta[feature_i] > 0 else "benign_up"
        top_features.append({
            "feature_idx": int(feature_i),
            "score": round(float(score[feature_i]), 6),
            "effect": round(float(effect[feature_i]), 6),
            "mean_delta": round(float(mean_delta[feature_i]), 6),
            "std_delta": round(float(std_delta[feature_i]), 6),
            "sign_consistency": round(float(sign_consistency[feature_i]), 6),
            "coverage": round(float(coverage[feature_i]), 6),
            "pos_frac": round(float(pos_frac[feature_i]), 6),
            "neg_frac": round(float(neg_frac[feature_i]), 6),
            "direction": direction,
        })

    return {
        "n_pairs": int(n_pairs),
        "top_features": top_features,
        "arrays": {
            "mean_delta": mean_delta,
            "effect": effect,
            "sign_consistency": sign_consistency,
            "coverage": coverage,
            "score": score,
        },
    }


def _transfer_stats(
    train_stats: dict[str, Any],
    val_csr: Any,
    val_key_to_idx: dict[str, int],
    val_pairs: list[tuple[str, str]],
    top_k_values: list[int],
) -> list[dict[str, Any]]:
    if not val_pairs or not train_stats["top_features"]:
        return []

    hateful_i = np.array([val_key_to_idx[hateful_key] for hateful_key, _ in val_pairs], dtype=np.int32)
    benign_i = np.array([val_key_to_idx[benign_key] for _, benign_key in val_pairs], dtype=np.int32)
    delta = (val_csr[hateful_i] - val_csr[benign_i]).tocsr().astype(np.float32)

    out = []
    max_top_k = min(max(top_k_values), len(train_stats["top_features"]))
    feature_i = np.array([row["feature_idx"] for row in train_stats["top_features"][:max_top_k]], dtype=np.int32)
    direction = np.array([1.0 if row["mean_delta"] > 0 else -1.0 for row in train_stats["top_features"][:max_top_k]], dtype=np.float32)

    for top_k in top_k_values:
        top_k = min(top_k, max_top_k)
        if top_k <= 0:
            continue
        sub_delta = delta[:, feature_i[:top_k]].toarray()
        scores = sub_delta @ direction[:top_k]
        out.append({
            "top_k": int(top_k),
            "n_val_pairs": int(len(val_pairs)),
            "positive_score_rate": round(float(np.mean(scores > 0)), 6),
            "mean_score": round(float(np.mean(scores)), 6),
            "median_score": round(float(np.median(scores)), 6),
            "q25_score": round(float(np.quantile(scores, 0.25)), 6),
            "q75_score": round(float(np.quantile(scores, 0.75)), 6),
        })
    return out


def _score_pairs(
    train_stats: dict[str, Any],
    csr: Any,
    key_to_idx: dict[str, int],
    pairs: list[tuple[str, str]],
    top_k: int,
) -> list[dict[str, Any]]:
    if not pairs or not train_stats["top_features"]:
        return []

    top_k = min(top_k, len(train_stats["top_features"]))
    feature_i = np.array([row["feature_idx"] for row in train_stats["top_features"][:top_k]], dtype=np.int32)
    direction = np.array([1.0 if row["mean_delta"] > 0 else -1.0 for row in train_stats["top_features"][:top_k]], dtype=np.float32)
    hateful_i = np.array([key_to_idx[hateful_key] for hateful_key, _ in pairs], dtype=np.int32)
    benign_i = np.array([key_to_idx[benign_key] for _, benign_key in pairs], dtype=np.int32)
    delta = (csr[hateful_i] - csr[benign_i]).tocsr().astype(np.float32)
    scores = delta[:, feature_i].toarray() @ direction

    return [
        {
            "hateful_key": hateful_key,
            "benign_key": benign_key,
            "top_k": int(top_k),
            "score": round(float(score), 6),
            "hateful_above_benign": bool(score > 0),
        }
        for (hateful_key, benign_key), score in zip(pairs, scores)
    ]


def _load_lr_top_features(path: Path) -> set[int]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    task_key = "Hateful_Memes_Dataset::hateful_memes_binary::label"
    if task_key not in data:
        return set()
    feature_indices = set()
    for rows in data[task_key].values():
        feature_indices.update(int(row["feature_idx"]) for row in rows)
    return feature_indices


def _plot_scatter(summary: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    for token_selection, by_source in summary["audits"].items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, source in zip(axes, ["pseudo_img", "pseudo_text"]):
            if by_source[source]["n_pairs"] == 0:
                ax.set_title(f"{token_selection} / {source}: no pairs")
                ax.axis("off")
                continue
            arrays = by_source[source]["arrays"]
            effect = arrays["effect"]
            consistency = arrays["sign_consistency"]
            coverage = arrays["coverage"]
            ax.scatter(np.abs(effect), consistency, s=np.maximum(coverage, 0.01) * 30, alpha=0.25)
            for row in by_source[source]["top_features"][:10]:
                feature_i = row["feature_idx"]
                ax.text(abs(effect[feature_i]), consistency[feature_i], f"f{feature_i}", fontsize=7)
            ax.set_title(f"{token_selection} / {source}")
            ax.set_xlabel("abs paired effect")
            ax.set_ylabel("sign consistency")
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"scatter_effect_consistency_{token_selection}.png", dpi=150)
        plt.close(fig)


def _plot_transfer(summary: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = []
    rates = []
    for token_selection, by_source in summary["audits"].items():
        for source, audit in by_source.items():
            for row in audit["transfer"]:
                if row["top_k"] == 20:
                    labels.append(f"{token_selection}\n{source}")
                    rates.append(row["positive_score_rate"])
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.8), 4))
    ax.bar(np.arange(len(labels)), rates, color="#4c78a8")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("val pair positive-score rate")
    ax.set_title("Train-discovered top-20 confounder features transfer to validation pairs")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "validation_transfer_top20.png", dpi=150)
    plt.close(fig)


def _plot_overlap(summary: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    keys = []
    feature_sets = []
    for token_selection, by_source in summary["audits"].items():
        for source, audit in by_source.items():
            if not audit["top_features"]:
                continue
            keys.append(f"{token_selection}:{source}")
            feature_sets.append(set(row["feature_idx"] for row in audit["top_features"][:20]))

    if not keys:
        return

    mat = np.zeros((len(keys), len(keys)), dtype=np.float32)
    for row_i, row_set in enumerate(feature_sets):
        for col_i, col_set in enumerate(feature_sets):
            union = row_set | col_set
            mat[row_i, col_i] = len(row_set & col_set) / len(union) if union else 0.0

    fig, ax = plt.subplots(figsize=(max(7, len(keys) * 0.7), max(6, len(keys) * 0.6)))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(np.arange(len(keys)))
    ax.set_yticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(keys, fontsize=7)
    ax.set_title("Top-20 confounder feature Jaccard overlap")
    fig.colorbar(im, ax=ax, label="Jaccard", shrink=0.75)
    fig.tight_layout()
    fig.savefig(output_dir / "top_feature_overlap_jaccard.png", dpi=150)
    plt.close(fig)


def _plot_pair_score_hist(summary: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    for token_selection, by_source in summary["audits"].items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, source in zip(axes, ["pseudo_img", "pseudo_text"]):
            scores = [row["score"] for row in by_source[source].get("validation_pair_scores", [])]
            if not scores:
                ax.set_title(f"{token_selection} / {source}: no val scores")
                ax.axis("off")
                continue
            ax.hist(scores, bins=30, color="#72b7b2", edgecolor="white")
            ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
            ax.set_title(f"{token_selection} / {source}")
            ax.set_xlabel("top-20 signed feature score")
            ax.set_ylabel("validation pairs")
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"validation_pair_score_hist_{token_selection}.png", dpi=150)
        plt.close(fig)


def _strip_arrays(summary: dict[str, Any]) -> dict[str, Any]:
    clean = dict(summary)
    clean["audits"] = {}
    for token_selection, by_source in summary["audits"].items():
        clean["audits"][token_selection] = {}
        for source, audit in by_source.items():
            audit_clean = {key: value for key, value in audit.items() if key != "arrays"}
            clean["audits"][token_selection][source] = audit_clean
    return clean


def run_audit(cfg: FHMConfounderFeatureAuditConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_cache_dir = Path(cfg.train_cache_dir)
    val_cache_dir = Path(cfg.val_cache_dir)
    metadata_path = Path(cfg.fhm_confounder_metadata_path)
    assert train_cache_dir.exists(), f"Train cache missing: {train_cache_dir}"
    assert val_cache_dir.exists(), f"Validation cache missing: {val_cache_dir}"
    assert metadata_path.exists(), f"Metadata missing: {metadata_path}"

    device = resolve_visible_device(cfg.device)
    logger.info("Loading SAE from %s", cfg.residual_sae_path)
    sae = load_residual_sae(cfg.residual_sae_path, device=device)
    stats = load_residual_stats(cfg.residual_sae_path)
    metadata = _load_fhm_confounder_metadata(metadata_path)

    train_rows = _load_fhm_rows(train_cache_dir, {"train"})
    val_rows = _load_fhm_rows(val_cache_dir, {"validation", "val", "dev"}, cache_tag=True)
    logger.info("FHM rows: train=%d val=%d metadata=%d", len(train_rows), len(val_rows), len(metadata))

    summary: dict[str, Any] = {
        "config": vars(cfg),
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "audits": {},
    }
    lr_top_features = _load_lr_top_features(Path(cfg.compare_top_features_json))

    for token_selection in cfg.token_selections:
        logger.info("Extracting %s features", token_selection)
        train_keys, train_csr = _extract_features(
            train_rows,
            train_cache_dir,
            sae,
            stats,
            token_selection=token_selection,
            aggregation=cfg.aggregation,
            batch_size=cfg.batch_size,
            device=device,
            n_features=sae.num_latents,
        )
        val_keys, val_csr = _extract_features_multi(
            val_rows,
            sae,
            stats,
            token_selection=token_selection,
            aggregation=cfg.aggregation,
            batch_size=cfg.batch_size,
            device=device,
            n_features=sae.num_latents,
        )
        train_key_to_idx = {key: idx for idx, key in enumerate(train_keys)}
        val_key_to_idx = {key: idx for idx, key in enumerate(val_keys)}

        summary["audits"][token_selection] = {}
        for source in ["pseudo_img", "pseudo_text"]:
            train_pairs = _build_pairs(train_rows, metadata, set(train_keys), source)
            val_pairs = _build_pairs(val_rows, metadata, set(val_keys), source)
            audit = _feature_stats(train_csr, train_key_to_idx, train_pairs, cfg.top_k_features)
            audit["n_train_pairs"] = len(train_pairs)
            audit["n_val_pairs"] = len(val_pairs)
            audit["transfer"] = _transfer_stats(audit, val_csr, val_key_to_idx, val_pairs, cfg.transfer_top_k)
            audit["validation_pair_scores"] = _score_pairs(audit, val_csr, val_key_to_idx, val_pairs, 20)
            audit_top = set(row["feature_idx"] for row in audit["top_features"][:20])
            audit["overlap_with_lr_fhm_top20"] = {
                "n_overlap": len(audit_top & lr_top_features),
                "features": sorted(audit_top & lr_top_features),
            }
            summary["audits"][token_selection][source] = audit
            logger.info(
                "%s/%s: train_pairs=%d val_pairs=%d top_lr_overlap=%s",
                token_selection,
                source,
                len(train_pairs),
                len(val_pairs),
                audit["overlap_with_lr_fhm_top20"],
            )

    _plot_scatter(summary, output_dir)
    _plot_transfer(summary, output_dir)
    _plot_overlap(summary, output_dir)
    _plot_pair_score_hist(summary, output_dir)

    clean_summary = _strip_arrays(summary)
    (output_dir / "confounder_feature_audit_summary.json").write_text(json.dumps(clean_summary, indent=2))
    with (output_dir / "confounder_feature_top_features.jsonl").open("w") as fd:
        for token_selection, by_source in clean_summary["audits"].items():
            for source, audit in by_source.items():
                for rank, row in enumerate(audit["top_features"], start=1):
                    fd.write(json.dumps({
                        "token_selection": token_selection,
                        "source": source,
                        "rank": rank,
                        **row,
                    }) + "\n")
    with (output_dir / "validation_pair_feature_scores.jsonl").open("w") as fd:
        for token_selection, by_source in clean_summary["audits"].items():
            for source, audit in by_source.items():
                for row in audit["validation_pair_scores"]:
                    fd.write(json.dumps({
                        "token_selection": token_selection,
                        "source": source,
                        **row,
                    }) + "\n")

    logger.info("Wrote FHM confounder feature audit to %s", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(FHMConfounderFeatureAuditConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run_audit(cfg)


if __name__ == "__main__":
    main()