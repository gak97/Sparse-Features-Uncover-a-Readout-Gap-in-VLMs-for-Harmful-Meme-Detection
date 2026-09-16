

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simple_parsing import ArgumentGenerationMode, parse


from gemma3_experiments.launch.residual.probe_validate import (
    _extract_features,
    _load_fhm_confounder_metadata,
    _normalise_fhm_id,
    _normalise_group_value,
    _read_jsonl,
    _row_gold_label,
)
from gemma3_experiments.residual_sae import (
    MemeDatasetConfig,
    load_meme_samples,
    load_residual_sae,
    load_residual_stats,
    resolve_visible_device,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerateDistillationTargetsConfig:
    residual_sae_path: str = ""
    train_cache_dir: str = ""
    probe_dir: str = ""                        # residual_sae_probe_validation_fhm_mlp_pseudo_hn
    fhm_confounder_metadata_path: str = ""
    dataset_name: str = "Hateful_Memes_Dataset"
    task_name: str = "hateful_memes_binary"
    positive_label: str = "hateful"
    token_selection: str = "all"
    aggregation: str = "max"
    batch_size: int = 256
    device: str = "cuda:0"
    output_dir: str = "./lora_distillation_targets"


def run(cfg: GenerateDistillationTargetsConfig) -> None:
    import joblib
    import numpy as np

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_visible_device(cfg.device)

    sae = load_residual_sae(cfg.residual_sae_path, device=device)
    stats = load_residual_stats(cfg.residual_sae_path)

    probe_dir = Path(cfg.probe_dir)
    clf = joblib.load(probe_dir / "probe.joblib")
    scaler = joblib.load(probe_dir / "scaler.joblib")
    le = joblib.load(probe_dir / "label_encoder.joblib")
    assert cfg.positive_label in le.classes_.tolist(), (
        f"positive_label={cfg.positive_label!r} not in probe classes {le.classes_.tolist()}"
    )

    train_rows = [
        row for row in _read_jsonl(Path(cfg.train_cache_dir) / "manifest.jsonl")
        if row.get("dataset_name") == cfg.dataset_name
        and row.get("task_name") == cfg.task_name
        and row.get("split") == "train"
    ]
    assert train_rows, f"No train rows found for {cfg.dataset_name}/{cfg.task_name} in {cfg.train_cache_dir}"
    logger.info("Loaded %d train rows", len(train_rows))

    keys, csr = _extract_features(
        train_rows, Path(cfg.train_cache_dir), sae, stats,
        token_selection=cfg.token_selection, aggregation=cfg.aggregation,
        batch_size=cfg.batch_size, device=device, n_features=sae.num_latents,
    )
    assert csr.shape[0] > 0, "No features extracted -- check token_selection/shard alignment"

    X = scaler.transform(csr)
    X_dense = X.toarray() if hasattr(X, "toarray") else X
    proba = clf.predict_proba(X_dense)
    classes = le.classes_.tolist()
    pos_idx = classes.index(cfg.positive_label)

    proba_by_sid: dict[str, float] = {}
    for i, key in enumerate(keys):
        sid = key.split("::")[-1]
        proba_by_sid[sid] = float(proba[i, pos_idx])
    logger.info("Computed probe P(%s) for %d samples", cfg.positive_label, len(proba_by_sid))

    samples = load_meme_samples(MemeDatasetConfig(datasets=[cfg.dataset_name], splits=["train"]))
    sample_by_sid = {
        s.sample_id: s for s in samples
        if s.dataset_name == cfg.dataset_name and s.task_name == cfg.task_name
    }

    targets: list[dict[str, Any]] = []
    n_missing_sample = 0
    for row in train_rows:
        sid = row["sample_id"]
        if sid not in proba_by_sid:
            continue
        sample = sample_by_sid.get(sid)
        if sample is None:
            n_missing_sample += 1
            continue
        gold = _row_gold_label(row)
        if gold is None:
            continue
        targets.append({
            "sample_id": sid,
            "image_path": sample.image_path,
            "meme_text": sample.meme_text,
            "gold_label": gold,
            "probe_positive_proba": round(proba_by_sid[sid], 6),
            "positive_label": cfg.positive_label,
        })
    logger.info("Wrote %d distillation targets (%d missing sample lookups)", len(targets), n_missing_sample)

    with (out_dir / "distillation_targets.jsonl").open("w") as fd:
        for target in targets:
            fd.write(json.dumps(target) + "\n")

    pairs: list[dict[str, Any]] = []
    if cfg.fhm_confounder_metadata_path:
        confounder_meta = _load_fhm_confounder_metadata(Path(cfg.fhm_confounder_metadata_path))
        gold_by_sid = {row["sample_id"]: _row_gold_label(row) for row in train_rows}
        for group_key_name in ("pseudo_img_idx", "pseudo_text_idx"):
            groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for sid, proba_pos in proba_by_sid.items():
                gold = gold_by_sid.get(sid)
                if gold is None:
                    continue
                meta = confounder_meta.get(_normalise_fhm_id(sid))
                if meta is None:
                    continue
                group_val = _normalise_group_value(meta.get(group_key_name))
                if group_val is None:
                    continue
                groups[group_val].append((sid, gold))

            for group_val, members in groups.items():
                hateful_ids = [sid for sid, gold in members if gold == cfg.positive_label]
                benign_ids = [sid for sid, gold in members if gold != cfg.positive_label]
                for h_sid in hateful_ids:
                    for b_sid in benign_ids:
                        pairs.append({
                            "source": group_key_name,
                            "group": group_val,
                            "hateful_sample_id": h_sid,
                            "benign_sample_id": b_sid,
                            "probe_margin": round(proba_by_sid[h_sid] - proba_by_sid[b_sid], 6),
                        })
        logger.info("Built %d pseudo-confounder pairs", len(pairs))
    else:
        logger.warning("fhm_confounder_metadata_path not set -- skipping pair generation")

    with (out_dir / "distillation_pairs.jsonl").open("w") as fd:
        for pair in pairs:
            fd.write(json.dumps(pair) + "\n")

    summary = {
        "n_targets": len(targets),
        "n_pairs": len(pairs),
        "mean_probe_margin": float(np.mean([p["probe_margin"] for p in pairs])) if pairs else None,
        "frac_positive_margin": float(np.mean([p["probe_margin"] > 0 for p in pairs])) if pairs else None,
    }
    (out_dir / "distillation_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote distillation_targets.jsonl, distillation_pairs.jsonl, distillation_summary.json to %s", out_dir)
    print(json.dumps(summary, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(GenerateDistillationTargetsConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
