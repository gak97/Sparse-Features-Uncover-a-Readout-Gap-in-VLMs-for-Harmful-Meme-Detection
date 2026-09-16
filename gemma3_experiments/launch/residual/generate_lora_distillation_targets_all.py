

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
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


def _task_slug(task_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", task_key).strip("_")


# One entry per task. `collapse_negative_labels` is only used to build pairs
# and the calibration prior for multi-class tasks (`None` = skip pairs
# entirely for that task).
TASK_SPECS: list[dict[str, Any]] = [
    dict(
        dataset_name="Hateful_Memes_Dataset", task_name="hateful_memes_binary", field_name="label",
        labels=("not hateful", "hateful"), decode_strategy="yes_no",
        yes_no_positive_label="hateful", semantic_positive_label="hateful",
        collapse_negative_labels=("not hateful",), probe_kind="fhm",
        token_selection="all", aggregation="max", calibration=False,
        pair_source="fhm_confounder",
    ),
    dict(
        dataset_name="MAMI", task_name="misogyny_and_subtypes", field_name="label",
        labels=("non-misogynous", "misogynous"), decode_strategy="yes_no",
        yes_no_positive_label="misogynous", semantic_positive_label="misogynous",
        collapse_negative_labels=("non-misogynous",), probe_kind="other",
        token_selection="image", aggregation="max", calibration=True,
        pair_source="text_match",
    ),
    dict(
        dataset_name="MultiOFF", task_name="offensive_detection", field_name="label",
        labels=("Non-offensiv", "offensive"), decode_strategy="yes_no",
        yes_no_positive_label="offensive", semantic_positive_label="offensive",
        collapse_negative_labels=("Non-offensiv",), probe_kind="other",
        token_selection="image", aggregation="max", calibration=False,
        pair_source="text_match",
    ),
    dict(
        dataset_name="Crisis_HateMM", task_name="subtask_a_hate_speech", field_name="label",
        labels=("Hate Speech", "No Hate Speech"), decode_strategy="yes_no",
        # NOTE polarity: schema's LAST label is "No Hate Speech", so
        # build_simple_classify_prompt's implicit yes/no positive label is
        # "No Hate Speech" here, the OPPOSITE of the semantic hateful class.
        yes_no_positive_label="No Hate Speech", semantic_positive_label="Hate Speech",
        collapse_negative_labels=("No Hate Speech",), probe_kind="other",
        token_selection="image", aggregation="max", calibration=False,
        pair_source="text_match",
    ),
    dict(
        dataset_name="HarMeme", task_name="harmfulness_and_target", field_name="harmfulness",
        labels=("not harmful", "somewhat harmful", "very harmful"), decode_strategy="label_score",
        yes_no_positive_label=None, semantic_positive_label="harmful",
        collapse_negative_labels=("not harmful",), probe_kind="other",
        token_selection="image", aggregation="max", calibration=True,
        pair_source="text_match",
    ),
    dict(
        dataset_name="MMHS150K", task_name="hate_category", field_name="label",
        labels=("NotHate", "Racist", "Sexist", "Homophobe", "Religion", "OtherHate"), decode_strategy="label_score",
        yes_no_positive_label=None, semantic_positive_label="Hate",
        collapse_negative_labels=("NotHate",), probe_kind="other",
        token_selection="image", aggregation="max", calibration=True,
        pair_source="text_match",
    ),
    dict(
        dataset_name="Crisis_HateMM", task_name="subtask_b_hate_target", field_name="target",
        labels=("Community", "Individual", "Organization"), decode_strategy="label_score",
        yes_no_positive_label=None, semantic_positive_label=None,
        collapse_negative_labels=None, probe_kind="other",  # no natural binary collapse -- no pairs
        token_selection="image", aggregation="max", calibration=False,
        pair_source="none",
    ),
]


@dataclass
class GenerateAllDistillationTargetsConfig:
    residual_sae_path: str = ""
    train_cache_dir: str = ""
    fhm_probe_dir: str = ""       # residual_sae_probe_validation_fhm_mlp_pseudo_hn
    other_probe_dir: str = ""     # residual_sae_probe_validation_all_tasks_saved
    fhm_confounder_metadata_path: str = ""
    task_keys: list[str] = field(default_factory=list)  # empty = all TASK_SPECS
    batch_size: int = 256
    device: str = "cuda:0"
    output_dir: str = "./lora_distillation_targets_all"


def _load_probe(probe_dir: Path, prefix: str) -> tuple[Any, Any, Any] | None:
    import joblib

    probe_path = probe_dir / "probe.joblib" if prefix == "probe" else probe_dir / f"{prefix}.joblib"
    scaler_path = probe_dir / ("scaler.joblib" if prefix == "probe" else f"{prefix.replace('probe', 'scaler')}.joblib")
    le_path = probe_dir / (
        "label_encoder.joblib" if prefix == "probe" else f"{prefix.replace('probe', 'label_encoder')}.joblib"
    )
    if not probe_path.exists():
        return None
    clf = joblib.load(probe_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    le = joblib.load(le_path) if le_path.exists() else None
    return clf, scaler, le


def _collapse(label: str, negative_labels: tuple[str, ...]) -> str:
    return "negative" if label in negative_labels else "positive"


def _process_task(
    spec: dict[str, Any], cfg: GenerateAllDistillationTargetsConfig, sae: Any, stats: Any, device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dataset_name = spec["dataset_name"]
    task_name = spec["task_name"]
    field_name = spec["field_name"]
    task_key = f"{dataset_name}::{task_name}::{field_name}"
    slug = _task_slug(task_key)
    logger.info("=== Task %s (slug=%s) ===", task_key, slug)

    if spec["probe_kind"] == "fhm":
        probe = _load_probe(Path(cfg.fhm_probe_dir), "probe")
    else:
        probe = _load_probe(Path(cfg.other_probe_dir), f"probe__{slug}")
    assert probe is not None, f"No probe found for {task_key} (probe_kind={spec['probe_kind']})"
    clf, scaler, le = probe
    classes = le.classes_.tolist() if le is not None else list(spec["labels"])
    assert set(classes) == set(spec["labels"]), (
        f"Probe classes {classes} do not match expected schema labels {spec['labels']} for {task_key}"
    )

    train_rows = [
        row for row in _read_jsonl(Path(cfg.train_cache_dir) / "manifest.jsonl")
        if row.get("dataset_name") == dataset_name
        and row.get("task_name") == task_name
        and row.get("split") == "train"
    ]
    assert train_rows, f"No train rows found for {task_key} in {cfg.train_cache_dir}"
    logger.info("Loaded %d train rows for %s", len(train_rows), task_key)

    keys, csr = _extract_features(
        train_rows, Path(cfg.train_cache_dir), sae, stats,
        token_selection=spec["token_selection"], aggregation=spec["aggregation"],
        batch_size=cfg.batch_size, device=device, n_features=sae.num_latents,
    )
    assert csr.shape[0] > 0, f"No features extracted for {task_key}"

    X = scaler.transform(csr) if scaler is not None else csr
    X_dense = X.toarray() if hasattr(X, "toarray") else X
    proba = clf.predict_proba(X_dense)

    scores_by_sid: dict[str, dict[str, float]] = {}
    for i, key in enumerate(keys):
        sid = key.split("::")[-1]
        scores_by_sid[sid] = {cls: float(proba[i, j]) for j, cls in enumerate(classes)}
    logger.info("Computed probe scores for %d samples (%s)", len(scores_by_sid), task_key)

    samples = load_meme_samples(MemeDatasetConfig(datasets=[dataset_name], splits=["train"]))
    sample_by_sid = {s.sample_id: s for s in samples if s.dataset_name == dataset_name and s.task_name == task_name}

    gold_by_sid: dict[str, str] = {}
    targets: list[dict[str, Any]] = []
    n_missing_sample = 0
    for row in train_rows:
        sid = row["sample_id"]
        if sid not in scores_by_sid:
            continue
        sample = sample_by_sid.get(sid)
        if sample is None:
            n_missing_sample += 1
            continue
        gold = _row_gold_label(row)
        if gold is None:
            continue
        gold_by_sid[sid] = gold
        targets.append({
            "dataset_name": dataset_name,
            "task_name": task_name,
            "field_name": field_name,
            "sample_id": sid,
            "image_path": sample.image_path,
            "meme_text": sample.meme_text,
            "labels": list(spec["labels"]),
            "decode_strategy": spec["decode_strategy"],
            "yes_no_positive_label": spec["yes_no_positive_label"],
            "gold_label": gold,
            "probe_label_scores": {k: round(v, 6) for k, v in scores_by_sid[sid].items()},
        })
    logger.info(
        "Task %s: wrote %d targets (%d missing sample lookups)", task_key, len(targets), n_missing_sample,
    )

    pairs: list[dict[str, Any]] = []
    negative_labels = spec["collapse_negative_labels"]
    if spec["pair_source"] == "none" or negative_labels is None:
        logger.info("Task %s: pair_source=none, skipping pairs (no natural binary collapse)", task_key)
    elif spec["pair_source"] == "fhm_confounder":
        assert cfg.fhm_confounder_metadata_path, "fhm_confounder_metadata_path required for pair_source=fhm_confounder"
        confounder_meta = _load_fhm_confounder_metadata(Path(cfg.fhm_confounder_metadata_path))
        for group_key_name in ("pseudo_img_idx", "pseudo_text_idx"):
            groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for sid, gold in gold_by_sid.items():
                meta = confounder_meta.get(_normalise_fhm_id(sid))
                if meta is None:
                    continue
                group_val = _normalise_group_value(meta.get(group_key_name))
                if group_val is None:
                    continue
                groups[group_val].append((sid, gold))
            for members in groups.values():
                pos_ids = [sid for sid, gold in members if _collapse(gold, negative_labels) == "positive"]
                neg_ids = [sid for sid, gold in members if _collapse(gold, negative_labels) == "negative"]
                for p_sid in pos_ids:
                    for n_sid in neg_ids:
                        p_prob = 1.0 - sum(scores_by_sid[p_sid][lab] for lab in negative_labels)
                        n_prob = 1.0 - sum(scores_by_sid[n_sid][lab] for lab in negative_labels)
                        pairs.append({
                            "dataset_name": dataset_name, "task_name": task_name, "field_name": field_name,
                            "source": group_key_name,
                            "positive_sample_id": p_sid, "negative_sample_id": n_sid,
                            "probe_margin": round(p_prob - n_prob, 6),
                        })
    elif spec["pair_source"] == "text_match":
        by_text: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for row in train_rows:
            sid = row["sample_id"]
            if sid not in gold_by_sid:
                continue
            text = (row.get("meme_text") or "").lower().strip()
            if text:
                by_text[text].append((sid, gold_by_sid[sid]))
        for members in by_text.values():
            pos_ids = [sid for sid, gold in members if _collapse(gold, negative_labels) == "positive"]
            neg_ids = [sid for sid, gold in members if _collapse(gold, negative_labels) == "negative"]
            for p_sid in pos_ids:
                for n_sid in neg_ids:
                    p_prob = 1.0 - sum(scores_by_sid[p_sid][lab] for lab in negative_labels)
                    n_prob = 1.0 - sum(scores_by_sid[n_sid][lab] for lab in negative_labels)
                    pairs.append({
                        "dataset_name": dataset_name, "task_name": task_name, "field_name": field_name,
                        "source": "text_match",
                        "positive_sample_id": p_sid, "negative_sample_id": n_sid,
                        "probe_margin": round(p_prob - n_prob, 6),
                    })
    logger.info("Task %s: built %d pairs (pair_source=%s)", task_key, len(pairs), spec["pair_source"])

    empirical_prior = None
    if negative_labels is not None:
        n_pos = sum(1 for gold in gold_by_sid.values() if _collapse(gold, negative_labels) == "positive")
        empirical_prior = n_pos / len(gold_by_sid) if gold_by_sid else None

    manifest_entry = {
        "task_key": task_key,
        "slug": slug,
        "dataset_name": dataset_name,
        "task_name": task_name,
        "field_name": field_name,
        "labels": list(spec["labels"]),
        "decode_strategy": spec["decode_strategy"],
        "yes_no_positive_label": spec["yes_no_positive_label"],
        "semantic_positive_label": spec["semantic_positive_label"],
        "collapse_negative_labels": list(negative_labels) if negative_labels is not None else None,
        "calibration": spec["calibration"],
        "empirical_positive_prior": empirical_prior,
        "n_targets": len(targets),
        "n_pairs": len(pairs),
    }
    return targets, pairs, manifest_entry


def run(cfg: GenerateAllDistillationTargetsConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_visible_device(cfg.device)

    sae = load_residual_sae(cfg.residual_sae_path, device=device)
    stats = load_residual_stats(cfg.residual_sae_path)

    specs = TASK_SPECS
    if cfg.task_keys:
        wanted = set(cfg.task_keys)
        specs = [
            s for s in TASK_SPECS
            if f"{s['dataset_name']}::{s['task_name']}::{s['field_name']}" in wanted
        ]
        assert specs, f"No TASK_SPECS matched task_keys={cfg.task_keys}"

    manifest: list[dict[str, Any]] = []
    for spec in specs:
        targets, pairs, manifest_entry = _process_task(spec, cfg, sae, stats, device)
        slug = manifest_entry["slug"]
        with (out_dir / f"distillation_targets_{slug}.jsonl").open("w") as fd:
            for target in targets:
                fd.write(json.dumps(target) + "\n")
        with (out_dir / f"distillation_pairs_{slug}.jsonl").open("w") as fd:
            for pair in pairs:
                fd.write(json.dumps(pair) + "\n")
        manifest.append(manifest_entry)

    (out_dir / "distillation_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote distillation_manifest.json + per-task jsonl files to %s", out_dir)
    print(json.dumps(manifest, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(GenerateAllDistillationTargetsConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
