"""Evaluate public Gemma Scope SAE features on held-out fine-grained tasks."""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from simple_parsing import parse, list_field

from gemma3_experiments.residual_sae.gemma import GemmaSae

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskSpec:
    dataset_name: str
    task_name: str
    field_name: str
    labels: tuple[str, ...]


TASK_SPECS = (
    TaskSpec(
        "HarMeme",
        "harmfulness_and_target",
        "harmfulness",
        ("not harmful", "somewhat harmful", "very harmful"),
    ),
    TaskSpec(
        "MMHS150K",
        "hate_category",
        "label",
        ("NotHate", "Racist", "Sexist", "Homophobe", "Religion", "OtherHate"),
    ),
)


@dataclass
class BaseSaeProbeConfig:
    train_cache_dir: str
    test_cache_dir: str
    output_dir: str
    datasets: list[str] = list_field("HarMeme", "MMHS150K")
    sae_repo_id: str = "google/gemma-scope-2-4b-it"
    sae_hookpoint: str = "resid_post/layer_22_width_65k_l0_medium"
    device: str = "cuda:0"
    encode_chunk_size: int = 128
    C: float = 1.0
    max_iter: int = 2000


def _read_manifest(cache_dir: Path, split: str, spec: TaskSpec) -> list[dict]:
    manifest_fpath = cache_dir / "manifest.jsonl"
    assert manifest_fpath.exists(), f"Missing dense-cache manifest: {manifest_fpath}"
    rows = []
    with manifest_fpath.open() as fd:
        for line in fd:
            row = json.loads(line)
            if (
                row.get("split") == split
                and row.get("dataset_name") == spec.dataset_name
                and row.get("task_name") == spec.task_name
                and (row.get("gold_fields") or {}).get(spec.field_name) in spec.labels
            ):
                rows.append(row)
    assert rows, f"No {split} rows found for {spec.dataset_name}/{spec.task_name} in {manifest_fpath}"
    return rows


def _extract_features(
    sae: GemmaSae,
    cache_dir: Path,
    rows: list[dict],
    spec: TaskSpec,
    chunk_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    by_shard: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for row_i, row in enumerate(rows):
        by_shard[row["shard_file"]].append((row_i, row))

    features = np.empty((len(rows), sae.cfg.num_latents), dtype=np.float32)
    labels = np.empty(len(rows), dtype=object)
    for shard_name, shard_rows in sorted(by_shard.items()):
        shard_fpath = cache_dir / shard_name
        assert shard_fpath.exists(), f"Missing dense shard: {shard_fpath}"
        hidden_all = load_file(str(shard_fpath))["hidden"].to(device=device, dtype=torch.float32)
        for row_i, row in shard_rows:
            offset = int(row["offset"])
            length = int(row["length"])
            hidden = hidden_all[offset : offset + length]
            assert hidden.shape[0] == length and length > 0, f"Invalid hidden span for {row['sample_id']}"
            pooled = torch.zeros(sae.cfg.num_latents, device=device, dtype=torch.float32)
            with torch.no_grad():
                for start in range(0, length, chunk_size):
                    acts = sae.encode(hidden[start : start + chunk_size])
                    pooled = torch.maximum(pooled, acts.float().max(dim=0).values)
            features[row_i] = pooled.cpu().numpy()
            labels[row_i] = row["gold_fields"][spec.field_name]
    return features, labels


def _fit_predict(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_test: np.ndarray,
    C: float,
    max_iter: int,
) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MaxAbsScaler

    scaler = MaxAbsScaler()
    train_scaled = scaler.fit_transform(features_train)
    test_scaled = scaler.transform(features_test)
    clf = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=max_iter,
        solver="lbfgs",
    )
    clf.fit(train_scaled, labels_train)
    return clf.predict(test_scaled)


def _metrics(labels: np.ndarray, predictions: np.ndarray, classes: tuple[str, ...]) -> dict[str, float | int]:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "macro_f1": round(float(f1_score(labels, predictions, labels=list(classes), average="macro", zero_division=0)), 4),
        "n_eval": int(len(labels)),
    }


def _run_hierarchical_mmhs(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_test: np.ndarray,
    labels_test: np.ndarray,
    cfg: BaseSaeProbeConfig,
) -> dict[str, float | int]:
    stage1_train = np.where(labels_train == "NotHate", "NotHate", "Hate")
    stage1_predictions = _fit_predict(features_train, stage1_train, features_test, cfg.C, cfg.max_iter)
    hate_mask = labels_train != "NotHate"
    stage2_predictions = _fit_predict(
        features_train[hate_mask], labels_train[hate_mask], features_test, cfg.C, cfg.max_iter
    )
    predictions = np.where(stage1_predictions == "NotHate", "NotHate", stage2_predictions)
    return _metrics(labels_test, predictions, TASK_SPECS[1].labels)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = parse(BaseSaeProbeConfig)
    train_cache_dir = Path(cfg.train_cache_dir)
    test_cache_dir = Path(cfg.test_cache_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sae = GemmaSae.load_from_hub(cfg.sae_repo_id, cfg.sae_hookpoint, device=cfg.device)
    logger.info("Loaded base SAE: d_in=%d n_latents=%d", sae.cfg.d_in, sae.cfg.num_latents)
    results: dict[str, dict] = {}
    for spec in TASK_SPECS:
        if spec.dataset_name not in cfg.datasets:
            continue
        train_rows = _read_manifest(train_cache_dir, "train", spec)
        test_rows = _read_manifest(test_cache_dir, "test", spec)
        logger.info("Extracting %s features: train=%d test=%d", spec.dataset_name, len(train_rows), len(test_rows))
        features_train, labels_train = _extract_features(
            sae, train_cache_dir, train_rows, spec, cfg.encode_chunk_size, cfg.device
        )
        features_test, labels_test = _extract_features(
            sae, test_cache_dir, test_rows, spec, cfg.encode_chunk_size, cfg.device
        )
        predictions = _fit_predict(features_train, labels_train, features_test, cfg.C, cfg.max_iter)
        key = f"{spec.dataset_name}::{spec.task_name}"
        results[key] = {
            "dataset_name": spec.dataset_name,
            "task_name": spec.task_name,
            "view": "flat",
            "split": "proper",
            "n_train": int(len(labels_train)),
            "classes": list(spec.labels),
            **_metrics(labels_test, predictions, spec.labels),
        }
        logger.info("%s flat: %s", key, results[key])
        if spec.dataset_name == "MMHS150K":
            results[f"{key}::hierarchical"] = {
                "dataset_name": spec.dataset_name,
                "task_name": spec.task_name,
                "view": "hierarchical",
                "split": "proper",
                "n_train": int(len(labels_train)),
                "classes": list(spec.labels),
                **_run_hierarchical_mmhs(features_train, labels_train, features_test, labels_test, cfg),
            }
            logger.info("%s hierarchical: %s", key, results[f"{key}::hierarchical"])
        del features_train, features_test

    output_fpath = output_dir / "probe_base_sae_fine_grained_results.json"
    output_fpath.write_text(json.dumps(results, indent=2) + "\n")
    logger.info("Wrote %s", output_fpath)


if __name__ == "__main__":
    main()