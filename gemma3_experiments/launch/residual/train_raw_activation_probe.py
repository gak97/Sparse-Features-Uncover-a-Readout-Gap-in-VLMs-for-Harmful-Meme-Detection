

import json
import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simple_parsing import ArgumentGenerationMode, parse


from gemma3_experiments.launch.residual.probe_validate import (
    _extract_raw_features,
    _per_class_metrics,
    _read_jsonl,
    _row_gold_label,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainRawActivationProbeConfig:
    dense_cache_dir: str = ""          # self-contained dense cache (e.g. dense_fhm_layer22)
    train_splits: tuple[str, ...] = ("train",)
    val_splits: tuple[str, ...] = ("validation", "val", "dev")
    dataset_name: str = "Hateful_Memes_Dataset"
    task_name: str = "hateful_memes_binary"
    field_name: str = "label"

    token_selection: str = "all"
    aggregation: str = "max"

    # Match the production Stage 1 MLP hyperparameters for a controlled comparison.
    mlp_hidden_size: int = 512
    mlp_dropout: float = 0.3
    mlp_alpha: float = 1e-4
    mlp_max_iter: int = 200

    output_dir: str = "./raw_activation_probe"


def _load_rows(cache_dir: Path, splits: tuple[str, ...], dataset_name: str, task_name: str) -> list[dict[str, Any]]:
    rows = []
    for row in _read_jsonl(cache_dir / "manifest.jsonl"):
        if row.get("dataset_name") != dataset_name:
            continue
        if row.get("task_name") != task_name:
            continue
        if row.get("split") not in splits:
            continue
        rows.append(row)
    return rows


def run(cfg: TrainRawActivationProbeConfig) -> None:
    import numpy as np
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder, MaxAbsScaler

    cache_dir = Path(cfg.dense_cache_dir)
    assert cache_dir.exists(), f"Dense cache not found: {cache_dir}"
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _load_rows(cache_dir, cfg.train_splits, cfg.dataset_name, cfg.task_name)
    val_rows = _load_rows(cache_dir, cfg.val_splits, cfg.dataset_name, cfg.task_name)
    assert train_rows, f"No train rows found for {cfg.dataset_name}/{cfg.task_name} in {cache_dir}"
    assert val_rows, f"No val rows found for {cfg.dataset_name}/{cfg.task_name} in {cache_dir}"
    logger.info("Loaded %d train rows, %d val rows from %s", len(train_rows), len(val_rows), cache_dir)

    tr_keys, X_train = _extract_raw_features(
        train_rows, cache_dir, token_selection=cfg.token_selection, aggregation=cfg.aggregation,
    )
    va_keys, X_val = _extract_raw_features(
        val_rows, cache_dir, token_selection=cfg.token_selection, aggregation=cfg.aggregation,
    )
    assert X_train.shape[0] > 0, "No train features extracted -- check token_selection/shard alignment"
    assert X_val.shape[0] > 0, "No val features extracted -- check token_selection/shard alignment"

    tr_by_key = {row["sample_id"]: row for row in train_rows}
    va_by_key = {row["sample_id"]: row for row in val_rows}
    # tr_keys/va_keys come back as _row_sample_key(row) = "dataset::task::sample_id"
    tr_gold = [_row_gold_label(tr_by_key[k.split("::")[-1]]) for k in tr_keys]
    va_gold = [_row_gold_label(va_by_key[k.split("::")[-1]]) for k in va_keys]
    assert all(g is not None for g in tr_gold), "Missing gold label for some train rows"
    assert all(g is not None for g in va_gold), "Missing gold label for some val rows"

    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    le = LabelEncoder()
    y_train = le.fit_transform(tr_gold)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = MLPClassifier(
            hidden_layer_sizes=(cfg.mlp_hidden_size,),
            activation="relu",
            alpha=cfg.mlp_alpha,
            max_iter=cfg.mlp_max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42,
            verbose=False,
        )
        clf.fit(X_train, y_train)
        y_pred_enc = clf.predict(X_val)
    preds = le.inverse_transform(y_pred_enc).tolist()

    metrics = _per_class_metrics(va_gold, preds)
    logger.info("Raw-activation probe: accuracy=%.4f macro_f1=%.4f", metrics["accuracy"], metrics["macro_f1"])

    import joblib
    joblib.dump(clf, out_dir / "probe_raw.joblib")
    joblib.dump(scaler, out_dir / "scaler_raw.joblib")
    joblib.dump(le, out_dir / "label_encoder.joblib")
    logger.info("Saved probe_raw.joblib, scaler_raw.joblib, label_encoder.joblib to %s", out_dir)

    task_key = f"{cfg.dataset_name}::{cfg.task_name}::{cfg.field_name}"
    summary = {
        task_key: {
            "task_key": task_key,
            "dataset_name": cfg.dataset_name,
            "task_name": cfg.task_name,
            "field_name": cfg.field_name,
            "n_train": len(tr_keys),
            "n_val": len(va_keys),
            "classes": le.classes_.tolist(),
            "raw_residual_control": {
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "hidden_dim": int(X_train.shape[1]),
                "n_tr_extracted": len(tr_keys),
                "per_class": metrics["per_class"],
                "gold_counts": metrics["gold_counts"],
                "pred_counts": metrics["pred_counts"],
            },
        }
    }
    (out_dir / "probe_val_metrics.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote probe_val_metrics.json to %s", out_dir)

    print("\n" + "=" * 72)
    print(f"Raw-activation Stage 2 probe -- {task_key}")
    print(f"n_train={len(tr_keys)}  n_val={len(va_keys)}  hidden_dim={X_train.shape[1]}")
    print(f"macro_f1={metrics['macro_f1']}  accuracy={metrics['accuracy']}")
    print("=" * 72)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(TrainRawActivationProbeConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
