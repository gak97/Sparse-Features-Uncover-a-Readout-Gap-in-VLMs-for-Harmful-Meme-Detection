"""Probe public base-SAE representations on the locked MultiBully holdout."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from simple_parsing import Serializable, parse

from ..qwen import QwenSae
from ..task_labels import fit_scaled_logreg, predict_scaled
from .analyze_exist_multilingual import ROLE_NAMES, _extract_features, _read_jsonl
from .probe_fhm_bilinear import _train_bilinear


logger = logging.getLogger(__name__)


@dataclass
class MultiBullyAnalysisConfig(Serializable):
    cache_dir: str = "./outputs/multibully/dense_raw"
    output_dir: str = "./outputs/multibully/analysis_raw_ocr"
    device: str = "cuda:0"
    sae_repo_id: str = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
    sae_layer: int = 20
    C: float = 1.0
    max_iter: int = 3000
    top_k_features: int = 256
    bilinear_rank: int = 16
    bilinear_epochs: int = 200
    bilinear_lr: float = 1e-3
    bilinear_weight_decay: float = 1e-3
    bilinear_patience: int = 20
    feature_top_k: int = 50
    audit_limit: int = 100


def _labels(rows: list[dict]) -> np.ndarray:
    return np.asarray([int(row["gold_fields"]["label"] == "bully") for row in rows], dtype=np.int32)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_bully": round(float(f1_score(y_true, y_pred, average="binary", zero_division=0)), 4),
        "pred_bully_rate": round(float(y_pred.mean()), 4),
    }


def _fit_linear(
    train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, eval_y: np.ndarray, cfg: MultiBullyAnalysisConfig
) -> tuple[dict, object, object, np.ndarray]:
    classifier, scaler = fit_scaled_logreg(train_x, train_y, C=cfg.C, max_iter=cfg.max_iter)
    prediction = predict_scaled(classifier, scaler, eval_x)
    return _metrics(eval_y, prediction), classifier, scaler, prediction


def _top_features(classifier, top_k: int) -> dict:
    coefficient = classifier.coef_[0].copy()
    if list(classifier.classes_)[1] != 1:
        coefficient *= -1
    up = np.argsort(-coefficient)[:top_k]
    down = np.argsort(coefficient)[:top_k]
    return {
        "label_up": [{"feature_idx": int(index), "weight": round(float(coefficient[index]), 6)} for index in up],
        "label_down": [{"feature_idx": int(index), "weight": round(float(coefficient[index]), 6)} for index in down],
    }


def _fit_bilinear(
    train_image: np.ndarray,
    train_prompt: np.ndarray,
    train_y: np.ndarray,
    eval_image: np.ndarray,
    eval_prompt: np.ndarray,
    eval_y: np.ndarray,
    cfg: MultiBullyAnalysisConfig,
) -> dict:
    from sklearn.preprocessing import StandardScaler

    selected_image = np.argsort(-train_image.var(axis=0))[:cfg.top_k_features]
    selected_prompt = np.argsort(-train_prompt.var(axis=0))[:cfg.top_k_features]
    image_scaler, prompt_scaler = StandardScaler(), StandardScaler()
    train_image = image_scaler.fit_transform(train_image[:, selected_image])
    train_prompt = prompt_scaler.fit_transform(train_prompt[:, selected_prompt])
    eval_image = image_scaler.transform(eval_image[:, selected_image])
    eval_prompt = prompt_scaler.transform(eval_prompt[:, selected_prompt])
    model, internal_accuracy = _train_bilinear(
        train_image,
        train_prompt,
        train_y,
        rank=cfg.bilinear_rank,
        n_epochs=cfg.bilinear_epochs,
        lr=cfg.bilinear_lr,
        weight_decay=cfg.bilinear_weight_decay,
        patience=cfg.bilinear_patience,
        device=cfg.device,
    )
    with torch.no_grad():
        logits = model(
            torch.tensor(eval_image, dtype=torch.float32, device=cfg.device),
            torch.tensor(eval_prompt, dtype=torch.float32, device=cfg.device),
        )
    return {**_metrics(eval_y, (logits.cpu().numpy() >= 0).astype(np.int32)), "internal_val_accuracy": round(float(internal_accuracy), 4)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(MultiBullyAnalysisConfig)
    cache_dir, output_dir = Path(cfg.cache_dir), Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = [
        row for row in _read_jsonl(cache_dir / "manifest.jsonl")
        if row.get("task_name") == "cyberbullying_detection" and row.get("gold_fields")
    ]
    if {row.get("split") for row in manifest} != {"probe_train", "eval"}:
        raise ValueError("MultiBully analysis requires the fixed probe_train and eval partitions")

    sae = QwenSae.load_from_hub(cfg.sae_repo_id, cfg.sae_layer, device=cfg.device)
    metadata, features = _extract_features(sae, cache_dir, manifest, cfg.device)
    labels = _labels(metadata)
    train_index = np.asarray([index for index, row in enumerate(metadata) if row["split"] == "probe_train"], dtype=np.int64)
    eval_index = np.asarray([index for index, row in enumerate(metadata) if row["split"] == "eval"], dtype=np.int64)
    if not len(train_index) or len(eval_index) != 1000:
        raise ValueError(f"Expected non-empty probe train and locked 1,000-example eval split, found {len(train_index)}/{len(eval_index)}")

    results: list[dict] = []
    top_features: dict[str, dict] = {}
    audit_rows: list[dict] = []
    for role, matrix in features.items():
        metric, classifier, scaler, prediction = _fit_linear(
            matrix[train_index], labels[train_index], matrix[eval_index], labels[eval_index], cfg
        )
        results.append({
            "representation": role,
            "model": "linear_base_sae",
            "n_train": int(len(train_index)),
            "n_eval": int(len(eval_index)),
            **metric,
        })
        top_features[f"MultiBully::cyberbullying_detection::{role}"] = _top_features(classifier, cfg.feature_top_k)
        if role == "all":
            standardized = scaler.transform(matrix[eval_index])
            coefficient = classifier.coef_[0] if list(classifier.classes_)[1] == 1 else -classifier.coef_[0]
            for local_index, predicted in enumerate(prediction):
                if predicted or labels[eval_index[local_index]]:
                    continue
                source = metadata[eval_index[local_index]]
                top = np.argsort(-(standardized[local_index] * coefficient))[:5]
                audit_rows.append({
                    "sample_id": source["sample_id"],
                    "gold": "non-bully",
                    "prediction": "bully",
                    "image_path": source["image_path"],
                    "ocr_text": source.get("raw_fields", {}).get("ocr_text", ""),
                    "raw_labels": {key: source.get("raw_fields", {}).get(key, "") for key in ("image", "text", "harmful_score", "target")},
                    "top_bully_feature_contributions": [
                        {"feature_idx": int(index), "contribution": round(float(standardized[local_index, index] * coefficient[index]), 5)}
                        for index in top
                    ],
                })

    bilinear = _fit_bilinear(
        features["image"][train_index],
        features["prompt"][train_index],
        labels[train_index],
        features["image"][eval_index],
        features["prompt"][eval_index],
        labels[eval_index],
        cfg,
    )
    results.append({
        "representation": "image_x_prompt",
        "model": "low_rank_bilinear_base_sae",
        "n_train": int(len(train_index)),
        "n_eval": int(len(eval_index)),
        **bilinear,
    })
    with (output_dir / "probe_results.jsonl").open("w") as fd:
        for row in results:
            fd.write(json.dumps(row) + "\n")
    (output_dir / "top_features.json").write_text(json.dumps(top_features, indent=2) + "\n")
    audit_rows.sort(key=lambda row: row["top_bully_feature_contributions"][0]["contribution"], reverse=True)
    with (output_dir / "nonbully_false_positive_audit.jsonl").open("w") as fd:
        for row in audit_rows[:cfg.audit_limit]:
            fd.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "analysis_summary.json").write_text(json.dumps({
        "cache_dir": str(cache_dir),
        "label_source": "Img-Text-Label",
        "n_probe_train": int(len(train_index)),
        "n_eval": int(len(eval_index)),
        "native_hinglish_ocr": True,
        "translation_used": False,
        "false_positive_audit_count": min(len(audit_rows), cfg.audit_limit),
    }, indent=2) + "\n")
    logger.info("Wrote MultiBully probe and audit results to %s", output_dir)


if __name__ == "__main__":
    main()
