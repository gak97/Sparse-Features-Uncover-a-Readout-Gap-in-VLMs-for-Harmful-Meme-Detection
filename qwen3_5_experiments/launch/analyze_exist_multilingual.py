

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from simple_parsing import Serializable, parse

from ..qwen import QwenSae
from ..task_labels import fit_scaled_logreg, predict_scaled
from .probe_fhm_bilinear import ROLE_GENERATED, ROLE_IMAGE, ROLE_PROMPT, _train_bilinear

logger = logging.getLogger(__name__)

ROLE_NAMES = {
    "all": None,
    "image": ROLE_IMAGE,
    "prompt": ROLE_PROMPT,
    "generated": ROLE_GENERATED,
}

WOMEN_CENTRED_SPANISH = re.compile(
    r"\b(mujer(?:es)?|femin(?:ismo|ista|istas)|madre(?:s)?|ni(?:ña|ñas)|chica(?:s)?|ella(?:s)?|esposa(?:s)?)\b",
    flags=re.IGNORECASE,
)


@dataclass
class ExistAnalysisConfig(Serializable):
    cache_dir: str = "./outputs/exist_multilingual/dense_raw"
    output_dir: str = "./outputs/exist_multilingual/analysis_raw_ocr"
    mami_top_features_path: str = "./outputs/qwen_probe_base_sae/top_features.json"
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


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fd:
        return [json.loads(line) for line in fd if line.strip()]


def _language(row: dict) -> str:
    return str(row.get("raw_fields", {}).get("language", row.get("language", ""))).lower()


def _label(row: dict) -> int:
    value = str(row["gold_fields"]["label"]).lower()
    if value not in {"sexist", "non-sexist"}:
        raise ValueError(f"Unexpected EXIST label {value!r} for {row.get('sample_id')}")
    return int(value == "sexist")


def _extract_features(
    sae: QwenSae, cache_dir: Path, rows: list[dict], device: str
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Encode each cached sample once and max-pool all requested token roles."""
    by_shard: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_shard[row["shard_file"]].append(row)

    vectors: dict[str, list[np.ndarray]] = {name: [] for name in ROLE_NAMES}
    metadata: list[dict] = []
    for shard_name, shard_rows in sorted(by_shard.items()):
        tensors = load_file(str(cache_dir / shard_name))
        hidden = tensors["hidden"].to(device=device, dtype=torch.float32)
        roles = tensors["token_roles"]
        for row in shard_rows:
            offset, length = int(row["offset"]), int(row["length"])
            sample_hidden = hidden[offset: offset + length]
            if not len(sample_hidden):
                continue
            with torch.no_grad():
                acts, _ = sae.encode(sample_hidden)
            sample_roles = roles[offset: offset + length].to(device)
            pooled: dict[str, torch.Tensor] = {"all": acts.max(dim=0).values}
            for name, role in ROLE_NAMES.items():
                if role is None:
                    continue
                role_acts = acts[sample_roles == role]
                pooled[name] = (
                    role_acts.max(dim=0).values
                    if len(role_acts)
                    else torch.zeros(sae.cfg.d_sae, device=device)
                )
            for name, vector in pooled.items():
                vectors[name].append(vector.cpu().float().numpy())
            metadata.append(row)

    if not metadata:
        raise ValueError(f"No usable cached samples in {cache_dir}")
    result = {name: np.asarray(values, dtype=np.float32) for name, values in vectors.items()}
    logger.info("Extracted %d samples with role matrices: %s", len(metadata), {k: v.shape for k, v in result.items()})
    return metadata, result


def _indices(metadata: list[dict], split: str, language: str) -> np.ndarray:
    return np.asarray(
        [i for i, row in enumerate(metadata) if row.get("split") == split and _language(row) == language],
        dtype=np.int64,
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_sexist": round(float(f1_score(y_true, y_pred, average="binary", zero_division=0)), 4),
        "pred_sexist_rate": round(float(y_pred.mean()), 4),
    }


def _fit_linear(
    X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray, cfg: ExistAnalysisConfig
) -> tuple[dict, object, object, np.ndarray]:
    clf, scaler = fit_scaled_logreg(X_train, y_train, C=cfg.C, max_iter=cfg.max_iter)
    prediction = predict_scaled(clf, scaler, X_eval)
    return _metrics(y_eval, prediction), clf, scaler, prediction


def _top_features(clf, top_k: int) -> dict:
    coefficient = clf.coef_[0].copy()
    if list(clf.classes_)[1] != 1:
        coefficient *= -1
    positive = np.argsort(-coefficient)[:top_k]
    negative = np.argsort(coefficient)[:top_k]
    return {
        "label_up": [{"feature_idx": int(index), "weight": round(float(coefficient[index]), 6)} for index in positive],
        "label_down": [{"feature_idx": int(index), "weight": round(float(coefficient[index]), 6)} for index in negative],
    }


def _fit_bilinear(
    X_image_train: np.ndarray,
    X_prompt_train: np.ndarray,
    y_train: np.ndarray,
    X_image_eval: np.ndarray,
    X_prompt_eval: np.ndarray,
    y_eval: np.ndarray,
    cfg: ExistAnalysisConfig,
) -> dict:
    from sklearn.preprocessing import StandardScaler

    selected_image = np.argsort(-X_image_train.var(axis=0))[:cfg.top_k_features]
    selected_prompt = np.argsort(-X_prompt_train.var(axis=0))[:cfg.top_k_features]
    image_scaler, prompt_scaler = StandardScaler(), StandardScaler()
    image_train = image_scaler.fit_transform(X_image_train[:, selected_image])
    prompt_train = prompt_scaler.fit_transform(X_prompt_train[:, selected_prompt])
    image_eval = image_scaler.transform(X_image_eval[:, selected_image])
    prompt_eval = prompt_scaler.transform(X_prompt_eval[:, selected_prompt])
    model, internal_accuracy = _train_bilinear(
        image_train, prompt_train, y_train,
        rank=cfg.bilinear_rank,
        n_epochs=cfg.bilinear_epochs,
        lr=cfg.bilinear_lr,
        weight_decay=cfg.bilinear_weight_decay,
        patience=cfg.bilinear_patience,
        device=cfg.device,
    )
    with torch.no_grad():
        logits = model(
            torch.tensor(image_eval, dtype=torch.float32, device=cfg.device),
            torch.tensor(prompt_eval, dtype=torch.float32, device=cfg.device),
        )
    prediction = (logits.cpu().numpy() >= 0).astype(np.int32)
    return {**_metrics(y_eval, prediction), "internal_val_accuracy": round(float(internal_accuracy), 4)}


def _mami_overlap(top_features: dict, path: Path, top_k: int) -> dict:
    if not path.exists():
        return {"status": "missing_mami_features", "path": str(path)}
    existing = json.loads(path.read_text())
    mami = existing.get("MAMI::misogyny_and_subtypes", {})
    mami_up = {int(row["feature_idx"]) for row in mami.get("label_up", [])[:top_k]}
    if not mami_up:
        return {"status": "missing_mami_misogyny_features", "path": str(path)}
    overlaps: dict[str, dict] = {}
    for condition, directions in top_features.items():
        exist_up = {int(row["feature_idx"]) for row in directions.get("label_up", [])[:top_k]}
        shared = sorted(exist_up & mami_up)
        union = exist_up | mami_up
        overlaps[condition] = {
            "n_shared_label_up": len(shared),
            "jaccard_label_up": round(len(shared) / len(union), 4) if union else 0.0,
            "shared_feature_indices": shared,
        }
    return {"status": "ok", "mami_source": str(path), "top_k": top_k, "conditions": overlaps}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(ExistAnalysisConfig)
    cache_dir, output_dir = Path(cfg.cache_dir), Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_jsonl(cache_dir / "manifest.jsonl")
    manifest = [row for row in manifest if row.get("task_name") == "sexism_detection" and row.get("gold_fields")]
    if {row.get("split") for row in manifest} != {"probe_train", "eval"}:
        raise ValueError("EXIST analysis requires both probe_train and eval rows in one cache")

    sae = QwenSae.load_from_hub(cfg.sae_repo_id, cfg.sae_layer, device=cfg.device)
    metadata, features = _extract_features(sae, cache_dir, manifest, cfg.device)
    labels = np.asarray([_label(row) for row in metadata], dtype=np.int32)
    train_en, train_es = _indices(metadata, "probe_train", "en"), _indices(metadata, "probe_train", "es")
    eval_en, eval_es = _indices(metadata, "eval", "en"), _indices(metadata, "eval", "es")
    if min(len(train_en), len(train_es), len(eval_en), len(eval_es)) == 0:
        raise ValueError("Both languages need non-empty train and held-out evaluation partitions")

    conditions = {
        "en_to_es": (train_en, eval_es),
        "es_to_en": (train_es, eval_en),
        "bilingual_to_en": (np.concatenate([train_en, train_es]), eval_en),
        "bilingual_to_es": (np.concatenate([train_en, train_es]), eval_es),
    }
    results: list[dict] = []
    discovered: dict[str, dict] = {}
    audit_rows: list[dict] = []

    for condition, (train_index, eval_index) in conditions.items():
        logger.info("=== %s: train=%d eval=%d ===", condition, len(train_index), len(eval_index))
        for role, matrix in features.items():
            metric, clf, scaler, prediction = _fit_linear(
                matrix[train_index], labels[train_index], matrix[eval_index], labels[eval_index], cfg
            )
            record = {
                "condition": condition, "representation": role, "model": "linear_base_sae",
                "n_train": int(len(train_index)), "n_eval": int(len(eval_index)), **metric,
            }
            results.append(record)
            discovered[f"{condition}::{role}"] = _top_features(clf, cfg.feature_top_k)
            if condition == "bilingual_to_es" and role == "all":
                scaled_eval = scaler.transform(matrix[eval_index])
                positive_coefficient = clf.coef_[0] if list(clf.classes_)[1] == 1 else -clf.coef_[0]
                for local_index, predicted in enumerate(prediction):
                    source = metadata[eval_index[local_index]]
                    ocr_text = str(source.get("raw_fields", {}).get("ocr_text", ""))
                    if predicted or labels[eval_index[local_index]] or not WOMEN_CENTRED_SPANISH.search(ocr_text):
                        continue
                    top = np.argsort(-(scaled_eval[local_index] * positive_coefficient))[:5]
                    audit_rows.append({
                        "sample_id": source["sample_id"], "language": "es", "gold": "non-sexist",
                        "prediction": "sexist", "image_path": source["image_path"],
                        "ocr_text": ocr_text,
                        "prompt_text": source.get("prompt_text", ""),
                        "generated_text": source.get("generated_text", ""),
                        "top_sexist_feature_contributions": [
                            {"feature_idx": int(index), "contribution": round(float(scaled_eval[local_index, index] * positive_coefficient[index]), 5)}
                            for index in top
                        ],
                    })

        bilinear = _fit_bilinear(
            features["image"][train_index], features["prompt"][train_index], labels[train_index],
            features["image"][eval_index], features["prompt"][eval_index], labels[eval_index], cfg,
        )
        results.append({
            "condition": condition, "representation": "image_x_prompt", "model": "low_rank_bilinear_base_sae",
            "n_train": int(len(train_index)), "n_eval": int(len(eval_index)), **bilinear,
        })

    with (output_dir / "transfer_results.jsonl").open("w") as fd:
        for row in results:
            fd.write(json.dumps(row) + "\n")
    (output_dir / "top_features.json").write_text(json.dumps(discovered, indent=2) + "\n")
    (output_dir / "mami_feature_overlap.json").write_text(
        json.dumps(_mami_overlap(discovered, Path(cfg.mami_top_features_path), cfg.feature_top_k), indent=2) + "\n"
    )
    audit_rows.sort(key=lambda row: row["top_sexist_feature_contributions"][0]["contribution"] if row["top_sexist_feature_contributions"] else 0, reverse=True)
    with (output_dir / "spanish_benign_women_fp_audit.jsonl").open("w") as fd:
        for row in audit_rows[:cfg.audit_limit]:
            fd.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "analysis_summary.json").write_text(json.dumps({
        "cache_dir": str(cache_dir), "n_samples": len(metadata),
        "n_probe_train_en": int(len(train_en)), "n_probe_train_es": int(len(train_es)),
        "n_eval_en": int(len(eval_en)), "n_eval_es": int(len(eval_es)),
        "native_language_ocr": True,
        "translation_used": False,
        "women_centred_audit_rule": WOMEN_CENTRED_SPANISH.pattern,
        "false_positive_audit_count": min(len(audit_rows), cfg.audit_limit),
    }, indent=2) + "\n")
    logger.info("Wrote transfer, feature-overlap, and false-positive audit results to %s", output_dir)


if __name__ == "__main__":
    main()
