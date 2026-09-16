

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from simple_parsing import Serializable, parse

from ..qwen import QwenSae
from ..task_labels import fit_scaled_logreg
from .analyze_exist_multilingual import ROLE_NAMES, _extract_features, _indices, _metrics, _read_jsonl

logger = logging.getLogger(__name__)

MAMI_DATASET = "MAMI"
MAMI_TASK = "misogyny_and_subtypes"


@dataclass
class ExistMamiTransferConfig(Serializable):
    mami_train_cache_dir: str = "./outputs/qwen_residual_sae_cache/dense"
    mami_test_cache_dir: str = "./outputs/qwen_residual_sae_cache/dense_val_test"
    exist_cache_dir: str = "./outputs/exist_multilingual/dense_raw"
    output_dir: str = "./outputs/exist_multilingual/mami_transfer_raw_ocr"
    device: str = "cuda:0"
    sae_repo_id: str = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
    sae_layer: int = 20
    C: float = 1.0
    max_iter: int = 3000


def _binary_label(rows: list[dict], field: str, positive: str) -> np.ndarray:
    return np.asarray(
        [int(str(row["gold_fields"][field]).lower() == positive.lower()) for row in rows], dtype=np.int32
    )


def _fit_evaluate(
    X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray, cfg: ExistMamiTransferConfig
) -> dict:
    from sklearn.metrics import f1_score

    clf, scaler = fit_scaled_logreg(X_train, y_train, C=cfg.C, max_iter=cfg.max_iter)
    prediction = clf.predict(scaler.transform(X_eval))
    metrics = _metrics(y_eval, prediction)
    metrics["probe_feature_top_20"] = [
        int(index)
        for index in np.argsort(-(clf.coef_[0] if list(clf.classes_)[1] == 1 else -clf.coef_[0]))[:20]
    ]
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(ExistMamiTransferConfig)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mami_train_dir = Path(cfg.mami_train_cache_dir)
    mami_test_dir = Path(cfg.mami_test_cache_dir)
    exist_dir = Path(cfg.exist_cache_dir)
    mami_train_rows = [
        row for row in _read_jsonl(mami_train_dir / "manifest.jsonl")
        if row.get("dataset_name") == MAMI_DATASET and row.get("task_name") == MAMI_TASK and row.get("split") == "train"
    ]
    mami_test_rows = [
        row for row in _read_jsonl(mami_test_dir / "manifest.jsonl")
        if row.get("dataset_name") == MAMI_DATASET and row.get("task_name") == MAMI_TASK and row.get("split") == "test"
    ]
    exist_rows = [
        row for row in _read_jsonl(exist_dir / "manifest.jsonl") if row.get("task_name") == "sexism_detection"
    ]
    if not all((mami_train_rows, mami_test_rows, exist_rows)):
        raise ValueError("Missing MAMI train/test or EXIST rows in the supplied cache manifests")

    sae = QwenSae.load_from_hub(cfg.sae_repo_id, cfg.sae_layer, device=cfg.device)
    mami_train_meta, mami_train = _extract_features(sae, mami_train_dir, mami_train_rows, cfg.device)
    mami_test_meta, mami_test = _extract_features(sae, mami_test_dir, mami_test_rows, cfg.device)
    exist_meta, exist = _extract_features(sae, exist_dir, exist_rows, cfg.device)
    y_mami_train = _binary_label(mami_train_meta, "label", "misogynous")
    y_mami_test = _binary_label(mami_test_meta, "label", "misogynous")
    y_exist = _binary_label(exist_meta, "label", "sexist")
    exist_train = np.concatenate([_indices(exist_meta, "probe_train", "en"), _indices(exist_meta, "probe_train", "es")])
    exist_eval = {language: _indices(exist_meta, "eval", language) for language in ("en", "es")}

    results: list[dict] = []
    for role in ROLE_NAMES:
        logger.info("=== role=%s ===", role)
        for language, evaluation in exist_eval.items():
            metrics = _fit_evaluate(mami_train[role], y_mami_train, exist[role][evaluation], y_exist[evaluation], cfg)
            results.append({
                "direction": f"mami_to_exist_{language}", "representation": role,
                "n_train": int(len(y_mami_train)), "n_eval": int(len(evaluation)), **metrics,
            })
        metrics = _fit_evaluate(exist[role][exist_train], y_exist[exist_train], mami_test[role], y_mami_test, cfg)
        results.append({
            "direction": "exist_bilingual_to_mami", "representation": role,
            "n_train": int(len(exist_train)), "n_eval": int(len(y_mami_test)), **metrics,
        })

    results_path = output_dir / "mami_exist_transfer_results.jsonl"
    with results_path.open("w") as fd:
        for row in results:
            fd.write(json.dumps(row) + "\n")
    (output_dir / "summary.json").write_text(json.dumps({
        "mami_train_rows": len(y_mami_train), "mami_test_rows": len(y_mami_test),
        "exist_bilingual_train_rows": len(exist_train),
        "exist_eval_rows": {language: len(rows) for language, rows in exist_eval.items()},
        "native_spanish_ocr": True,
    }, indent=2) + "\n")
    logger.info("Wrote MAMI <-> EXIST transfer results to %s", results_path)


if __name__ == "__main__":
    main()
