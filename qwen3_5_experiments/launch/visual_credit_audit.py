

from __future__ import annotations

import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from simple_parsing import Serializable, list_field, parse

from ..config import QwenResidualConfig
from ..qwen import (
    QwenSae,
    build_multimodal_inputs,
    classify_constrained,
    load_qwen3_model,
    resolve_target_module,
    score_label_candidates,
)
from ..task_labels import fit_scaled_logreg
from .analyze_exist_multilingual import _read_jsonl

logger = logging.getLogger(__name__)

SEMANTIC_CONDITIONS = {"original", "blank_image", "shuffled_image", "no_ocr"}
CONDITIONS = ("original", "blank_image", "shuffled_image", "no_ocr", "ocr_only", "neutral_description")


@dataclass(frozen=True)
class TaskSpec:
    key: str
    dataset_name: str
    task_name: str
    positive_label: str
    train_dir: str
    eval_dir: str
    train_split: str
    eval_split: str
    language: str | None = None
    label_field: str = "label"
    label_candidates: tuple[str, ...] = ()
    positive_candidates: tuple[str, ...] = ()


def _specs(root: str) -> dict[str, TaskSpec]:
    dense = f"{root}/qwen_residual_sae_cache"
    return {
        "fhm": TaskSpec("fhm", "Hateful_Memes_Dataset", "hateful_memes_binary", "hateful", f"{dense}/dense", f"{dense}/dense_val_test", "train", "test"),
        "mami": TaskSpec("mami", "MAMI", "misogyny_and_subtypes", "misogynous", f"{dense}/dense", f"{dense}/dense_val_test", "train", "test"),
        "harmeme": TaskSpec(
            "harmeme", "HarMeme", "harmfulness_and_target", "harmful", f"{dense}/dense", f"{dense}/dense_val_test", "train", "test",
            label_field="harmfulness", label_candidates=("not harmful", "somewhat harmful", "very harmful"),
            positive_candidates=("somewhat harmful", "very harmful"),
        ),
        "multibully": TaskSpec("multibully", "MultiBully", "cyberbullying_detection", "bully", f"{root}/multibully/dense_raw_v2", f"{root}/multibully/dense_raw_v2", "probe_train", "eval"),
        "exist_es": TaskSpec("exist_es", "EXIST2025_Memes", "sexism_detection", "sexist", f"{root}/exist_multilingual/dense_raw", f"{root}/exist_multilingual/dense_raw", "probe_train", "eval", language="es"),
    }


@dataclass
class VisualCreditAuditConfig(Serializable):
    dataset: str = "fhm"
    activations_root: str = "./outputs"
    output_dir: str = "./outputs/visual_credit_audit/fhm"
    device: str = "cuda:0"
    model_id: str = "Qwen/Qwen3.5-9B-Base"
    sae_repo_id: str = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
    sae_layer: int = 20
    C: float = 1.0
    max_iter: int = 3000
    calibration_fraction: float = 0.30
    alpha_grid: list[float] = list_field(0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
    seed: int = 20260808
    fhm_metadata_path: str = "ivl_plus_gemini_captions_complete.json"
    resume: bool = True


def _rows(directory: Path, spec: TaskSpec, split: str) -> list[dict]:
    return [
        row for row in _read_jsonl(directory / "manifest.jsonl")
        if row.get("dataset_name") == spec.dataset_name
        and row.get("task_name") == spec.task_name
        and row.get("split") == split
        and row.get("gold_fields", {}).get(spec.label_field) is not None
        and (spec.language is None or str(row.get("raw_fields", {}).get("language", "")).lower() == spec.language)
    ]


def _label(row: dict, spec: TaskSpec) -> int:
    value = str(row["gold_fields"][spec.label_field]).lower()
    if spec.positive_candidates:
        return int(value in {candidate.lower() for candidate in spec.positive_candidates})
    return int(value == spec.positive_label.lower())


def _extract_original_features(sae: QwenSae, cache_dir: Path, rows: list[dict], device: str) -> np.ndarray:
    by_shard: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_shard[row["shard_file"]].append(row)
    vectors = np.empty((len(rows), sae.cfg.d_sae), dtype=np.float32)
    row_index = {id(row): index for index, row in enumerate(rows)}
    for shard_name, shard_rows in sorted(by_shard.items()):
        tensors = load_file(str(cache_dir / shard_name))
        hidden = tensors["hidden"].to(device=device, dtype=torch.float32)
        for row in shard_rows:
            offset, length = int(row["offset"]), int(row["length"])
            with torch.no_grad():
                acts, _ = sae.encode(hidden[offset: offset + length])
            vectors[row_index[id(row)]] = acts.max(dim=0).values.cpu().numpy()
    return vectors


def _prompt_without_ocr(row: dict) -> str:
    return str(row["prompt_text"]).rsplit("\n\nMeme text:", 1)[0]


def _ocr_text(row: dict) -> str:
    return str(row.get("raw_fields", {}).get("ocr_text", row.get("meme_text", "")))


def _prompt(row: dict, condition: str) -> str:
    task_prompt = _prompt_without_ocr(row)
    ocr = _ocr_text(row)
    if condition in {"original", "blank_image", "shuffled_image"}:
        return str(row["prompt_text"])
    if condition == "no_ocr":
        return task_prompt
    if condition == "ocr_only":
        return f"Meme text:\n{ocr}"
    if condition == "neutral_description":
        return (
            "Describe the visible people, objects, layout, and relationships in this meme. "
            "Do not assess harmfulness, sexism, abuse, or any label.\n\n"
            f"Meme text:\n{ocr}"
        )
    raise ValueError(f"Unknown condition: {condition}")


def _blank_like(image: Image.Image) -> Image.Image:
    return Image.new("RGB", image.size, color=(128, 128, 128))


def _derangement(rows: list[dict], seed: int) -> dict[str, str]:
    ids = sorted(str(row["sample_id"]) for row in rows)
    shuffled = ids.copy()
    random.Random(seed).shuffle(shuffled)
    if len(ids) > 1:
        while any(source == target for source, target in zip(ids, shuffled)):
            shuffled = shuffled[1:] + shuffled[:1]
    return dict(zip(ids, shuffled))


def _capture_feature(model, processor, sae: QwenSae, image: Image.Image, prompt: str, cfg: VisualCreditAuditConfig) -> np.ndarray:
    inputs = build_multimodal_inputs(processor, image, prompt, cfg.device)
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden.detach()[0])

    handle = resolve_target_module(model, QwenResidualConfig(model_id=cfg.model_id, model_device=cfg.device, sae_layer=cfg.sae_layer)).register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs, use_cache=False)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("Layer hook did not capture hidden states")
    with torch.no_grad():
        acts, _ = sae.encode(captured[-1].to(torch.float32))
    return acts.max(dim=0).values.cpu().numpy()


def _native_logit(model, processor, image: Image.Image, prompt: str, spec: TaskSpec, device: str) -> float:
    if not spec.label_candidates:
        _prediction, _text, probability = classify_constrained(model, processor, image, prompt, device)
        probability = min(max(probability, 1e-6), 1 - 1e-6)
        return math.log(probability / (1 - probability))
    _label_name, scores, _confidence = score_label_candidates(model, processor, image, prompt, spec.label_candidates, device)
    positive = [scores[name] for name in spec.positive_candidates]
    negative = [scores[name] for name in spec.label_candidates if name not in spec.positive_candidates]
    return _logsumexp(positive) - _logsumexp(negative)


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _stratified_indices(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices: list[int] = []
    for label in (0, 1):
        group = np.flatnonzero(labels == label)
        rng.shuffle(group)
        indices.extend(group[:max(1, round(len(group) * fraction))])
    return np.asarray(sorted(indices), dtype=np.int64)


def _select_alpha(logits: np.ndarray, scores: np.ndarray, labels: np.ndarray, grid: list[float]) -> float:
    from sklearn.metrics import f1_score

    best_alpha, best_f1 = 0.0, -1.0
    for alpha in grid:
        metric = f1_score(labels, logits + alpha * scores > 0, average="macro", zero_division=0)
        if metric > best_f1:
            best_alpha, best_f1 = alpha, float(metric)
    return best_alpha


def _load_records(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    return {(str(row["sample_id"]), str(row["condition"])): row for row in _read_jsonl(path)}


def _metrics(labels: np.ndarray, prediction: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    false_positive = np.logical_and(labels == 0, prediction == 1)
    return {
        "n": int(len(labels)),
        "accuracy": round(float(accuracy_score(labels, prediction)), 4),
        "macro_f1": round(float(f1_score(labels, prediction, average="macro", zero_division=0)), 4),
        "false_positive_rate": round(float(false_positive.mean()), 4),
        "false_positive_count": int(false_positive.sum()),
    }


def _summarize(records: list[dict], report_ids: set[str], spec: TaskSpec, fhm_metadata_path: Path) -> dict:
    by_condition = {condition: {row["sample_id"]: row for row in records if row["condition"] == condition} for condition in CONDITIONS}
    result: dict[str, object] = {"dataset": spec.key, "report_n": len(report_ids), "conditions": {}, "visual_credit": {}}
    original = by_condition["original"]
    for condition, rows in by_condition.items():
        selected = [rows[sample_id] for sample_id in sorted(report_ids) if sample_id in rows]
        metrics = {}
        for system in ("native", "probe", "router"):
            prediction = np.asarray([row[f"{system}_prediction"] for row in selected], dtype=np.int32)
            labels = np.asarray([row["gold"] for row in selected], dtype=np.int32)
            metrics[system] = _metrics(labels, prediction)
            if condition != "original":
                original_scores = np.asarray([original[row["sample_id"]][f"{system}_gold_margin"] for row in selected])
                control_scores = np.asarray([row[f"{system}_gold_margin"] for row in selected])
                metrics[system]["original_minus_control_gold_margin"] = round(float((original_scores - control_scores).mean()), 4)
                original_fp = np.asarray([original[row["sample_id"]]["gold"] == 0 and original[row["sample_id"]][f"{system}_prediction"] == 1 for row in selected])
                control_fp = np.asarray([row["gold"] == 0 and row[f"{system}_prediction"] == 1 for row in selected])
                metrics[system]["false_positive_change_vs_original"] = int(control_fp.sum() - original_fp.sum())
        result["conditions"][condition] = {"semantic_classification": condition in SEMANTIC_CONDITIONS, "metrics": metrics}

    for system in ("native", "probe", "router"):
        selected = [original[sample_id] for sample_id in sorted(report_ids)]
        original_correct = np.asarray([row["gold"] == row[f"{system}_prediction"] for row in selected])
        blank_correct = np.asarray([by_condition["blank_image"][row["sample_id"]]["gold"] == by_condition["blank_image"][row["sample_id"]][f"{system}_prediction"] for row in selected])
        shuffled_correct = np.asarray([by_condition["shuffled_image"][row["sample_id"]]["gold"] == by_condition["shuffled_image"][row["sample_id"]][f"{system}_prediction"] for row in selected])
        credited = original_correct & ~(blank_correct & shuffled_correct)
        uncredited = original_correct & blank_correct & shuffled_correct
        result["visual_credit"][system] = {
            "original_correct_n": int(original_correct.sum()),
            "image_credited_correctness": round(float(credited.mean()), 4),
            "correct_but_image_uncredited_rate": round(float(uncredited.mean()), 4),
            "image_credited_share_of_original_correct": round(float(credited.sum() / max(original_correct.sum(), 1)), 4),
        }

    if spec.key == "fhm":
        result["fhm_pairs"] = _fhm_pair_metrics(by_condition["original"], report_ids, fhm_metadata_path)
    return result


def _normalise_fhm_id(value: object) -> str:
    return str(value).strip().lower().removesuffix(".png").removesuffix(".jpg")


def _fhm_pair_metrics(rows: dict[str, dict], report_ids: set[str], metadata_path: Path) -> dict:
    payload = json.loads(metadata_path.read_text())
    entries = payload.values() if isinstance(payload, dict) else payload
    metadata = {_normalise_fhm_id(row.get("id", row.get("img", ""))): row for row in entries if isinstance(row, dict)}
    report_rows = [row for sample_id, row in rows.items() if sample_id in report_ids]
    output: dict[str, dict] = {}
    for source, column in (("pseudo_img", "pseudo_img_idx"), ("pseudo_text", "pseudo_text_idx")):
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in report_rows:
            meta = metadata.get(_normalise_fhm_id(row["sample_id"]))
            value = meta.get(column) if meta else None
            if value is not None and str(value) not in {"", "None", "nan"}:
                groups[str(value)].append(row)
        pairs = []
        for group in groups.values():
            for positive in (item for item in group if item["gold"] == 1):
                for negative in (item for item in group if item["gold"] == 0):
                    pairs.append((positive, negative))
        output[source] = {"n_cross_label_pairs": len(pairs), "systems": {}}
        for system in ("native", "probe", "router"):
            if not pairs:
                output[source]["systems"][system] = {"both_correct_rate": None, "separated_rate": None, "positive_margin_rate": None}
                continue
            both_correct = [positive[f"{system}_prediction"] == 1 and negative[f"{system}_prediction"] == 0 for positive, negative in pairs]
            separated = [positive[f"{system}_prediction"] != negative[f"{system}_prediction"] for positive, negative in pairs]
            correct_margin = [positive[f"{system}_score"] > negative[f"{system}_score"] for positive, negative in pairs]
            output[source]["systems"][system] = {
                "both_correct_rate": round(float(np.mean(both_correct)), 4),
                "separated_rate": round(float(np.mean(separated)), 4),
                "positive_margin_rate": round(float(np.mean(correct_margin)), 4),
            }
    return output


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(VisualCreditAuditConfig)
    specs = _specs(cfg.activations_root)
    if cfg.dataset not in specs:
        raise ValueError(f"Unknown dataset {cfg.dataset!r}; choose from {sorted(specs)}")
    spec = specs[cfg.dataset]
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _rows(Path(spec.train_dir), spec, spec.train_split)
    eval_rows = _rows(Path(spec.eval_dir), spec, spec.eval_split)
    if not train_rows or not eval_rows:
        raise ValueError(f"Missing train/eval rows for {spec.key}: {len(train_rows)}/{len(eval_rows)}")
    logger.info("%s: train=%d eval=%d", spec.key, len(train_rows), len(eval_rows))

    sae = QwenSae.load_from_hub(cfg.sae_repo_id, cfg.sae_layer, device=cfg.device)
    train_features = _extract_original_features(sae, Path(spec.train_dir), train_rows, cfg.device)
    train_labels = np.asarray([_label(row, spec) for row in train_rows], dtype=np.int32)
    probe, scaler = fit_scaled_logreg(train_features, train_labels, C=cfg.C, max_iter=cfg.max_iter)

    calibration = _stratified_indices(np.asarray([_label(row, spec) for row in eval_rows]), cfg.calibration_fraction, cfg.seed)
    report_indices = np.setdiff1d(np.arange(len(eval_rows)), calibration)
    report_ids = {str(eval_rows[index]["sample_id"]) for index in report_indices}
    shuffle = _derangement(eval_rows, cfg.seed)
    by_id = {str(row["sample_id"]): row for row in eval_rows}
    records_path = output_dir / "per_sample.jsonl"
    existing = _load_records(records_path) if cfg.resume else {}

    model, processor = load_qwen3_model(QwenResidualConfig(model_id=cfg.model_id, model_device=cfg.device, sae_layer=cfg.sae_layer))
    original_values: dict[str, tuple[float, float]] = {}
    original_scores: list[float] = []
    original_logits: list[float] = []
    for row in eval_rows:
        sample_id = str(row["sample_id"])
        key = (sample_id, "original")
        cached = existing.get(key)
        if cached is not None:
            probe_score, native_score = float(cached["probe_score"]), float(cached["native_score"])
        else:
            image = Image.open(row["image_path"]).convert("RGB")
            prompt = _prompt(row, "original")
            feature = _capture_feature(model, processor, sae, image, prompt, cfg)
            probe_score = float(probe.decision_function(scaler.transform(feature[None]))[0])
            native_score = _native_logit(model, processor, image, prompt, spec, cfg.device)
        original_values[sample_id] = (probe_score, native_score)
        original_scores.append(probe_score)
        original_logits.append(native_score)
    original_scores_array = np.asarray(original_scores)
    original_logits_array = np.asarray(original_logits)
    score_mean, score_std = float(original_scores_array[calibration].mean()), max(float(original_scores_array[calibration].std()), 1e-8)
    alpha = _select_alpha(original_logits_array[calibration], (original_scores_array[calibration] - score_mean) / score_std, np.asarray([_label(eval_rows[index], spec) for index in calibration]), cfg.alpha_grid)
    logger.info("Frozen router: alpha=%g, calibration=%d, reporting=%d", alpha, len(calibration), len(report_indices))

    with records_path.open("a") as fd:
        for index, row in enumerate(eval_rows):
            sample_id = str(row["sample_id"])
            for condition in CONDITIONS:
                if (sample_id, condition) in existing:
                    continue
                source = by_id[shuffle[sample_id]] if condition == "shuffled_image" else row
                if condition == "original":
                    probe_score, native_score = original_values[sample_id]
                else:
                    source_image = Image.open(source["image_path"]).convert("RGB")
                    image = _blank_like(source_image) if condition in {"blank_image", "ocr_only"} else source_image
                    prompt = _prompt(row, condition)
                    feature = _capture_feature(model, processor, sae, image, prompt, cfg)
                    probe_score = float(probe.decision_function(scaler.transform(feature[None]))[0])
                    native_score = _native_logit(model, processor, image, prompt, spec, cfg.device)
                probe_z = (probe_score - score_mean) / score_std
                router_score = native_score + alpha * probe_z
                gold = _label(row, spec)
                record = {
                    "sample_id": sample_id,
                    "condition": condition,
                    "gold": gold,
                    "source_image_sample_id": str(source["sample_id"]),
                    "semantic_classification": condition in SEMANTIC_CONDITIONS,
                    "native_score": native_score,
                    "probe_score": probe_score,
                    "router_score": router_score,
                    "native_prediction": int(native_score > 0),
                    "probe_prediction": int(probe_score > 0),
                    "router_prediction": int(router_score > 0),
                    "native_gold_margin": native_score if gold else -native_score,
                    "probe_gold_margin": probe_score if gold else -probe_score,
                    "router_gold_margin": router_score if gold else -router_score,
                }
                fd.write(json.dumps(record) + "\n")
                fd.flush()
            if (index + 1) % 10 == 0 or index + 1 == len(eval_rows):
                logger.info("%s progress: %d/%d", spec.key, index + 1, len(eval_rows))

    records = _read_jsonl(records_path)
    summary = _summarize(records, report_ids, spec, Path(cfg.fhm_metadata_path))
    summary["router"] = {"alpha": alpha, "calibration_n": int(len(calibration)), "report_n": int(len(report_indices)), "zscore_mean": score_mean, "zscore_std": score_std}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Wrote visual-credit audit to %s", output_dir)


if __name__ == "__main__":
    main()