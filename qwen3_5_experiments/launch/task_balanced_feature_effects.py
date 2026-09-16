

from __future__ import annotations

import json
import logging
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from simple_parsing import Serializable, list_field, parse

from ..config import QwenResidualConfig
from ..qwen import (
    QwenSae,
    build_multimodal_inputs,
    get_yes_no_token_ids,
    load_qwen3_model,
)
from ..task_labels import fit_scaled_logreg
from .analyze_exist_multilingual import _read_jsonl

logger = logging.getLogger(__name__)

ROLE_NAMES = {"all": None, "image": 1, "prompt": 0, "generated": 2}
CARD_ROLES = tuple(ROLE_NAMES)
STATUS_NAMES = ("TP", "FP", "TN", "FN")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
STOPWORDS = {"and", "are", "but", "for", "from", "have", "not", "that", "the", "this", "with", "you"}


@dataclass(frozen=True)
class CardTask:
    name: str
    dataset_name: str
    task_name: str
    positive_label: str
    train_cache: str
    eval_cache: str
    train_split: str
    eval_split: str
    top_features_path: str
    top_features_key: str
    label_field: str = "label"
    language: str | None = None


def _tasks(root: str) -> list[CardTask]:
    dense = f"{root}/qwen_residual_sae_cache"
    base = f"{root}/qwen_probe_base_sae/top_features.json"
    return [
        CardTask("crisis", "Crisis_HateMM", "subtask_a_hate_speech", "hate speech", f"{dense}/dense", f"{dense}/dense_val_test", "train", "validation", base, "Crisis_HateMM::subtask_a_hate_speech"),
        CardTask("fhm", "Hateful_Memes_Dataset", "hateful_memes_binary", "hateful", f"{dense}/dense", f"{dense}/dense_val_test", "train", "test", base, "Hateful_Memes_Dataset::hateful_memes_binary"),
        CardTask("mami", "MAMI", "misogyny_and_subtypes", "misogynous", f"{dense}/dense", f"{dense}/dense_val_test", "train", "test", base, "MAMI::misogyny_and_subtypes"),
        CardTask("multioff", "MultiOFF", "offensive_detection", "offensive", f"{dense}/dense", f"{dense}/dense_val_test", "train", "test", base, "MultiOFF::offensive_detection"),
        CardTask("harmeme", "HarMeme", "harmfulness_and_target", "harmful", f"{dense}/dense", f"{dense}/dense_val_test", "train", "test", base, "HarMeme::harmfulness_and_target [binary]", "harmfulness"),
        CardTask("mmhs", "MMHS150K", "hate_category", "hate", f"{dense}/dense", f"{dense}/dense_val_test", "train", "test", base, "MMHS150K::hate_category [binary]"),
        CardTask("multibully", "MultiBully", "cyberbullying_detection", "bully", f"{root}/multibully/dense_raw_v2", f"{root}/multibully/dense_raw_v2", "probe_train", "eval", f"{root}/multibully/analysis_raw_ocr_v2/top_features.json", "MultiBully::cyberbullying_detection::all"),
        CardTask("exist_es", "EXIST2025_Memes", "sexism_detection", "sexist", f"{root}/exist_multilingual/dense_raw", f"{root}/exist_multilingual/dense_raw", "probe_train", "eval", f"{root}/exist_multilingual/analysis_raw_ocr/top_features.json", "bilingual_to_es::all", language="es"),
    ]


@dataclass
class FeatureEffectConfig(Serializable):
    activations_root: str = "./outputs"
    output_dir: str = "./outputs/qwen_task_balanced_feature_effects"
    model_id: str = "Qwen/Qwen3.5-9B-Base"
    sae_repo_id: str = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
    sae_layer: int = 20
    device: str = "cuda:0"
    max_features_per_direction: int = 12
    train_rows_per_label: int = 800
    eval_rows_per_label: int = 200
    examples_per_role_status: int = 2
    cross_dataset_quota: int = 2
    fega_features_per_task: int = 3
    fega_contexts_per_feature: int = 48
    random_feature_pool_size: int = 64
    seed: int = 20260816


def _label(row: dict, task: CardTask) -> int:
    value = str(row.get("gold_fields", {}).get(task.label_field, "")).lower()
    if task.name == "harmeme":
        return int(value != "not harmful")
    if task.name == "mmhs":
        return int(value != "nothate")
    return int(value == task.positive_label.lower())


def _task_rows(cache: Path, task: CardTask, split: str) -> list[dict]:
    rows = []
    for row in _read_jsonl(cache / "manifest.jsonl"):
        if row.get("dataset_name") != task.dataset_name or row.get("task_name") != task.task_name or row.get("split") != split:
            continue
        if task.language and str(row.get("raw_fields", {}).get("language", "")).lower() != task.language:
            continue
        if task.label_field not in row.get("gold_fields", {}):
            continue
        rows.append(row)
    return rows


def _balanced(rows: list[dict], task: CardTask, per_label: int, seed: int) -> list[dict]:
    by_label = {0: [], 1: []}
    for row in rows:
        by_label[_label(row, task)].append(row)
    rng = random.Random(seed)
    selected = []
    for label in (0, 1):
        values = sorted(by_label[label], key=lambda row: str(row["sample_id"]))
        rng.shuffle(values)
        selected.extend(values[:per_label])
    return sorted(selected, key=lambda row: str(row["sample_id"]))


def _feature_entries(task: CardTask, limit: int) -> dict[str, list[dict]]:
    data = json.loads(Path(task.top_features_path).read_text())
    directions = data.get(task.top_features_key, {})
    return {direction: directions.get(direction, [])[:limit] for direction in ("label_up", "label_down")}


def _push(bucket: list[dict], item: dict, quota: int) -> None:
    bucket.append(item)
    bucket.sort(key=lambda row: row["activation"], reverse=True)
    del bucket[quota:]


def _status(gold: int, pred: int) -> str:
    return ("TP", "FP", "TN", "FN")[(gold << 1) | pred] if False else (
        "TP" if gold and pred else "FN" if gold else "FP" if pred else "TN"
    )


def _example(row: dict, activation: float, gold: int, status: str, role: str) -> dict:
    return {
        "activation": round(float(activation), 6), "gold_binary": gold, "prediction_status": status,
        "role": role, "dataset_name": row.get("dataset_name"), "task_name": row.get("task_name"),
        "sample_id": row.get("sample_id"), "split": row.get("split"),
        "language": row.get("raw_fields", {}).get("language"), "image_path": row.get("image_path"),
        "ocr_text": row.get("raw_fields", {}).get("ocr_text", row.get("meme_text", "")),
        "generated_text": row.get("generated_text", ""),
    }


def _alignment_lookup(paths: list[Path]) -> dict[int, list[dict]]:
    lookup: dict[int, list[dict]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if "feature_idx" in value and "logit_effect" in value:
                    lookup[int(value["feature_idx"])].append(value)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        visit(json.loads(path.read_text()))
    return lookup


def _role_vectors(acts: torch.Tensor, roles: torch.Tensor, feature_t: torch.Tensor) -> dict[str, np.ndarray]:
    selected = acts[:, feature_t]
    output = {}
    for role, role_id in ROLE_NAMES.items():
        subset = selected if role_id is None else selected[roles == role_id]
        output[role] = (subset.max(dim=0).values if len(subset) else torch.zeros(len(feature_t), device=acts.device)).cpu().numpy()
    return output


def _vectors_for_rows(sae: QwenSae, cache: Path, rows: list[dict], features: list[int], device: str) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    by_shard: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_shard[row["shard_file"]].append(row)
    feature_t = torch.tensor(features, dtype=torch.long, device=device)
    for shard_name, shard_rows in sorted(by_shard.items()):
        tensors = load_file(str(cache / shard_name))
        hidden = tensors["hidden"].to(device=device, dtype=torch.float32)
        roles = tensors["token_roles"].to(device=device)
        for row in shard_rows:
            offset, length = int(row["offset"]), int(row["length"])
            with torch.no_grad():
                acts, _ = sae.encode(hidden[offset: offset + length])
            result[str(row["sample_id"])] = _role_vectors(acts, roles[offset: offset + length], feature_t)
    return result


def _lexical_match(items: dict[int, list[dict]]) -> dict | None:
    token_sets = {}
    counter = Counter()
    for label, rows in items.items():
        for row in rows:
            tokens = {token.lower() for token in TOKEN_RE.findall(str(row.get("ocr_text", ""))) if token.lower() not in STOPWORDS}
            token_sets[id(row)] = tokens
            if label == 1:
                counter.update(tokens)
    for marker, _count in counter.most_common():
        positive = next((row for row in items[1] if marker in token_sets[id(row)]), None)
        negative = next((row for row in items[0] if marker in token_sets[id(row)]), None)
        if positive and negative:
            return {"marker": marker, "harmful": positive, "benign": negative}
    return None


def _direct_yes_no(model, processor, image: Image.Image, prompt: str, device: str, hook=None) -> tuple[float, np.ndarray]:
    inputs = build_multimodal_inputs(processor, image, prompt, device)
    yes_ids, no_ids = get_yes_no_token_ids(processor)
    downstream: list[torch.Tensor] = []
    final_module = model.get_submodule("model.language_model.layers.31")
    def final_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        downstream.append(hidden.detach()[0, -1].float().cpu())
    handles = [final_module.register_forward_hook(final_hook)]
    if hook is not None:
        handles.append(model.get_submodule("model.language_model.layers.20").register_forward_hook(hook))
    try:
        with torch.no_grad():
            output = model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    logits = output.logits[0, -1].float()
    score = max(float(logits[index]) for index in yes_ids) - max(float(logits[index]) for index in no_ids)
    return score, downstream[-1].numpy()


def _ablation_hook(sae: QwenSae, feature_idx: int, role: str):
    feature = torch.tensor([feature_idx], dtype=torch.long, device=sae.W_enc.device)
    role_id = ROLE_NAMES[role]
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        state = hidden.detach().to(torch.float32)
        with torch.no_grad():
            acts, _ = sae.encode(state)
            positions = torch.ones(state.shape[1], dtype=torch.bool, device=state.device)
            # Roles are recreated from the image span in the caller and attached below.
            active = (acts[0, :, feature[0]] > 0) & hook.role_mask.to(state.device)
            modified = acts.clone()
            modified[0, active, feature[0]] = 0.0
            delta = sae.decode(modified) - sae.decode(acts)
            updated = state + delta
        return (updated.to(hidden.dtype), *output[1:]) if isinstance(output, tuple) else updated.to(hidden.dtype)
    hook.role_mask = torch.empty(0, dtype=torch.bool)  # type: ignore[attr-defined]
    return hook


def _role_mask_from_inputs(inputs: dict, processor, role: str) -> torch.Tensor:
    ids = inputs["input_ids"][0].detach().cpu()
    seq_len = len(ids)
    from ..qwen import _image_token_span
    start, end = _image_token_span(ids, processor, seq_len)
    mask = torch.zeros(seq_len, dtype=torch.bool)
    if role == "all":
        mask[:] = True
    elif role == "image":
        mask[start:end] = True
    elif role == "prompt":
        mask[:] = True
        mask[start:end] = False
    else:
        raise ValueError(f"FEGA does not support generated-only prefill role: {role}")
    return mask


def _context_pick(contexts: list[dict], max_contexts: int) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for context in contexts:
        groups[(context["task"], context["role"], context["status"], context.get("language", ""))].append(context)
    for values in groups.values():
        values.sort(key=lambda row: (-row["activation"], str(row["sample_id"])))
    picked = []
    keys = sorted(groups)
    while len(picked) < max_contexts and any(groups.values()):
        for key in keys:
            if groups[key] and len(picked) < max_contexts:
                picked.append(groups[key].pop(0))
    return picked


def _geometry(vectors: list[np.ndarray]) -> dict:
    if len(vectors) < 2:
        return {"effective_dimensionality": None, "directional_consistency": None, "n_clusters": 0, "cluster_sizes": [], "silhouette": None}
    matrix = np.stack(vectors)
    norms = np.linalg.norm(matrix, axis=1)
    mean = matrix.mean(axis=0)
    mean_norm = np.linalg.norm(mean)
    consistency = float(np.mean((matrix @ mean) / (norms * mean_norm + 1e-8))) if mean_norm else 0.0
    singular = np.linalg.svd(matrix, compute_uv=False) ** 2
    effective_dim = float(singular.sum() ** 2 / (np.square(singular).sum() + 1e-8))
    if len(vectors) < 4:
        return {"effective_dimensionality": round(effective_dim, 4), "directional_consistency": round(consistency, 4), "n_clusters": 1, "cluster_sizes": [len(vectors)], "silhouette": None}
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    normalized = matrix / (norms[:, None] + 1e-8)
    labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(normalized)
    counts = [int((labels == label).sum()) for label in range(2)]
    silhouette = float(silhouette_score(normalized, labels)) if min(counts) > 1 else None
    return {
        "effective_dimensionality": round(effective_dim, 4), "directional_consistency": round(consistency, 4),
        "n_clusters": 2, "cluster_sizes": counts, "silhouette": round(silhouette, 4) if silhouette is not None else None,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(FeatureEffectConfig)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tasks = _tasks(cfg.activations_root)
    alignment = _alignment_lookup([
        Path(cfg.activations_root) / "qwen_base_sae_feature_logit_alignment/feature_logit_alignment.json",
        Path(cfg.activations_root) / "multibully/feature_logit_alignment_raw_ocr_v2/feature_logit_alignment.json",
        Path(cfg.activations_root) / "exist_multilingual/feature_logit_alignment_raw_ocr/feature_logit_alignment.json",
    ])
    entries = {task.name: _feature_entries(task, cfg.max_features_per_direction) for task in tasks}
    task_features = {task.name: sorted({int(row["feature_idx"]) for direction in entries[task.name].values() for row in direction}) for task in tasks}
    global_card_features = sorted({feature for features in task_features.values() for feature in features})
    rng = random.Random(cfg.seed)
    random_pool = rng.sample([feature for feature in range(65536) if feature not in global_card_features], cfg.random_feature_pool_size)
    sae = QwenSae.load_from_hub(cfg.sae_repo_id, cfg.sae_layer, device=cfg.device)

    cards: dict[str, dict] = {}
    cross: dict[int, dict[str, dict[str, list[dict]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    contexts: dict[tuple[str, int], list[dict]] = defaultdict(list)
    fega_targets: dict[str, list[int]] = {}
    activity: dict[tuple[str, int], list[float]] = defaultdict(list)

    for task_index, task in enumerate(tasks):
        train_rows = _balanced(_task_rows(Path(task.train_cache), task, task.train_split), task, cfg.train_rows_per_label, cfg.seed + task_index)
        eval_rows = _balanced(_task_rows(Path(task.eval_cache), task, task.eval_split), task, cfg.eval_rows_per_label, cfg.seed + 100 + task_index)
        local_features = task_features[task.name]
        scan_features = sorted(set(global_card_features + random_pool))
        train_vectors = _vectors_for_rows(sae, Path(task.train_cache), train_rows, local_features, cfg.device)
        eval_vectors = _vectors_for_rows(sae, Path(task.eval_cache), eval_rows, scan_features, cfg.device)
        feature_to_column = {feature: index for index, feature in enumerate(scan_features)}
        local_columns = [feature_to_column[feature] for feature in local_features]
        train_x = np.stack([train_vectors[str(row["sample_id"])]["all"] for row in train_rows])
        train_y = np.asarray([_label(row, task) for row in train_rows])
        clf, scaler = fit_scaled_logreg(train_x, train_y)
        eval_x = np.stack([eval_vectors[str(row["sample_id"])]["all"][local_columns] for row in eval_rows])
        predictions = clf.predict(scaler.transform(eval_x))
        local_index = {feature: index for index, feature in enumerate(local_features)}
        buffers: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
        lexical: dict[tuple[int, int], list[dict]] = defaultdict(list)
        all_features = sorted({feature for values in entries[task.name].values() for feature in [int(row["feature_idx"]) for row in values]})
        for row, pred in zip(eval_rows, predictions):
            sample_id = str(row["sample_id"])
            gold = _label(row, task)
            status = _status(gold, int(pred))
            vectors = eval_vectors[sample_id]
            for feature in all_features:
                col = feature_to_column[feature]
                for role in CARD_ROLES:
                    value = float(vectors[role][col])
                    if value > 0:
                        item = _example(row, value, gold, status, role)
                        _push(buffers[(feature, role, status)], item, cfg.examples_per_role_status)
                _push(lexical[(feature, gold)], _example(row, float(vectors["all"][col]), gold, status, "all"), 100)
            # The cross-dataset card scan intentionally covers the GLOBAL
            # feature set and fixes the same quota for every dataset.
            for feature in global_card_features:
                col = feature_to_column[feature]
                for role in CARD_ROLES:
                    value = float(vectors[role][col])
                    if value > 0:
                        _push(cross[feature][task.dataset_name][role], _example(row, value, gold, status, role), cfg.cross_dataset_quota)
            for feature in local_features + random_pool:
                activity[(task.name, feature)].append(float(vectors["all"][feature_to_column[feature]]))

        target_features = [int(row["feature_idx"]) for row in entries[task.name]["label_up"][:cfg.fega_features_per_task]]
        if task.name not in {"fhm", "mami", "multibully", "exist_es"}:
            target_features = []
        fega_targets[task.name] = target_features
        for row, pred in zip(eval_rows, predictions):
            sample_id = str(row["sample_id"])
            gold = _label(row, task)
            status = _status(gold, int(pred))
            vectors = eval_vectors[sample_id]
            for feature in target_features:
                col = feature_to_column[feature]
                for role in ("image", "prompt"):
                    value = float(vectors[role][col])
                    if value > 0:
                        contexts[(task.name, feature)].append({
                            "task": task.name, "feature_idx": feature, "row": row, "sample_id": sample_id,
                            "gold": gold, "status": status, "role": role, "language": row.get("raw_fields", {}).get("language", ""),
                            "activation": value,
                        })

        cards[task.name] = {
            "task": {"dataset_name": task.dataset_name, "task_name": task.task_name, "language": task.language},
            "sampling": {"train_per_label_max": cfg.train_rows_per_label, "eval_per_label_max": cfg.eval_rows_per_label, "n_train": len(train_rows), "n_eval": len(eval_rows), "n_eval_positive": int(sum(_label(row, task) for row in eval_rows))},
            "card_probe": {"type": "balanced compact logistic regression", "features": local_features, "n_features": len(local_features)},
            "directions": {},
        }
        for direction, rows in entries[task.name].items():
            cards[task.name]["directions"][direction] = []
            for row in rows:
                feature = int(row["feature_idx"])
                cards[task.name]["directions"][direction].append({
                    "feature_idx": feature, "probe_weight": row.get("weight"), "output_logit_alignment": alignment.get(feature, []),
                    "examples_by_role_and_status": {role: {status: buffers[(feature, role, status)] for status in STATUS_NAMES} for role in CARD_ROLES},
                    "matched_lexical_harmful_benign": _lexical_match({0: lexical[(feature, 0)], 1: lexical[(feature, 1)]}),
                })
        logger.info("Prepared task-balanced card cohort for %s: train=%d eval=%d", task.name, len(train_rows), len(eval_rows))

    fega_targets_flat = [(task, feature) for task, features in fega_targets.items() for feature in features]
    random_match: dict[tuple[str, int], int] = {}
    for task, feature in fega_targets_flat:
        mean_activation = float(np.mean(activity[(task, feature)]))
        random_match[(task, feature)] = min(random_pool, key=lambda candidate: abs(float(np.mean(activity[(task, candidate)])) - mean_activation))

    qwen_cfg = QwenResidualConfig(model_id=cfg.model_id, model_device=cfg.device, sae_layer=cfg.sae_layer)
    model, processor = load_qwen3_model(qwen_cfg)
    effect_rows = []
    effect_summary: dict[str, dict] = {}
    task_map = {task.name: task for task in tasks}
    for task_name, feature in fega_targets_flat:
        selected_contexts = _context_pick(contexts[(task_name, feature)], cfg.fega_contexts_per_feature)
        downstream_vectors, random_vectors, deltas, random_deltas = [], [], [], []
        strata: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        random_feature = random_match[(task_name, feature)]
        for context in selected_contexts:
            row = context["row"]
            image = Image.open(row["image_path"]).convert("RGB")
            baseline_score, baseline_vector = _direct_yes_no(model, processor, image, row["prompt_text"], cfg.device)
            for feature_kind, ablated_feature in (("target", feature), ("matched_random", random_feature)):
                inputs = build_multimodal_inputs(processor, image, row["prompt_text"], cfg.device)
                hook = _ablation_hook(sae, ablated_feature, context["role"])
                hook.role_mask = _role_mask_from_inputs(inputs, processor, context["role"])  # type: ignore[attr-defined]
                score, vector = _direct_yes_no(model, processor, image, row["prompt_text"], cfg.device, hook)
                gold_signed = 1 if context["gold"] else -1
                effect = score - baseline_score
                vector_delta = vector - baseline_vector
                effect_row = {
                    "task": task_name, "feature_idx": feature, "matched_random_feature_idx": random_feature,
                    "feature_kind": feature_kind, "sample_id": context["sample_id"], "role": context["role"],
                    "status": context["status"], "language": context["language"], "activation": context["activation"],
                    "baseline_logit": baseline_score, "ablated_logit": score, "delta_logit": effect,
                    "delta_gold_margin": gold_signed * effect, "prelogit_delta_norm": float(np.linalg.norm(vector_delta)),
                    "reconstruction_error_preserved": True,
                }
                effect_rows.append(effect_row)
                if feature_kind == "target":
                    downstream_vectors.append(vector_delta); deltas.append(effect)
                    strata[(context["role"], context["status"], str(context["language"]))].append(effect)
                else:
                    random_vectors.append(vector_delta); random_deltas.append(effect)
        key = f"{task_name}::{feature}"
        effect_summary[key] = {
            "task": task_name, "feature_idx": feature, "matched_random_feature_idx": random_feature,
            "n_contexts": len(selected_contexts), "context_sampling": "balanced active contexts across role/status/language, capped",
            "target": {"mean_delta_logit": round(float(np.mean(deltas)), 6) if deltas else None, "mean_abs_delta_logit": round(float(np.mean(np.abs(deltas))), 6) if deltas else None, **_geometry(downstream_vectors)},
            "matched_random": {"mean_delta_logit": round(float(np.mean(random_deltas)), 6) if random_deltas else None, "mean_abs_delta_logit": round(float(np.mean(np.abs(random_deltas))), 6) if random_deltas else None, **_geometry(random_vectors)},
            "strata": {
                f"role={role}|status={status}|language={language or 'na'}": {
                    "n": len(values), "mean_delta_logit": round(float(np.mean(values)), 6),
                    "mean_abs_delta_logit": round(float(np.mean(np.abs(values))), 6),
                }
                for (role, status, language), values in strata.items()
            },
        }
        logger.info("FEGA %s: %d active contexts", key, len(selected_contexts))

    effect_by_feature = {(value["task"], value["feature_idx"]): value for value in effect_summary.values()}
    for task_name, card in cards.items():
        for directions in card["directions"].values():
            for feature_card in directions:
                feature_card["targeted_ablation_effect"] = effect_by_feature.get((task_name, int(feature_card["feature_idx"])))
    cross_cards = {str(feature): {dataset: dict(roles) for dataset, roles in datasets.items()} for feature, datasets in cross.items()}
    (output / "task_balanced_feature_cards.json").write_text(json.dumps(cards, indent=2, ensure_ascii=False) + "\n")
    (output / "cross_dataset_quota_cards.json").write_text(json.dumps(cross_cards, indent=2, ensure_ascii=False) + "\n")
    with (output / "fega_context_effects.jsonl").open("w") as fd:
        for row in effect_rows:
            fd.write(json.dumps(row) + "\n")
    (output / "fega_summary.json").write_text(json.dumps(effect_summary, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps({
        "tasks": list(cards), "fega_targets": len(fega_targets_flat), "fega_contexts": len(effect_rows) // 2,
        "task_balanced_cards": "task-local balanced train/eval cohorts; cross-dataset cards use equal per-dataset quotas",
    }, indent=2) + "\n")
    logger.info("Wrote task-balanced cards and FEGA effects to %s", output)


if __name__ == "__main__":
    main()