

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from simple_parsing import parse, Serializable, list_field

from ..config import QwenResidualConfig
from ..qwen import (
    QwenSae,
    classify_constrained,
    load_qwen3_model,
    resolve_target_module,
    score_label_candidates,
)
from ..task_labels import (
    BINARY_TASK_DATASET,
    BINARY_TASK_FIELD,
    BINARY_TASKS,
    MULTICLASS_TASKS,
    fit_scaled_logreg,
)
from .analyze_feature_logit_alignment import (
    _build_unembed_eff,
    _get_yes_no_ids,
    _load_base_sae_w_dec,
    _load_qwen_lm_weights,
    _yes_no_unembed_diff,
)
from .direct_logit_projection import (
    _extract_features_with_ids,
    _logit_diff_from_prediction_row,
    _read_jsonl,
)

logger = logging.getLogger(__name__)


@dataclass
class CausalMediationConfig(Serializable):
    predictions_jsonl: str = ""       # baseline_proper.jsonl
    train_cache_dir: str = "./qwen_residual_sae_cache/dense"
    val_cache_dir: str = "./qwen_residual_sae_cache/dense_val_test"
    alignment_json_path: str = "./qwen_base_sae_feature_logit_alignment/feature_logit_alignment.json"
    output_dir: str = "./qwen_causal_mediation"

    sae_repo_id: str = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
    sae_layer: int = 20
    model_id: str = "Qwen/Qwen3.5-9B-Base"
    model_device: str = "cuda:0"
    hook_module_path: str = "model.language_model.layers.20"
    aggregation: str = "max"

    eval_splits: list[str] = list_field("validation", "val", "dev", "test", "test_seen")
    C: float = 1.0
    max_iter: int = 3000
    min_class_samples: int = 20

    top_k_features: int = 20             # features ablated/patched per intervention
    n_examples_knockout: int = 15         # Intervention A example budget per task
    n_pairs_patch: int = 5                # Intervention B/C target budget per task


def _make_ablate_patch_hook(base_sae: QwenSae, feature_indices: list[int], patch_value: float | None):
    """Forward hook: at the target layer, zero (patch_value=None) or set
    (patch_value=v) the given base-SAE feature activations uniformly across
    all token positions, and add the resulting reconstruction delta to the
    residual stream. This is the live-model analogue of the pooled
    feature-vector edit used for the probe score."""
    idx_t = None

    def hook(_module, _inputs, output):
        nonlocal idx_t
        hidden = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            h = hidden.detach().to(torch.float32)
            if idx_t is None or idx_t.device != h.device:
                idx_t = torch.tensor(feature_indices, dtype=torch.long, device=h.device)
            acts, _ = base_sae.encode(h)          # [..., T, d_sae]
            recon_orig = base_sae.decode(acts)
            acts_mod = acts.clone()
            if patch_value is None:
                acts_mod[..., idx_t] = 0.0
            else:
                acts_mod[..., idx_t] = float(patch_value)
            recon_mod = base_sae.decode(acts_mod)
            delta = recon_mod - recon_orig
            updated = (h + delta).to(hidden.dtype).contiguous()
        if isinstance(output, tuple):
            return (updated, *output[1:])
        return updated

    return hook


def _run_with_hook(model, processor, target_module, hook, image_path: str, row: dict, device: str) -> float | None:
    """Re-run the model's classification decision with `hook` attached and
    return the resulting logit_diff (positive => favours positive class)."""
    image = Image.open(image_path).convert("RGB")
    handle = target_module.register_forward_hook(hook)
    try:
        strategy = row.get("decode_strategy")
        prompt_text = row.get("prompt_text")
        if strategy == "yes_no":
            _is_pos, _text, conf_gap = classify_constrained(model, processor, image, prompt_text, device)
            p = min(max(float(conf_gap), 1e-6), 1 - 1e-6)
            return math.log(p / (1 - p))
        elif strategy == "label_score":
            label_texts = row.get("candidate_labels")
            if not label_texts:
                return None
            _best, label_scores, _conf = score_label_candidates(model, processor, image, prompt_text, label_texts, device)
            row2 = {**row, "label_scores": label_scores}
            pos_group = row.get("_pos_group")
            neg_group = row.get("_neg_group")
            return _logit_diff_from_prediction_row(row2, row.get("_pos_label", ""), row.get("_neg_label", ""), pos_group, neg_group)
    finally:
        handle.remove()
    return None


def _compute_global_routing_features(cfg: CausalMediationConfig, top_k: int) -> list[int]:
    """Rank ALL base-SAE decoder directions by |dot(w_dec_i, yes_no_unembed_diff)|,
    independent of any task-specific probe -- the "J-lens routing features"."""
    w_dec = _load_base_sae_w_dec(cfg.sae_repo_id, cfg.sae_layer)   # [d_sae, d_model] cpu
    lm_head, norm_w = _load_qwen_lm_weights(cfg.model_id)
    W_U_eff = _build_unembed_eff(lm_head, norm_w)
    yes_ids, no_ids = _get_yes_no_ids(cfg.model_id)
    unembed_diff = _yes_no_unembed_diff(W_U_eff, yes_ids, no_ids)
    logit_effect = (w_dec.float() @ unembed_diff.float())  # [d_sae]
    top_idx = torch.argsort(-logit_effect.abs())[:top_k].tolist()
    logger.info("Global routing features (top %d by |logit_effect|): %s", top_k, top_idx[:10])
    return [int(i) for i in top_idx]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(CausalMediationConfig)

    assert cfg.predictions_jsonl, "--predictions_jsonl is required (baseline_proper.jsonl)"

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = Path(cfg.train_cache_dir)
    val_dir = Path(cfg.val_cache_dir)

    base_sae = QwenSae.load_from_hub(cfg.sae_repo_id, cfg.sae_layer, device=cfg.model_device)
    logger.info("Base SAE loaded: d_model=%d d_sae=%d k=%d", base_sae.cfg.d_model, base_sae.cfg.d_sae, base_sae.cfg.k)

    qwen_cfg = QwenResidualConfig(
        model_id=cfg.model_id, sae_repo_id=cfg.sae_repo_id, sae_layer=cfg.sae_layer,
        hook_module_path=cfg.hook_module_path, model_device=cfg.model_device,
    )
    model, processor = load_qwen3_model(qwen_cfg)
    target_module = resolve_target_module(model, qwen_cfg)

    # Global routing features (Intervention C), computed once.
    global_routing_features = _compute_global_routing_features(cfg, cfg.top_k_features)

    alignment_data: dict = {}
    align_path = Path(cfg.alignment_json_path)
    if align_path.exists():
        alignment_data = json.loads(align_path.read_text())
    else:
        logger.warning("Missing feature_logit_alignment.json at %s; silent-feature selection will fall back to top_features weight order.", align_path)

    pred_rows = _read_jsonl(Path(cfg.predictions_jsonl))
    logger.info("Loaded %d prediction rows from %s", len(pred_rows), cfg.predictions_jsonl)

    train_manifest = _read_jsonl(train_dir / "manifest.jsonl")
    train_manifest = [r for r in train_manifest if r.get("split") == "train"]
    eval_splits = set(cfg.eval_splits)
    val_manifest = _read_jsonl(val_dir / "manifest.jsonl")
    val_manifest = [r for r in val_manifest if r.get("split") in eval_splits]

    task_specs = []
    for task_name, pos_label in BINARY_TASKS.items():
        ds = BINARY_TASK_DATASET[task_name]
        field = BINARY_TASK_FIELD[task_name]
        task_specs.append((ds, task_name, field, pos_label, None, None, None))
    for task_name, spec in MULTICLASS_TASKS.items():
        pos_group = {c for c in spec.classes if spec.collapse(c) == spec.binary_positive}
        neg_group = {c for c in spec.classes if spec.collapse(c) == spec.binary_negative}
        task_specs.append((
            spec.dataset_name, task_name, spec.field,
            spec.binary_positive, spec.binary_negative, pos_group, neg_group,
        ))

    all_rows: list[dict] = []
    summary: dict[str, dict] = {}

    for ds, task_name, field, pos_label, neg_label, pos_group, neg_group in task_specs:
        task_key_binary = f"{ds}::{task_name}" if pos_group is None else f"{ds}::{task_name} [binary]"
        logger.info("=== Task: %s ===", task_key_binary)

        train_rows = [r for r in train_manifest if r.get("task_name") == task_name and r.get("gold_fields")]
        val_rows = [r for r in val_manifest if r.get("task_name") == task_name and r.get("gold_fields")]
        if not train_rows or not val_rows:
            logger.info("Missing train/eval rows; skipping")
            continue

        X_tr, y_tr_raw, _ = _extract_features_with_ids(base_sae, train_dir, train_rows, field, cfg.aggregation, cfg.model_device)
        if pos_group is None:
            y_tr_bin = np.array([1 if v.lower() == pos_label.lower() else 0 for v in y_tr_raw], dtype=np.int32)
        else:
            y_tr_bin = np.array([1 if v in pos_group else 0 for v in y_tr_raw], dtype=np.int32)
        if int(y_tr_bin.sum()) < cfg.min_class_samples or int((y_tr_bin == 0).sum()) < cfg.min_class_samples:
            logger.info("Too few train class samples; skipping")
            continue
        clf, scaler = fit_scaled_logreg(X_tr, y_tr_bin, C=cfg.C, max_iter=cfg.max_iter)

        X_va, y_va_raw, sample_ids = _extract_features_with_ids(base_sae, val_dir, val_rows, field, cfg.aggregation, cfg.model_device)
        if len(X_va) < 20:
            logger.info("Too few eval rows; skipping")
            continue
        if pos_group is None:
            y_va_bin = np.array([1 if v.lower() == pos_label.lower() else 0 for v in y_va_raw], dtype=np.int32)
        else:
            y_va_bin = np.array([1 if v in pos_group else 0 for v in y_va_raw], dtype=np.int32)
        X_va_scaled = scaler.transform(X_va)
        probe_score = clf.decision_function(X_va_scaled)

        pred_by_id = {
            str(r.get("sample_id")): r
            for r in pred_rows
            if r.get("dataset_name") == ds and r.get("task_name") == task_name
        }

        # Silent features (from static alignment) — fall back to top_features
        # order if the alignment file is missing this task.
        task_align = alignment_data.get(task_key_binary, {})
        silent_entries = [e for e in task_align.get("label_up", []) if e.get("quadrant") == "silent"]
        silent_entries.sort(key=lambda e: -abs(e["probe_weight"]))
        silent_features = [int(e["feature_idx"]) for e in silent_entries[: cfg.top_k_features]]
        if not silent_features:
            logger.warning("No 'silent' features found for %s in alignment file; skipping this task.", task_key_binary)
            continue
        routed_features = global_routing_features[: cfg.top_k_features]

        # ── Build per-sample records (baseline logit_diff + probe_score + row) ──
        records = []
        for i, sid in enumerate(sample_ids):
            row = pred_by_id.get(sid)
            if row is None:
                continue
            ld = _logit_diff_from_prediction_row(row, pos_label, neg_label or "", pos_group, neg_group)
            if ld is None:
                continue
            row_aug = {**row, "_pos_label": pos_label, "_neg_label": neg_label, "_pos_group": pos_group, "_neg_group": neg_group}
            records.append({
                "i": i, "sample_id": sid, "row": row_aug,
                "gold": int(y_va_bin[i]), "probe_score": float(probe_score[i]),
                "logit_diff": ld, "base_pred_positive": ld > 0,
            })

        pos_correct_probe = [r for r in records if r["gold"] == 1 and r["probe_score"] > 0]
        neg_correct_probe = [r for r in records if r["gold"] == 0 and r["probe_score"] <= 0]

        # ── Intervention A set: gold=1, probe right, baseline wrong ──
        set_a = [r for r in pos_correct_probe if not r["base_pred_positive"]]
        set_a.sort(key=lambda r: -r["probe_score"])
        set_a = set_a[: cfg.n_examples_knockout]

        # ── Donor (Intervention B/C): most confidently-correct positive ──
        donor = max(pos_correct_probe, key=lambda r: r["probe_score"]) if pos_correct_probe else None
        donor_vec = X_va[donor["i"]] if donor is not None else None

        # ── Targets (Intervention B/C): gold=0, probe right (negative), baseline wrong (positive) ──
        targets = [r for r in neg_correct_probe if r["base_pred_positive"]]
        targets.sort(key=lambda r: -r["logit_diff"])
        targets = targets[: cfg.n_pairs_patch]

        task_summary = {"n_knockout": len(set_a), "n_patch_targets": len(targets), "donor_found": donor is not None}

        # ── Intervention A: silent-feature knockout ──
        delta_probe_a, delta_logit_a = [], []
        for r in set_a:
            X_mod = X_va[r["i"]].copy()
            X_mod[silent_features] = 0.0
            probe_after = float(clf.decision_function(scaler.transform(X_mod.reshape(1, -1)))[0])
            hook = _make_ablate_patch_hook(base_sae, silent_features, patch_value=None)
            logit_after = _run_with_hook(model, processor, target_module, hook, r["row"].get("image_path"), r["row"], cfg.model_device)
            if logit_after is None:
                continue
            d_probe = probe_after - r["probe_score"]
            d_logit = logit_after - r["logit_diff"]
            delta_probe_a.append(d_probe)
            delta_logit_a.append(d_logit)
            all_rows.append({
                "task_key": task_key_binary, "intervention": "A_silent_knockout", "sample_id": r["sample_id"],
                "probe_before": r["probe_score"], "probe_after": probe_after, "delta_probe": d_probe,
                "logit_before": r["logit_diff"], "logit_after": logit_after, "delta_logit": d_logit,
            })
        if delta_probe_a:
            task_summary["A_silent_knockout"] = {
                "n": len(delta_probe_a),
                "mean_abs_delta_probe": round(float(np.mean(np.abs(delta_probe_a))), 4),
                "mean_abs_delta_logit": round(float(np.mean(np.abs(delta_logit_a))), 4),
                "mean_delta_probe": round(float(np.mean(delta_probe_a)), 4),
                "mean_delta_logit": round(float(np.mean(delta_logit_a)), 4),
            }
            logger.info("  [A knockout] n=%d mean|Δprobe|=%.4f mean|Δlogit|=%.4f",
                        len(delta_probe_a), task_summary["A_silent_knockout"]["mean_abs_delta_probe"],
                        task_summary["A_silent_knockout"]["mean_abs_delta_logit"])

        # ── Interventions B & C: patch donor's activations into confounder targets ──
        for label, feature_set in (("B_silent_patch", silent_features), ("C_routed_patch", routed_features)):
            if donor is None or not targets:
                continue
            delta_probe_x, delta_logit_x = [], []
            for r in targets:
                X_mod = X_va[r["i"]].copy()
                patch_vals = donor_vec[feature_set]
                X_mod[feature_set] = patch_vals
                probe_after = float(clf.decision_function(scaler.transform(X_mod.reshape(1, -1)))[0])

                hook = _make_multi_patch_hook(base_sae, feature_set, patch_vals)
                logit_after = _run_with_hook(model, processor, target_module, hook, r["row"].get("image_path"), r["row"], cfg.model_device)
                if logit_after is None:
                    continue
                d_probe = probe_after - r["probe_score"]
                d_logit = logit_after - r["logit_diff"]
                delta_probe_x.append(d_probe)
                delta_logit_x.append(d_logit)
                all_rows.append({
                    "task_key": task_key_binary, "intervention": label, "sample_id": r["sample_id"],
                    "probe_before": r["probe_score"], "probe_after": probe_after, "delta_probe": d_probe,
                    "logit_before": r["logit_diff"], "logit_after": logit_after, "delta_logit": d_logit,
                })
            if delta_probe_x:
                task_summary[label] = {
                    "n": len(delta_probe_x),
                    "mean_abs_delta_probe": round(float(np.mean(np.abs(delta_probe_x))), 4),
                    "mean_abs_delta_logit": round(float(np.mean(np.abs(delta_logit_x))), 4),
                    "mean_delta_probe": round(float(np.mean(delta_probe_x)), 4),
                    "mean_delta_logit": round(float(np.mean(delta_logit_x)), 4),
                }
                logger.info("  [%s] n=%d mean|Δprobe|=%.4f mean|Δlogit|=%.4f",
                            label, len(delta_probe_x), task_summary[label]["mean_abs_delta_probe"],
                            task_summary[label]["mean_abs_delta_logit"])

        summary[task_key_binary] = task_summary

    results_path = output_dir / "causal_mediation_results.jsonl"
    with results_path.open("w") as fd:
        for r in all_rows:
            fd.write(json.dumps(r) + "\n")
    logger.info("Per-example results written to %s (%d rows)", results_path, len(all_rows))

    summary_path = output_dir / "causal_mediation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary written to %s", summary_path)

    print(f"\n{'Task':<45} {'intervention':<20} {'n':>4} {'mean|Δprobe|':>13} {'mean|Δlogit|':>13}")
    print("-" * 100)
    for task_key, s in summary.items():
        for label in ("A_silent_knockout", "B_silent_patch", "C_routed_patch"):
            if label not in s:
                continue
            m = s[label]
            print(f"  {task_key:<43} {label:<20} {m['n']:>4} {m['mean_abs_delta_probe']:>13.4f} {m['mean_abs_delta_logit']:>13.4f}")


def _make_multi_patch_hook(base_sae: QwenSae, feature_indices: list[int], patch_values: np.ndarray):
    """Like _make_ablate_patch_hook but with a distinct patch value per feature."""
    idx_t = None
    vals_t = None

    def hook(_module, _inputs, output):
        nonlocal idx_t, vals_t
        hidden = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            h = hidden.detach().to(torch.float32)
            if idx_t is None or idx_t.device != h.device:
                idx_t = torch.tensor(feature_indices, dtype=torch.long, device=h.device)
                vals_t = torch.tensor(patch_values, dtype=torch.float32, device=h.device)
            acts, _ = base_sae.encode(h)
            recon_orig = base_sae.decode(acts)
            acts_mod = acts.clone()
            acts_mod[..., idx_t] = vals_t
            recon_mod = base_sae.decode(acts_mod)
            delta = recon_mod - recon_orig
            updated = (h + delta).to(hidden.dtype).contiguous()
        if isinstance(output, tuple):
            return (updated, *output[1:])
        return updated

    return hook


if __name__ == "__main__":
    main()
