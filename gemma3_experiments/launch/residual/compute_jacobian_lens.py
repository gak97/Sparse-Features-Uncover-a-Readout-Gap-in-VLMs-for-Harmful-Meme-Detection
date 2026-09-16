

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from simple_parsing import ArgumentGenerationMode, parse

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Default target tokens for hate-speech meme analysis
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TARGET_TOKENS: list[str] = [
    # Binary classification labels
    "hateful", "benign", "safe",
    # Hate type labels
    "racist", "racism", "Racist",
    "misogynistic", "misogyny", "sexist", "Sexist",
    "offensive", "hate",
    # Identity group terms
    "Muslim", "Black", "women", "Jewish", "gay",
    # Linguistic markers
    "slur", "joke",
    # Yes/no constrained decode tokens
    "yes", "no",
    # Harmfulness labels
    "harmful",
]


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JacobianLensConfig:
    # Model
    model_id: str = "google/gemma-3-12b-it"
    model_device: str = "cuda:0"

    # Which residual-stream layers to probe (0-indexed transformer block indices)
    # For Gemma 12B (46 layers): 31 ≈ 67% depth, 41 ≈ 89% depth
    layer_indices: list[int] = field(default_factory=lambda: [31, 41])

    # Target vocabulary tokens — will be tokenised and deduplicated
    target_tokens: list[str] = field(default_factory=lambda: DEFAULT_TARGET_TOKENS)

    fhm_cache_dir: str = ""           # dense cache dir (manifest.jsonl + image paths)
    fhm_meta_path: str = ""           # ivl_plus_gemini_captions_complete.json
    splits: list[str] = field(default_factory=lambda: ["validation"])

    # Dataset generalisation (2026-07-18): compute_jacobian_lens.py was originally
    # FHM-only. dataset_name/gold_field_name/prompt_labels/prompt_task_description
    # let the same pipeline run on MAMI/MMHS150K/Crisis_HateMM/HarMeme too.
    # Defaults reproduce the exact original FHM behaviour unchanged.
    dataset_name: str = "Hateful_Memes_Dataset"
    gold_field_name: str = "label"           # manifest gold_fields key (HarMeme uses "harmfulness")
    # Pseudo-pair filtering only makes sense for FHM (via IVL confounder metadata);
    # other datasets have no such structure and should sample directly instead.
    use_pseudo_pairs: bool = True
    # Multi-class label-choice prompt (schema.primary_field), avoids the FHM yes/no
    # "does this qualify as <last label>" framing which is invalid for >2-class
    # schemas (e.g. MMHS150K's 6-class hate_category). Defaults match
    # HATEFUL_MEMES_SCHEMA so FHM runs are unaffected.
    prompt_labels: list[str] = field(default_factory=lambda: ["not hateful", "hateful"])
    prompt_task_description: str = "Decide whether the meme is hateful or not hateful."
    # Keep the ORIGINAL byte-for-byte FHM yes/no prompt when True (default). Set to
    # False for non-FHM datasets so the label-choice prompt (schema-correct for
    # >2-class tasks) is used instead -- see compute_jacobian_lens_*.sub.
    use_legacy_fhm_yesno_prompt: bool = True

    # Sampling
    n_samples: int = 200              # max FHM samples to average over
    random_seed: int = 42
    include_pseudo_pairs_only: bool = True  # only use samples that are in a pseudo-pair
    # Optional: path to a JSON file containing a list of sample IDs to use.
    # When provided, only those samples are used (overrides random sampling and
    # pseudo_pair filtering).  Format: ["10362", "89071", ...]  (bare numeric IDs
    # as they appear in the manifest, or the last :: segment of composite keys).
    sample_ids_filter_path: str = ""

    # Computation
    # "last_gen": gradient at last generated token position (the yes/no decision)
    # "all":      average gradient over all token positions
    # "image":    average gradient over image token positions only
    position_selection: str = "last_gen"

    # If True, compute n_target separate backward passes to get the full per-token
    # Jacobian matrix [n_target, d_model].  If False, compute one backward for the
    # aggregate (sum of target logits) — faster but loses per-token resolution.
    compute_per_token_jacobian: bool = True

    # Output
    output_dir: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_target_token_ids(
    tokenizer: Any,
    target_tokens: list[str],
) -> dict[str, int]:
    """Tokenise each target token and return {token_str: first_token_id}."""
    result: dict[str, int] = {}
    for tok_str in target_tokens:
        # Try with and without leading space (BPE tokenisers differ)
        for s in (tok_str, " " + tok_str, tok_str.lower(), " " + tok_str.lower()):
            ids = tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                result[tok_str] = ids[0]
                break
        else:
            # Fall back: take the first sub-token ID
            ids = tokenizer.encode(tok_str, add_special_tokens=False)
            if ids:
                result[tok_str] = ids[0]
                logger.warning(
                    "Target token %r tokenises to %d sub-tokens; using first ID %d",
                    tok_str, len(ids), ids[0],
                )
    logger.info("Resolved %d / %d target tokens", len(result), len(target_tokens))
    return result


def _load_fhm_rows(
    cache_dir: Path,
    splits: set[str],
    fhm_meta_path: str,
    include_pseudo_pairs_only: bool,
    dataset_name: str = "Hateful_Memes_Dataset",
) -> list[dict[str, Any]]:
    """Load manifest rows for `dataset_name`, optionally restricted to FHM pseudo-pair members.

    Pseudo-pair filtering (`include_pseudo_pairs_only`) only applies when
    `dataset_name == "Hateful_Memes_Dataset"` -- other datasets have no such
    confounder-pair structure and are returned split-filtered only.
    """
    manifest_path = cache_dir / "manifest.jsonl"
    assert manifest_path.exists(), f"Missing manifest at {manifest_path}"

    all_rows = [
        r for r in _read_jsonl(manifest_path)
        if r.get("dataset_name") == dataset_name
        and r.get("split") in splits
    ]

    if not include_pseudo_pairs_only or dataset_name != "Hateful_Memes_Dataset":
        return all_rows

    # Load IVL metadata for pseudo-pair IDs
    if not fhm_meta_path:
        logger.warning("include_pseudo_pairs_only=True but no fhm_meta_path; using all rows")
        return all_rows

    meta_by_id: dict[str, dict[str, Any]] = {}
    with open(fhm_meta_path) as f:
        meta_list = json.load(f)
    for m in meta_list:
        sid = str(m.get("id", ""))
        if sid:
            meta_by_id[sid] = m

    # Keep rows whose sample_id appears in a cross-label pseudo pair
    # (pseudo_img_idx or pseudo_text_idx group contains both labels)
    def _normalise_id(sid: str) -> str:
        return sid.split("/")[-1].split(".")[0]

    # Build group → sample set mapping
    img_groups: dict[Any, list[dict]] = {}
    txt_groups: dict[Any, list[dict]] = {}
    for row in all_rows:
        sid = _normalise_id(row["sample_id"])
        m = meta_by_id.get(sid)
        if m is None:
            continue
        img_grp = m.get("pseudo_img_idx")
        txt_grp = m.get("pseudo_text_idx")
        if img_grp is not None:
            img_groups.setdefault(img_grp, []).append(row)
        if txt_grp is not None:
            txt_groups.setdefault(txt_grp, []).append(row)

    pair_sids: set[str] = set()
    for grp_rows in (*img_groups.values(), *txt_groups.values()):
        labels = {r["gold_fields"].get("label", r.get("gold_label", "")) for r in grp_rows}
        if len(labels) > 1:  # cross-label group → proper pseudo-pair
            for r in grp_rows:
                pair_sids.add(r["sample_id"])

    filtered = [r for r in all_rows if r["sample_id"] in pair_sids]
    logger.info(
        "Pseudo-pair filter: %d / %d FHM rows kept (%d unique pseudo-pair groups)",
        len(filtered), len(all_rows), len(pair_sids),
    )
    return filtered


def _build_model_input(
    row: dict[str, Any],
    processor: Any,
    device: str,
    prompt_labels: list[str] | None = None,
    prompt_task_description: str = "Decide whether the meme is hateful or not hateful.",
    use_legacy_fhm_yesno_prompt: bool = True,
) -> dict[str, torch.Tensor] | None:
    pass
    from PIL import Image

    image_path = row.get("image_path", "")
    meme_text = row.get("meme_text", "")

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        logger.debug("Cannot open image %s: %s", image_path, exc)
        return None

    if use_legacy_fhm_yesno_prompt:
        prompt = (
            "<start_of_turn>user\n"
            f"<start_of_image>\n"
            f"Meme text: {meme_text}\n"
            "Is this meme hateful? Answer only: yes or no.<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
    else:
        labels = prompt_labels or ["not hateful", "hateful"]
        labels_str = ", ".join(f"'{label}'" for label in labels)
        prompt = (
            "<start_of_turn>user\n"
            f"<start_of_image>\n"
            f"Meme text: {meme_text}\n"
            f"Task: {prompt_task_description}\n"
            f"Choose exactly one label from: {labels_str}.<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

    try:
        inputs = processor(
            text=prompt,
            images=img,
            return_tensors="pt",
            padding=False,
        )
        return {k: v.to(device) for k, v in inputs.items()}
    except Exception as exc:
        logger.debug("Processor error for %s: %s", image_path, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Core Jacobian computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_jacobian_rows_for_sample(
    model: Any,
    inputs: dict[str, torch.Tensor],
    layer_idx: int,
    target_token_ids: list[int],
    position_selection: str,
    compute_per_token: bool,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    pass
    grads_collected: list[torch.Tensor] = []

    def _backward_hook(module: Any, grad_input: Any, grad_output: tuple) -> None:
        # grad_output[0]: [batch, seq_len, d_model] gradient w.r.t. layer output
        g = grad_output[0]
        if g is None:
            return
        grads_collected.append(g.detach().float())

    handle = model.language_model.layers[layer_idx].register_full_backward_hook(
        _backward_hook
    )

    try:
        with torch.enable_grad():
            # Forward pass — no gradient checkpointing needed since we intercept via hook
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs, use_cache=False)

            logits = outputs.logits.float()  # [1, seq_len, vocab]

            # Select position
            if position_selection == "last_gen":
                pos_logits = logits[0, -1, :]  # [vocab]
            elif position_selection == "all":
                pos_logits = logits[0, :, :].mean(0)  # [vocab]
            else:
                pos_logits = logits[0, -1, :]

            if compute_per_token:
                # n_target backward passes — each gives one row of the Jacobian
                j_rows: list[torch.Tensor] = []
                for tok_id in target_token_ids:
                    grads_collected.clear()
                    target_logit = pos_logits[tok_id]
                    target_logit.backward(retain_graph=True)
                    if grads_collected:
                        g = grads_collected[-1]  # [1, seq_len, d_model]
                        # Average over batch and position → [d_model]
                        j_row = g[0].mean(0)
                        j_rows.append(j_row.to(dtype))
                    else:
                        j_rows.append(torch.zeros(model.config.hidden_size, dtype=dtype))
                    grads_collected.clear()

                return torch.stack(j_rows)  # [n_target, d_model]

            else:
                # Single backward for aggregate
                agg_logit = pos_logits[target_token_ids].sum()
                agg_logit.backward()
                if grads_collected:
                    g = grads_collected[0]  # [1, seq_len, d_model]
                    j_agg = g[0].mean(0).to(dtype)  # [d_model]
                    # Return as [1, d_model] so caller can always treat as matrix
                    return j_agg.unsqueeze(0)
                return None

    except Exception as exc:
        logger.warning("Jacobian computation failed for layer %d: %s", layer_idx, exc)
        return None
    finally:
        handle.remove()
        # Clear gradient buffers
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _run(cfg: JacobianLensConfig) -> None:
    import random as _random

    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info("Loading model %s on %s …", cfg.model_id, cfg.model_device)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map=cfg.model_device,
        low_cpu_mem_usage=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(cfg.model_id)
    tokenizer = processor.tokenizer

    # ── Resolve target token IDs ──────────────────────────────────────────────
    target_id_map = _resolve_target_token_ids(tokenizer, cfg.target_tokens)
    # Stable ordering
    target_token_strs = sorted(target_id_map.keys())
    target_ids = [target_id_map[s] for s in target_token_strs]

    (out_dir / "target_token_ids.json").write_text(
        json.dumps({s: target_id_map[s] for s in target_token_strs}, indent=2)
    )
    logger.info("Target tokens: %s", target_token_strs)

    # ── Load dataset rows (FHM or otherwise) ──────────────────────────────────
    cache_dir = Path(cfg.fhm_cache_dir)
    rows = _load_fhm_rows(
        cache_dir,
        set(cfg.splits),
        cfg.fhm_meta_path,
        cfg.include_pseudo_pairs_only and cfg.use_pseudo_pairs,
        dataset_name=cfg.dataset_name,
    )

    if cfg.sample_ids_filter_path:
        allowed = set(json.loads(Path(cfg.sample_ids_filter_path).read_text()))
        rows = [r for r in rows if r["sample_id"] in allowed
                or r["sample_id"].split("::")[-1] in allowed]
        logger.info(
            "sample_ids_filter: %d rows match %d requested IDs",
            len(rows), len(allowed),
        )
    else:
        _rng = _random.Random(cfg.random_seed)
        _rng.shuffle(rows)
        rows = rows[: cfg.n_samples]
    logger.info("Using %d FHM rows for Jacobian averaging", len(rows))

    # ── Gold labels (used for the per-sample record, not just pseudo-pairs) ───
    pair_label_by_sid: dict[str, str] = {}  # sample_id → gold label string
    for row in rows:
        gl = row.get("gold_fields", {}).get(cfg.gold_field_name) or row.get("gold_label", "")
        pair_label_by_sid[row["sample_id"]] = str(gl)

    # ── Accumulate Jacobian matrices per layer ────────────────────────────────
    n_target = len(target_ids)
    d_model = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size

    # Accumulators: {layer_idx: running_sum [n_target, d_model], count: int}
    j_accum: dict[int, torch.Tensor] = {li: torch.zeros(n_target, d_model) for li in cfg.layer_indices}
    j_count: dict[int, int] = {li: 0 for li in cfg.layer_indices}

    per_sample_records: list[dict[str, Any]] = []

    t0 = time.time()
    for sample_idx, row in enumerate(rows):
        inputs = _build_model_input(
            row, processor, cfg.model_device,
            prompt_labels=cfg.prompt_labels,
            prompt_task_description=cfg.prompt_task_description,
            use_legacy_fhm_yesno_prompt=cfg.use_legacy_fhm_yesno_prompt,
        )
        if inputs is None:
            continue

        sample_record: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "split": row.get("split", ""),
            "gold_label": pair_label_by_sid.get(row["sample_id"], ""),
            "layers": {},
        }

        for layer_idx in cfg.layer_indices:
            j_mat = _compute_jacobian_rows_for_sample(
                model=model,
                inputs=inputs,
                layer_idx=layer_idx,
                target_token_ids=target_ids,
                position_selection=cfg.position_selection,
                compute_per_token=cfg.compute_per_token_jacobian,
            )
            if j_mat is None:
                continue

            # j_mat: [n_target, d_model] or [1, d_model] for aggregate
            j_accum[layer_idx] += j_mat.cpu()
            j_count[layer_idx] += 1

            # Per-sample: store norm of each target-token Jacobian row
            j_mat_cpu = j_mat.cpu()
            per_tok_norms = j_mat_cpu.norm(dim=-1).tolist()
            sample_record["layers"][str(layer_idx)] = {
                "per_token_norm": dict(zip(target_token_strs, per_tok_norms)),
                "aggregate_norm": float(j_mat_cpu.norm()),
            }

        per_sample_records.append(sample_record)

        if (sample_idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            logger.info(
                "Processed %d / %d samples (%.1f s elapsed)",
                sample_idx + 1, len(rows), elapsed,
            )

    # ── Save averaged Jacobian matrices ──────────────────────────────────────
    try:
        from safetensors.torch import save_file as safe_save
        use_safetensors = True
    except ImportError:
        use_safetensors = False

    layer_summary: dict[str, Any] = {}
    for layer_idx in cfg.layer_indices:
        n = j_count[layer_idx]
        if n == 0:
            logger.warning("No successful samples for layer %d", layer_idx)
            continue

        j_avg = j_accum[layer_idx] / n  # [n_target, d_model]

        fname = f"jlens_matrix_layer{layer_idx}"
        if use_safetensors:
            safe_save({f"jlens_l{layer_idx}": j_avg.contiguous()}, str(out_dir / f"{fname}.safetensors"))
        else:
            torch.save(j_avg, out_dir / f"{fname}.pt")

        # Top tokens by mean absolute Jacobian norm
        row_norms = j_avg.norm(dim=-1)  # [n_target]
        top_k = min(10, n_target)
        top_idx = row_norms.argsort(descending=True)[:top_k].tolist()
        top_tokens_by_norm = [(target_token_strs[i], float(row_norms[i])) for i in top_idx]

        layer_summary[str(layer_idx)] = {
            "n_samples_used": n,
            "d_model": d_model,
            "n_target_tokens": n_target,
            "j_avg_frobenius_norm": float(j_avg.norm("fro")),
            "top_tokens_by_j_norm": top_tokens_by_norm,
        }
        logger.info(
            "Layer %d: averaged over %d samples  |J_avg|_F = %.4f  top tokens: %s",
            layer_idx, n, float(j_avg.norm("fro")),
            [t for t, _ in top_tokens_by_norm[:5]],
        )

    # ── FHM pair analysis ─────────────────────────────────────────────────────
    pair_analysis: dict[str, Any] = {}
    if len(per_sample_records) > 1:
        # Split by gold label and compute mean norms per token per group
        for layer_idx in cfg.layer_indices:
            hateful_norms = {tok: [] for tok in target_token_strs}
            benign_norms = {tok: [] for tok in target_token_strs}
            for rec in per_sample_records:
                ldata = rec["layers"].get(str(layer_idx))
                if ldata is None:
                    continue
                grp = hateful_norms if rec["gold_label"] == "hateful" else benign_norms
                for tok, v in ldata["per_token_norm"].items():
                    grp[tok].append(v)

            tok_deltas = {}
            for tok in target_token_strs:
                h_vals = hateful_norms[tok]
                b_vals = benign_norms[tok]
                if h_vals and b_vals:
                    tok_deltas[tok] = {
                        "hateful_mean": float(np.mean(h_vals)),
                        "benign_mean": float(np.mean(b_vals)),
                        "delta": float(np.mean(h_vals) - np.mean(b_vals)),
                    }

            top_discriminative = sorted(tok_deltas.items(), key=lambda x: abs(x[1]["delta"]), reverse=True)[:10]
            pair_analysis[str(layer_idx)] = {
                "n_hateful": sum(1 for r in per_sample_records if r["gold_label"] == "hateful"),
                "n_benign": sum(1 for r in per_sample_records if r["gold_label"] != "hateful"),
                "top_discriminative_tokens": [
                    {"token": tok, **vals} for tok, vals in top_discriminative
                ],
            }

    # ── Write outputs ─────────────────────────────────────────────────────────
    with (out_dir / "jlens_per_sample.jsonl").open("w") as f:
        for rec in per_sample_records:
            f.write(json.dumps(rec) + "\n")

    (out_dir / "jlens_pair_analysis.json").write_text(json.dumps(pair_analysis, indent=2))

    summary = {
        "model_id": cfg.model_id,
        "layer_indices": cfg.layer_indices,
        "target_tokens": target_token_strs,
        "n_samples_requested": cfg.n_samples,
        "n_samples_processed": len(per_sample_records),
        "position_selection": cfg.position_selection,
        "compute_per_token_jacobian": cfg.compute_per_token_jacobian,
        "per_layer": layer_summary,
    }
    (out_dir / "jlens_summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("Done. Outputs written to %s", out_dir)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse(JacobianLensConfig, argument_generation_mode=ArgumentGenerationMode.FLAT)
    _run(cfg)


if __name__ == "__main__":
    main()
