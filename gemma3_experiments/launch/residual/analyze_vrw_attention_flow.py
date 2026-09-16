

import json
import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from simple_parsing import ArgumentGenerationMode, parse
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


from gemma3_experiments.launch.residual.probe_validate import (
    _load_fhm_confounder_metadata,
    _normalise_fhm_id,
    _normalise_group_value,
)
from gemma3_experiments.residual_sae import (
    GemmaResidualConfig,
    MemeDatasetConfig,
    build_multimodal_inputs,
    load_meme_samples,
    resolve_visible_device,
)
from gemma3_experiments.residual_sae.gemma import _image_token_span

logger = logging.getLogger(__name__)

FLOW_NAMES = ("Y_to_Q", "Y_to_V", "Y_to_O", "Y_to_Y", "V_to_V", "O_to_V")
STAGE_NAMES = ("early", "mid", "late")


@dataclass
class VRWAttentionFlowConfig:
    dataset: MemeDatasetConfig = field(
        default_factory=lambda: MemeDatasetConfig(datasets=["Hateful_Memes_Dataset"], splits=["validation"])
    )
    gemma: GemmaResidualConfig = field(default_factory=GemmaResidualConfig)
    output_dir: str = "./vrw_attention_flow"
    max_samples: int | None = None

    # REQUIRED: joint_infer.py-style predictions JSONL. Supplies the recorded
    # prompt_text (exact tokens the model saw) and prediction_text (recorded
    # answer -> becomes the Y group), so no regeneration is needed.
    base_predictions_path: str = ""

    # Optional: probe_val_predictions.jsonl -- enables base_correct_probe_correct /
    # base_wrong_probe_correct / base_probe_both_wrong split tags.
    probe_predictions_path: str = ""
    task_key: str = ""

    # Optional: jlens_quadrant_per_sample.jsonl from analyze_jlens_quadrants.py --
    # enables probe_wrong_jlens_correct / probe_jlens_both_wrong split tags.
    jlens_quadrant_path: str = ""

    # Optional: ivl_plus_gemini_captions_complete.json -- enables
    # pseudo_img_pair / pseudo_text_pair split tags (FHM only).
    fhm_confounder_metadata_path: str = ""

    device: str = "cuda:0"


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as fd:
        for line in fd:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_base_index(path: str) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id")): row for row in _read_jsonl(path)}


def _load_probe_index(path: str, task_key: str) -> dict[str, dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if task_key and row.get("task_key") != task_key:
            continue
        sid = str(row["sample_key"]).split("::")[-1]
        idx[sid] = row
    return idx


def _load_jlens_index(path: str) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id")): row for row in _read_jsonl(path)}


def _load_model(cfg: GemmaResidualConfig, device: str) -> tuple[Any, Any]:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        device_map={"": device},
        attn_implementation="eager",  # output_attentions requires eager attention
    ).eval()
    return model, processor


def _locate_substring_span(
    tokenizer: Any, ids: list[int], substring: str, start: int, end: int,
) -> tuple[int, int] | None:
    """Find the token span in ids[start:end] whose decoded text contains `substring`.

    Decodes tokens one at a time and slides a running-string match, which is
    robust to sub-word tokenization boundaries that don't line up with the
    chat template's character offsets.
    """
    target = (substring or "").strip().lower()
    if not target or start >= end:
        return None
    cum_text = ""
    boundaries: list[int] = []
    for idx in range(start, end):
        cum_text += tokenizer.decode([ids[idx]], skip_special_tokens=True)
        boundaries.append(len(cum_text))
    pos = cum_text.lower().find(target)
    if pos < 0:
        return None
    end_char = pos + len(target)
    start_tok = next((start + i for i, b in enumerate(boundaries) if b > pos), start)
    end_tok = next((start + i + 1 for i, b in enumerate(boundaries) if b >= end_char), end)
    return start_tok, min(end_tok, end)


def _token_groups(
    processor: Any, full_input_ids: torch.Tensor, image_span: tuple[int, int], input_len: int, meme_text: str,
) -> dict[str, torch.Tensor]:
    seq_len = int(full_input_ids.shape[0])
    all_idx = torch.arange(seq_len)
    v_idx = all_idx[image_span[0]:image_span[1]]
    y_idx = all_idx[input_len:] if input_len < seq_len else all_idx[:0]

    prompt_mask = torch.ones(seq_len, dtype=torch.bool)
    prompt_mask[image_span[0]:image_span[1]] = False
    prompt_mask[input_len:] = False

    o_span = _locate_substring_span(processor.tokenizer, full_input_ids.tolist(), meme_text, 0, input_len)
    if o_span is not None:
        o_idx = all_idx[o_span[0]:o_span[1]]
        prompt_mask[o_span[0]:o_span[1]] = False
    else:
        o_idx = all_idx[:0]

    q_idx = all_idx[prompt_mask]
    return {"V": v_idx, "O": o_idx, "Q": q_idx, "Y": y_idx}


def _flow_mass(layer_attn: torch.Tensor, a_idx: torch.Tensor, b_idx: torch.Tensor) -> float | None:
    if a_idx.numel() == 0 or b_idx.numel() == 0:
        return None
    head_mean = layer_attn[0].float().mean(dim=0)  # (seq, seq), averaged over heads
    sub = head_mean.index_select(0, a_idx.to(head_mean.device)).index_select(1, b_idx.to(head_mean.device))
    return float(sub.sum(dim=1).mean().item())


def _compute_sample_flows(
    model: Any, processor: Any, image_path: str, prompt_text: str, assistant_text: str, meme_text: str, device: str,
) -> dict[str, Any] | None:
    image = Image.open(image_path).convert("RGB")
    prompt_only_inputs = build_multimodal_inputs(processor, image, prompt_text, device)
    input_len = int(prompt_only_inputs["input_ids"].shape[1])

    full_inputs = build_multimodal_inputs(
        processor, image, prompt_text, device, assistant_text=assistant_text, add_generation_prompt=False,
    )
    full_input_ids = full_inputs["input_ids"][0].detach().cpu()
    seq_len = int(full_input_ids.shape[0])
    if input_len >= seq_len:
        return None  # assistant_text produced no extra tokens

    image_span = _image_token_span(full_input_ids, processor, seq_len)
    groups = _token_groups(processor, full_input_ids, image_span, input_len, meme_text)

    with torch.no_grad():
        outputs = model(**full_inputs, use_cache=False, output_attentions=True)
    attentions = outputs.attentions
    assert attentions is not None, (
        "Model returned no attentions -- ensure the model was loaded with attn_implementation='eager'."
    )
    n_layers = len(attentions)

    per_layer: dict[int, dict[str, float | None]] = {}
    for layer_idx, layer_attn in enumerate(attentions):
        per_layer[layer_idx] = {
            "Y_to_Q": _flow_mass(layer_attn, groups["Y"], groups["Q"]),
            "Y_to_V": _flow_mass(layer_attn, groups["Y"], groups["V"]),
            "Y_to_O": _flow_mass(layer_attn, groups["Y"], groups["O"]),
            "Y_to_Y": _flow_mass(layer_attn, groups["Y"], groups["Y"]),
            "V_to_V": _flow_mass(layer_attn, groups["V"], groups["V"]),
            "O_to_V": _flow_mass(layer_attn, groups["O"], groups["V"]),
        }
    del outputs, attentions

    return {
        "n_layers": n_layers,
        "group_sizes": {k: int(v.numel()) for k, v in groups.items()},
        "per_layer": per_layer,
    }


def _stage_buckets(n_layers: int) -> dict[str, range]:
    third = max(1, n_layers // 3)
    return {
        "early": range(0, third),
        "mid": range(third, 2 * third),
        "late": range(2 * third, n_layers),
    }


def _stage_averages(per_layer: dict[int, dict[str, float | None]], n_layers: int) -> dict[str, dict[str, float | None]]:
    buckets = _stage_buckets(n_layers)
    out: dict[str, dict[str, float | None]] = {}
    for stage_name, layer_range in buckets.items():
        stage_vals: dict[str, list[float]] = defaultdict(list)
        for layer_idx in layer_range:
            for flow_name, value in per_layer.get(layer_idx, {}).items():
                if value is not None:
                    stage_vals[flow_name].append(value)
        out[stage_name] = {
            flow_name: (sum(vals) / len(vals) if vals else None) for flow_name, vals in stage_vals.items()
        }
    return out


def _build_pair_membership(
    samples: list[Any], confounder_meta: dict[str, dict[str, Any]], group_key: str,
) -> set[str]:
    pass
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for sample in samples:
        if sample.dataset_name != "Hateful_Memes_Dataset":
            continue
        gold = sample.gold_fields.get(sample.schema.primary_field.name)
        if gold not in ("hateful", "not hateful"):
            continue
        norm_sid = _normalise_fhm_id(sample.sample_id)
        meta = confounder_meta.get(norm_sid)
        if meta is None:
            continue
        group_value = _normalise_group_value(meta.get(group_key))
        if group_value is None:
            continue
        groups[group_value].append((norm_sid, gold))

    paired_ids: set[str] = set()
    for members in groups.values():
        labels = {gold for _, gold in members}
        if "hateful" in labels and "not hateful" in labels:
            paired_ids.update(sid for sid, _ in members)
    return paired_ids


def _classify_tags(
    norm_sid: str,
    base_row: dict[str, Any] | None,
    sample_gold: str | None,
    probe_row: dict[str, Any] | None,
    jlens_row: dict[str, Any] | None,
    pseudo_img_paired_ids: set[str] | None,
    pseudo_text_paired_ids: set[str] | None,
) -> dict[str, bool]:
    tags: dict[str, bool] = {}
    base_correct = None
    if base_row is not None and sample_gold is not None:
        base_correct = base_row.get("prediction_label") == sample_gold
    probe_correct = probe_row.get("probe_correct") if probe_row is not None else None
    if base_correct is None and probe_row is not None:
        base_correct = probe_row.get("baseline_correct")

    if base_correct is not None and probe_correct is not None:
        tags["base_correct_probe_correct"] = bool(base_correct and probe_correct)
        tags["base_wrong_probe_correct"] = bool((not base_correct) and probe_correct)
        tags["base_probe_both_wrong"] = bool((not base_correct) and (not probe_correct))

    if jlens_row is not None:
        tags["probe_wrong_jlens_correct"] = jlens_row.get("quadrant") == "jlens_only"
        tags["probe_jlens_both_wrong"] = jlens_row.get("quadrant") == "both_miss"

    if pseudo_img_paired_ids is not None:
        tags["pseudo_img_pair"] = norm_sid in pseudo_img_paired_ids
    if pseudo_text_paired_ids is not None:
        tags["pseudo_text_pair"] = norm_sid in pseudo_text_paired_ids

    return tags


def _stage_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for stage in STAGE_NAMES:
        stage_out: dict[str, float | None] = {}
        for flow in FLOW_NAMES:
            vals = [r["stage"].get(stage, {}).get(flow) for r in rows]
            vals = [v for v in vals if v is not None]
            stage_out[flow] = round(sum(vals) / len(vals), 4) if vals else None
        out[stage] = stage_out
    return out


def _tag_group_stats(rows: list[dict[str, Any]], tag: str, want: bool) -> dict[str, Any]:
    subset = [r for r in rows if r["tags"].get(tag) == want]
    return _stage_stats(subset)


def _hyp_delta(
    rows: list[dict[str, Any]], tag_a: str, tag_b: str, stage: str, flows: tuple[str, ...],
) -> dict[str, Any]:
    def _combo(subset: list[dict[str, Any]]) -> tuple[float | None, int]:
        vals = []
        for r in subset:
            parts = [r["stage"].get(stage, {}).get(f) for f in flows]
            parts = [p for p in parts if p is not None]
            if parts:
                vals.append(sum(parts))
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    a_rows = [r for r in rows if r["tags"].get(tag_a) is True]
    b_rows = [r for r in rows if r["tags"].get(tag_b) is True]
    mean_a, n_a = _combo(a_rows)
    mean_b, n_b = _combo(b_rows)
    delta = (mean_a - mean_b) if (mean_a is not None and mean_b is not None) else None
    return {
        "group_a": tag_a, "n_a": n_a, "mean_a": round(mean_a, 4) if mean_a is not None else None,
        "group_b": tag_b, "n_b": n_b, "mean_b": round(mean_b, 4) if mean_b is not None else None,
        "stage": stage, "flows": list(flows),
        "delta_a_minus_b": round(delta, 4) if delta is not None else None,
    }


def _hyp_correlation(rows: list[dict[str, Any]], tag: str, flows: tuple[str, ...], stage: str) -> dict[str, Any]:
    subset = [r for r in rows if tag in r["tags"]]
    xs, ys = [], []
    for r in subset:
        parts = [r["stage"].get(stage, {}).get(f) for f in flows]
        parts = [p for p in parts if p is not None]
        if not parts:
            continue
        xs.append(sum(parts))
        ys.append(1.0 if r["tags"].get(tag) else 0.0)
    if len(xs) < 3 or len(set(ys)) < 2:
        return {"n": len(xs), "pearson_r": None}
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    r = cov / ((var_x * var_y) ** 0.5) if var_x > 0 and var_y > 0 else None
    return {"n": len(xs), "pearson_r": round(r, 4) if r is not None else None}


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_tags = sorted({tag for r in rows for tag in r["tags"].keys()})
    summary: dict[str, Any] = {
        "n_total": len(rows),
        "overall": _stage_stats(rows),
        "by_tag": {
            tag: {"true": _tag_group_stats(rows, tag, True), "false": _tag_group_stats(rows, tag, False)}
            for tag in all_tags
        },
    }

    hyps: dict[str, Any] = {}
    hyps["h1_base_wrong_probe_correct_vs_base_correct_probe_correct__late_YV_plus_YO"] = _hyp_delta(
        rows, "base_wrong_probe_correct", "base_correct_probe_correct", "late", ("Y_to_V", "Y_to_O"),
    )
    hyps["h2_pseudo_img_vs_pseudo_text_relay_signature"] = {
        "pseudo_img_pair": _tag_group_stats(rows, "pseudo_img_pair", True),
        "pseudo_text_pair": _tag_group_stats(rows, "pseudo_text_pair", True),
    }
    hyps["h4_jlens_correct_correlation__late_YV_plus_YO"] = _hyp_correlation(
        rows, tag="probe_wrong_jlens_correct", flows=("Y_to_V", "Y_to_O"), stage="late",
    )
    summary["hypotheses"] = hyps
    return summary


def _print_report(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print(f"VRW Attention-Flow Audit -- n_total={summary['n_total']}")
    print("-" * 88)
    print(f"{'stage':6s} " + " ".join(f"{f:>10s}" for f in FLOW_NAMES))
    for stage in STAGE_NAMES:
        vals = summary["overall"].get(stage, {})
        print(f"{stage:6s} " + " ".join(
            f"{vals.get(f):>10.4f}" if vals.get(f) is not None else f"{'n/a':>10s}" for f in FLOW_NAMES
        ))
    print("-" * 88)
    for tag, groups in summary["by_tag"].items():
        n_true = groups["true"]["n"]
        n_false = groups["false"]["n"]
        print(f"tag={tag}  n_true={n_true}  n_false={n_false}")
    print("-" * 88)
    print("Hypothesis checks:")
    print(json.dumps(summary["hypotheses"], indent=2))
    print("=" * 88)


def run(cfg: VRWAttentionFlowConfig) -> None:
    assert cfg.base_predictions_path, "base_predictions_path is required (recorded prompt_text/prediction_text)"
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_visible_device(cfg.device)
    samples = load_meme_samples(cfg.dataset)
    if cfg.max_samples is not None:
        samples = samples[: cfg.max_samples]
    logger.info("Loaded %d samples", len(samples))

    base_index = _load_base_index(cfg.base_predictions_path)
    probe_index = _load_probe_index(cfg.probe_predictions_path, cfg.task_key) if cfg.probe_predictions_path else {}
    jlens_index = _load_jlens_index(cfg.jlens_quadrant_path) if cfg.jlens_quadrant_path else {}
    confounder_meta = (
        _load_fhm_confounder_metadata(Path(cfg.fhm_confounder_metadata_path))
        if cfg.fhm_confounder_metadata_path else {}
    )
    pseudo_img_paired_ids = _build_pair_membership(samples, confounder_meta, "pseudo_img_idx") if confounder_meta else None
    pseudo_text_paired_ids = _build_pair_membership(samples, confounder_meta, "pseudo_text_idx") if confounder_meta else None
    if confounder_meta:
        logger.info(
            "Genuine pseudo-pair membership: pseudo_img=%d/%d, pseudo_text=%d/%d samples",
            len(pseudo_img_paired_ids), len(samples), len(pseudo_text_paired_ids), len(samples),
        )

    model, processor = _load_model(cfg.gemma, device)

    per_sample_rows: list[dict[str, Any]] = []
    n_skipped = 0
    log_every = max(1, len(samples) // 20)
    for i, sample in enumerate(samples, start=1):
        sid = sample.sample_id
        base_row = base_index.get(sid)
        if base_row is None:
            n_skipped += 1
            continue
        prompt_text = base_row.get("prompt_text")
        assistant_text = base_row.get("prediction_text") or base_row.get("prediction_label")
        if not prompt_text or not assistant_text:
            n_skipped += 1
            continue

        result = _compute_sample_flows(
            model, processor, sample.image_path, prompt_text, assistant_text, sample.meme_text, device,
        )
        if result is None:
            n_skipped += 1
            continue

        gold = sample.gold_fields.get(sample.schema.primary_field.name)
        norm_sid = _normalise_fhm_id(sid)
        tags = _classify_tags(
            norm_sid, base_row, gold, probe_index.get(sid), jlens_index.get(sid),
            pseudo_img_paired_ids, pseudo_text_paired_ids,
        )
        stage_avg = _stage_averages(result["per_layer"], result["n_layers"])
        per_sample_rows.append({
            "sample_id": sid,
            "dataset_name": sample.dataset_name,
            "task_name": sample.task_name,
            "gold": gold,
            "prediction_label": base_row.get("prediction_label"),
            "group_sizes": result["group_sizes"],
            "n_layers": result["n_layers"],
            "per_layer": result["per_layer"],
            "stage": stage_avg,
            "tags": tags,
        })

        if i % log_every == 0:
            logger.info("Processed %d/%d samples (%d skipped)", i, len(samples), n_skipped)

    logger.info("Done: %d flows computed, %d skipped", len(per_sample_rows), n_skipped)
    assert per_sample_rows, "No samples produced attention-flow rows -- check base_predictions_path matches sample ids"

    with (out_dir / "vrw_attention_flow_per_sample.jsonl").open("w") as fd:
        for row in per_sample_rows:
            fd.write(json.dumps(row) + "\n")

    summary = _build_summary(per_sample_rows)
    (out_dir / "vrw_attention_flow_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote vrw_attention_flow_per_sample.jsonl + vrw_attention_flow_summary.json to %s", out_dir)
    _print_report(summary)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(VRWAttentionFlowConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
