

import asyncio
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DESCRIPTION_PROMPT = """\
You are a mechanistic interpretability researcher auditing a Sparse Autoencoder (SAE) \
trained on a vision-language model (Gemma 3) processing internet memes.

Below are the top activating examples for a single SAE feature (feature index: {feat_idx}).
Each example shows the meme text and its ground-truth hate/harm label.

--- TOP EXAMPLES ---
{examples_block}
--- END ---

Label distribution: {label_dist}
Datasets: {dataset_dist}

Task: Write a concise 1-2 sentence semantic label for this feature. Describe:
1. What visual or textual concept the feature is detecting.
2. Whether it encodes a harmful, neutral, or structural/formatting pattern.

Be specific. Avoid generic phrases like "the feature activates on hateful content". \
Focus on the common visual/textual motif across examples.

Respond with ONLY the description, no preamble.
"""


@dataclass
class FeatureDescriptionConfig:
    exemplars_path: str = ""
    feature_cards_path: str = ""
    output_path: str = ""
    # LLM settings
    api_base: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "gemma-3-27b-it"
    max_exemplars: int = 8
    max_tokens: int = 150
    concurrency: int = 4
    # If True, generate template-based descriptions without API calls.
    dry_run: bool = False


def _load_exemplars(path: Path) -> dict[str, list[dict[str, Any]]]:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        if isinstance(raw.get("features"), dict):
            return raw["features"]
        if isinstance(raw.get("features"), list):
            return {str(f.get("feature_idx")): f.get("top_examples", []) for f in raw["features"]}
    return raw or {}


def _build_examples_block(examples: list[dict], max_k: int) -> str:
    lines = []
    for i, ex in enumerate(examples[:max_k], 1):
        text = (ex.get("meme_text") or "").strip().replace("\n", " ")
        label = ex.get("gold_label", "?")
        dataset = ex.get("dataset_name", "?")
        act = ex.get("activation") or ex.get("activation_value")
        act_str = f"  [act={act:.3f}]" if act is not None else ""
        lines.append(f"{i}. [{dataset}/{label}{act_str}] {text[:200]}")
    return "\n".join(lines)


def _template_description(feat_idx: int, examples: list[dict]) -> str:
    """Fallback description when LLM is not available (dry_run=True)."""
    if not examples:
        return f"Feature {feat_idx}: no exemplars available."
    label_counts = Counter(ex.get("gold_label", "?") for ex in examples)
    ds_counts = Counter(ex.get("dataset_name", "?") for ex in examples)
    top_label, top_label_n = label_counts.most_common(1)[0]
    top_ds, top_ds_n = ds_counts.most_common(1)[0]
    frac = top_label_n / len(examples)
    # Extract common words from meme texts
    all_text = " ".join((ex.get("meme_text") or "") for ex in examples).lower()
    words = re.findall(r"\b[a-z]{4,}\b", all_text)
    word_freq = Counter(words)
    stopwords = {"this", "that", "with", "your", "they", "their", "have", "will",
                 "from", "what", "when", "just", "been", "there", "which", "would"}
    common = [w for w, _ in word_freq.most_common(10) if w not in stopwords][:3]
    common_str = ", ".join(f'"{w}"' for w in common) if common else "(no clear pattern)"
    return (
        f"Activates predominantly on {top_ds} ({top_ds_n}/{len(examples)}) with "
        f"{frac:.0%} '{top_label}' labels. "
        f"Common text tokens: {common_str}. "
        f"[auto-generated template — run with LLM for semantic description]"
    )


async def _describe_feature(
    feat_idx: int,
    examples: list[dict],
    cfg: FeatureDescriptionConfig,
    client,
    semaphore: asyncio.Semaphore,
) -> str:
    if not examples:
        return f"Feature {feat_idx}: no exemplars available."

    if cfg.dry_run or client is None:
        return _template_description(feat_idx, examples)

    label_dist = ", ".join(
        f"{l}:{c}" for l, c in Counter(ex.get("gold_label") for ex in examples).most_common(5)
    )
    dataset_dist = ", ".join(
        f"{d}:{c}" for d, c in Counter(ex.get("dataset_name") for ex in examples).most_common(4)
    )
    prompt = DESCRIPTION_PROMPT.format(
        feat_idx=feat_idx,
        examples_block=_build_examples_block(examples, cfg.max_exemplars),
        label_dist=label_dist,
        dataset_dist=dataset_dist,
    )

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=cfg.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=cfg.max_tokens,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("LLM call failed for feature %d: %s — using template", feat_idx, exc)
            return _template_description(feat_idx, examples)


async def _run_async(cfg: FeatureDescriptionConfig) -> None:
    exemplars = _load_exemplars(Path(cfg.exemplars_path))

    feature_ids: list[int]
    if cfg.feature_cards_path and Path(cfg.feature_cards_path).exists():
        cards = json.loads(Path(cfg.feature_cards_path).read_text())
        feature_ids = [f["feature_idx"] for f in cards.get("features", [])]
    else:
        feature_ids = [int(k) for k in exemplars]

    client = None
    if not cfg.dry_run:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(base_url=cfg.api_base, api_key=cfg.api_key)
        except ImportError:
            logger.warning("openai package not available — falling back to template descriptions")

    semaphore = asyncio.Semaphore(cfg.concurrency)
    tasks = [
        _describe_feature(feat_idx, exemplars.get(str(feat_idx), []), cfg, client, semaphore)
        for feat_idx in feature_ids
    ]

    logger.info("Generating descriptions for %d features (dry_run=%s)", len(feature_ids), cfg.dry_run)
    descriptions_list = await asyncio.gather(*tasks)

    new_descriptions = {str(fid): desc for fid, desc in zip(feature_ids, descriptions_list)}
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Merge with existing file so incremental runs don't lose previous descriptions
    existing: dict = {}
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
            logger.info("Merging with %d existing descriptions from %s", len(existing), output_path)
        except Exception:
            pass
    # Never overwrite a real description with the fallback "no exemplars available" text
    _FALLBACK = "no exemplars available"
    merged = {k: v for k, v in existing.items()}
    for k, v in new_descriptions.items():
        if _FALLBACK in v and k in merged and _FALLBACK not in merged[k]:
            logger.warning("Skipping fallback description for f%s — keeping existing real description", k)
            continue
        merged[k] = v
    output_path.write_text(json.dumps(merged, indent=2))
    logger.info("Wrote %d descriptions to %s (%d new/updated)", len(merged), output_path, len(new_descriptions))


def run(cfg: FeatureDescriptionConfig) -> None:
    asyncio.run(_run_async(cfg))


def main() -> None:
    from simple_parsing import ArgumentGenerationMode, parse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(FeatureDescriptionConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
