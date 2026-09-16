

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from simple_parsing import parse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass
class ExistTranslationConfig:
    input_path: str = "./data/EXIST_2025_Memes_Dataset/training/EXIST2025_training.json"
    output_path: str = "./exist_es_to_en_translations.json"
    language: str = "es"
    model_id: str = "Helsinki-NLP/opus-mt-es-en"
    batch_size: int = 16
    max_input_tokens: int = 512
    max_new_tokens: int = 256
    device: str = "cuda:0"
    overwrite: bool = False


def run(cfg: ExistTranslationConfig) -> None:
    output_fpath = Path(cfg.output_path)
    assert cfg.batch_size > 0, "batch_size must be positive."
    assert cfg.max_input_tokens > 0, "max_input_tokens must be positive."
    assert cfg.max_new_tokens > 0, "max_new_tokens must be positive."
    if output_fpath.exists() and not cfg.overwrite:
        raise FileExistsError(f"Translation output already exists: {output_fpath}. Pass --overwrite true to replace it.")

    records = json.loads(Path(cfg.input_path).read_text())
    assert isinstance(records, dict), "Expected the EXIST training JSON object."
    rows = [
        (str(row["id_EXIST"]), str(row.get("text", "")).strip())
        for row in records.values()
        if isinstance(row, dict) and str(row.get("lang", "")).lower() == cfg.language.lower() and str(row.get("text", "")).strip()
    ]
    assert rows, f"No non-empty {cfg.language} OCR texts found in {cfg.input_path}."

    device = torch.device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_id).to(device).eval()
    translations: dict[str, str] = {}
    with torch.inference_mode():
        for start in range(0, len(rows), cfg.batch_size):
            batch = rows[start : start + cfg.batch_size]
            inputs = tokenizer(
                [text for _sample_id, text in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_input_tokens,
            ).to(device)
            generated = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens, do_sample=False)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for (sample_id, _text), translation in zip(batch, decoded, strict=True):
                translation = translation.strip()
                assert translation, f"Empty translation for EXIST sample {sample_id}."
                translations[sample_id] = translation
            logging.info("Translated %d/%d EXIST %s OCR texts", len(translations), len(rows), cfg.language)

    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    output_fpath.write_text(json.dumps(translations, ensure_ascii=False, indent=2) + "\n")
    logging.info("Wrote %d translations to %s", len(translations), output_fpath)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(parse(ExistTranslationConfig))


if __name__ == "__main__":
    main()