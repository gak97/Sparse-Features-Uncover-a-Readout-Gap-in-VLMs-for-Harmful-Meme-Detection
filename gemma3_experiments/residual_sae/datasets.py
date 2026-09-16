import csv
import hashlib
import json
import random
import warnings
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from PIL import Image, UnidentifiedImageError

from .config import MemeDatasetConfig


@dataclass(frozen=True)
class TaskField:
    name: str
    labels: tuple[str, ...]
    multi_label: bool = False


@dataclass(frozen=True)
class TaskSchema:
    task_name: str
    task_description: str
    primary_field: TaskField
    secondary_fields: tuple[TaskField, ...] = ()


@dataclass
class MemeSample:
    dataset_name: str
    task_name: str
    sample_id: str
    split: str
    image_path: str
    meme_text: str
    schema: TaskSchema
    gold_fields: dict[str, Any] = field(default_factory=dict)
    raw_fields: dict[str, Any] = field(default_factory=dict)

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "task_name": self.task_name,
            "sample_id": self.sample_id,
            "split": self.split,
            "image_path": self.image_path,
            "meme_text": self.meme_text,
            "gold_fields": self.gold_fields,
            "raw_fields": self.raw_fields,
            "schema": asdict(self.schema),
        }


CRISIS_A_SCHEMA = TaskSchema(
    task_name="subtask_a_hate_speech",
    task_description="Decide whether the crisis meme contains hate speech.",
    primary_field=TaskField("label", ("Hate Speech", "No Hate Speech")),
)
CRISIS_B_SCHEMA = TaskSchema(
    task_name="subtask_b_hate_target",
    task_description="Identify which group the hateful crisis meme targets.",
    primary_field=TaskField("target", ("Community", "Individual", "Organization")),
)
HARMEME_SCHEMA = TaskSchema(
    task_name="harmfulness_and_target",
    task_description="Classify the meme's harmfulness level and who it targets.",
    primary_field=TaskField("harmfulness", ("not harmful", "somewhat harmful", "very harmful")),
    secondary_fields=(TaskField("target", ("individual", "community", "society", "organization")),),
)
HATEFUL_MEMES_SCHEMA = TaskSchema(
    task_name="hateful_memes_binary",
    task_description="Decide whether the meme is hateful or not hateful.",
    primary_field=TaskField("label", ("not hateful", "hateful")),
)
MAMI_SCHEMA = TaskSchema(
    task_name="misogyny_and_subtypes",
    task_description="Decide whether the meme is misogynous and, if so, which misogyny subtypes apply.",
    primary_field=TaskField("label", ("non-misogynous", "misogynous")),
    secondary_fields=(
        TaskField("subtypes", ("shaming", "stereotype", "objectification", "violence"), multi_label=True),
    ),
)
MMHS_SCHEMA = TaskSchema(
    task_name="hate_category",
    task_description="Predict the primary hate category expressed by the meme.",
    primary_field=TaskField(
        "label",
        ("NotHate", "Racist", "Sexist", "Homophobe", "Religion", "OtherHate"),
    ),
)
MULTIOFF_SCHEMA = TaskSchema(
    task_name="offensive_detection",
    task_description="Decide whether the meme is offensive or non-offensive.",
    primary_field=TaskField("label", ("Non-offensiv", "offensive")),
)
MULTIBULLY_SCHEMA = TaskSchema(
    task_name="cyberbullying_detection",
    task_description="Decide whether the code-mixed Hindi meme contains cyberbullying.",
    primary_field=TaskField("label", ("Nonbully", "Bully")),
)
EXIST_SEXISM_SCHEMA = TaskSchema(
    task_name="sexism_detection",
    task_description="Decide whether the meme expresses sexism.",
    primary_field=TaskField("label", ("non-sexist", "sexist")),
)

_MISSING_IMAGE_COUNTS: Counter[str] = Counter()
_MISSING_IMAGE_EXAMPLES: dict[str, str] = {}
_UNREADABLE_IMAGE_COUNTS: Counter[str] = Counter()
_UNREADABLE_IMAGE_EXAMPLES: dict[str, str] = {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as fd:
        for line in fd:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="") as fd:
        return list(csv.DictReader(fd, delimiter=delimiter))


def _majority_label(labels: list[str]) -> str:
    assert labels, "Expected at least one label."
    counts = Counter(labels)
    label, _ = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return label


def _assign_synthetic_split(dataset_name: str, sample_id: str, seed: int) -> str:
    key = f"{dataset_name}:{sample_id}:{seed}".encode("utf-8")
    bucket = int(hashlib.sha256(key).hexdigest()[:8], 16) % 100
    if bucket < 60:
        return "train"
    if bucket < 80:
        return "validation"
    return "test"


def _finalize_splits(samples: list[MemeSample], requested_splits: set[str], seed: int) -> list[MemeSample]:
    canonical_splits = {"train", "validation", "test"}
    finalized: list[MemeSample] = []
    for sample in samples:
        split = sample.split
        if split not in canonical_splits:
            split = _assign_synthetic_split(sample.dataset_name, sample.sample_id, seed)
        if split == "validation" and "val" in requested_splits:
            finalized.append(sample)
            continue
        if split in requested_splits:
            finalized.append(sample)
            continue
        if split == "validation" and "validation" in requested_splits:
            finalized.append(sample)
    return finalized


def _make_sample(
    *,
    dataset_name: str,
    task_name: str,
    sample_id: str,
    split: str,
    image_path: Path,
    meme_text: str,
    schema: TaskSchema,
    gold_fields: dict[str, Any],
    raw_fields: dict[str, Any],
) -> MemeSample | None:
    if not image_path.exists():
        _MISSING_IMAGE_COUNTS[dataset_name] += 1
        _MISSING_IMAGE_EXAMPLES.setdefault(dataset_name, str(image_path))
        return None
    return MemeSample(
        dataset_name=dataset_name,
        task_name=task_name,
        sample_id=sample_id,
        split=split,
        image_path=str(image_path),
        meme_text=(meme_text or "").strip(),
        schema=schema,
        gold_fields=gold_fields,
        raw_fields=raw_fields,
    )


def _load_crisis_hatemm(root: Path, include_unlabeled: bool, splits: set[str]) -> list[MemeSample]:
    samples: list[MemeSample] = []
    dataset_name = "Crisis_HateMM"
    subtask_specs = {
        "subTaskA": CRISIS_A_SCHEMA,
        "subTaskB": CRISIS_B_SCHEMA,
    }
    # Labels for validation/test live in a CSV: Labels .../eval_subtaskA.csv
    # CSV format: index,Label  where index = image file stem, Label = integer.
    # Explicit int→string mappings derived from the dataset's README conventions:
    #   subTaskA: 0=No Hate Speech, 1=Hate Speech (0=negative, 1=positive)
    #   subTaskB: 0=Community, 1=Individual, 2=Organization (alphabetical)
    _CRISIS_INT_MAP: dict[str, dict[int, str]] = {
        "subTaskA": {0: "No Hate Speech", 1: "Hate Speech"},
        "subTaskB": {0: "Community", 1: "Individual", 2: "Organization"},
    }
    _labels_dpath = root / "Labels (Do not use dev data for training in evaluation phase)"
    _eval_csv_by_subtask: dict[str, dict[str, str]] = {}
    for subtask, schema in subtask_specs.items():
        csv_path = _labels_dpath / f"eval_subtask{subtask[-1]}.csv"  # eval_subtaskA.csv / eval_subtaskB.csv
        if not csv_path.exists():
            continue
        int_to_label = _CRISIS_INT_MAP[subtask]
        label_map: dict[str, str] = {}
        with csv_path.open() as fd:
            for i, line in enumerate(fd):
                if i == 0:  # skip header
                    continue
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                idx, lbl_int = parts[0].strip(), parts[1].strip()
                try:
                    label_map[idx] = int_to_label[int(lbl_int)]
                except (ValueError, IndexError):
                    pass
        _eval_csv_by_subtask[subtask] = label_map

    split_alias = {"validation": "validation", "val": "validation", "test": "test", "train": "train"}
    for subtask, schema in subtask_specs.items():
        for split_name in splits:
            if split_name not in split_alias:
                continue
            actual_split = split_alias[split_name]
            split_dpath = root / subtask / actual_split
            if not split_dpath.exists():
                continue
            if actual_split == "train":
                for label_dpath in sorted(p for p in split_dpath.iterdir() if p.is_dir()):
                    for image_fpath in sorted(label_dpath.iterdir()):
                        if not image_fpath.is_file():
                            continue
                        gold_key = schema.primary_field.name
                        sample = _make_sample(
                            dataset_name=dataset_name,
                            task_name=schema.task_name,
                            sample_id=f"{subtask}:{image_fpath.stem}",
                            split=actual_split,
                            image_path=image_fpath,
                            meme_text="",
                            schema=schema,
                            gold_fields={gold_key: label_dpath.name},
                            raw_fields={gold_key: label_dpath.name},
                        )
                        if sample is not None:
                            samples.append(sample)
                continue
            # Non-train split: try labeled CSV first, then fall back to unlabeled.
            label_map = _eval_csv_by_subtask.get(subtask, {})
            for image_fpath in sorted(split_dpath.iterdir()):
                if not image_fpath.is_file():
                    continue
                gold_label = label_map.get(image_fpath.stem)
                if gold_label is None and not include_unlabeled:
                    continue
                gold_key = schema.primary_field.name
                gold_fields = {gold_key: gold_label} if gold_label is not None else {}
                sample = _make_sample(
                    dataset_name=dataset_name,
                    task_name=schema.task_name,
                    sample_id=f"{subtask}:{image_fpath.stem}",
                    split=actual_split,
                    image_path=image_fpath,
                    meme_text="",
                    schema=schema,
                    gold_fields=gold_fields,
                    raw_fields=gold_fields,
                )
                if sample is not None:
                    samples.append(sample)
    return samples


def _load_harmeme(root: Path, splits: set[str]) -> list[MemeSample]:
    samples: list[MemeSample] = []
    split_to_fname = {"train": "train.jsonl", "validation": "val.jsonl", "val": "val.jsonl", "test": "test.jsonl"}
    for split_name in splits:
        fname = split_to_fname.get(split_name)
        if fname is None:
            continue
        for row in _read_jsonl(root / fname):
            labels = list(row.get("labels", []))
            gold = {}
            if labels:
                gold["harmfulness"] = labels[0]
            if len(labels) > 1:
                gold["target"] = labels[1]
            sample = _make_sample(
                dataset_name="HarMeme",
                task_name=HARMEME_SCHEMA.task_name,
                sample_id=str(row["id"]),
                split="validation" if split_name == "val" else split_name,
                image_path=root / "img" / row["image"],
                meme_text=row.get("text", ""),
                schema=HARMEME_SCHEMA,
                gold_fields=gold,
                raw_fields=row,
            )
            if sample is not None:
                samples.append(sample)
    return samples


def _load_fhm_label_metadata(path: str, label_split: str) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    rows = data.values() if isinstance(data, dict) else data
    by_id = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if label_split and row.get("split") != label_split:
            continue
        sample_id = row.get("id")
        if sample_id is None and row.get("img"):
            sample_id = Path(str(row["img"])).stem
        if sample_id is None:
            continue
        by_id[str(int(sample_id)) if isinstance(sample_id, float) and sample_id.is_integer() else str(sample_id)] = row
    return by_id


def _load_hateful_memes(root: Path, splits: set[str], include_unlabeled: bool, fhm_labels_path: str = "", fhm_label_split: str = "") -> list[MemeSample]:
    samples: list[MemeSample] = []
    split_to_fname = {"train": "train.jsonl", "validation": "dev.jsonl", "val": "dev.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    metadata_by_id = _load_fhm_label_metadata(fhm_labels_path, fhm_label_split)
    for split_name in splits:
        fname = split_to_fname.get(split_name)
        if fname is None:
            continue
        for row in _read_jsonl(root / fname):
            gold = {}
            if "label" in row and row["label"] is not None:
                gold["label"] = "hateful" if int(row["label"]) == 1 else "not hateful"
            elif str(row.get("id")) in metadata_by_id and metadata_by_id[str(row.get("id"))].get("label") is not None:
                gold["label"] = "hateful" if int(metadata_by_id[str(row.get("id"))]["label"]) == 1 else "not hateful"
            elif not include_unlabeled:
                continue
            sample = _make_sample(
                dataset_name="Hateful_Memes_Dataset",
                task_name=HATEFUL_MEMES_SCHEMA.task_name,
                sample_id=str(row["id"]),
                split="validation" if split_name in {"val", "dev"} else split_name,
                image_path=root / row["img"],
                meme_text=row.get("text", ""),
                schema=HATEFUL_MEMES_SCHEMA,
                gold_fields=gold,
                raw_fields=row,
            )
            if sample is not None:
                samples.append(sample)
    return samples


def _load_mami(root: Path, splits: set[str], include_unlabeled: bool) -> list[MemeSample]:
    samples: list[MemeSample] = []
    split_to_spec = {
        "train": ("train.tsv", "training_images"),
        "validation": ("validation.tsv", "training_images"),
        "val": ("validation.tsv", "training_images"),
        "test": ("test.tsv", "test_images"),
    }
    subtype_names = ("shaming", "stereotype", "objectification", "violence")
    for split_name in splits:
        spec = split_to_spec.get(split_name)
        if spec is None:
            continue
        fname, img_dir = spec
        for row in _read_csv(root / fname, delimiter="\t"):
            image_fname = (
                row.get("file_name")
                or row.get("mifile_name")
                or row.get("image_name")
                or row.get("img")
                or ""
            ).strip()
            if not image_fname:
                row_keys = sorted(row.keys())
                raise KeyError(f"Could not find MAMI image filename column in {fname}. Available keys: {row_keys}")

            gold: dict[str, Any] = {}
            if row.get("label") in {"0", "1"}:
                is_misogynous = row["label"] == "1"
                gold["label"] = "misogynous" if is_misogynous else "non-misogynous"
                if is_misogynous:
                    gold["subtypes"] = [name for name in subtype_names if row.get(name) == "1"]
            elif not include_unlabeled:
                continue
            sample = _make_sample(
                dataset_name="MAMI",
                task_name=MAMI_SCHEMA.task_name,
                sample_id=image_fname,
                split="validation" if split_name == "val" else split_name,
                image_path=root / img_dir / image_fname,
                meme_text=row.get("text", ""),
                schema=MAMI_SCHEMA,
                gold_fields=gold,
                raw_fields=row,
            )
            if sample is not None:
                samples.append(sample)
    return samples


def _load_mmhs150k(root: Path, splits: set[str]) -> list[MemeSample]:
    samples: list[MemeSample] = []
    with (root / "MMHS150K_GT.json").open() as fd:
        labels_by_id = json.load(fd)
    split_to_fname = {"train": "train_ids.txt", "validation": "val_ids.txt", "val": "val_ids.txt", "test": "test_ids.txt"}
    for split_name in splits:
        fname = split_to_fname.get(split_name)
        if fname is None:
            continue
        with (root / "splits" / fname).open() as fd:
            ids = [line.strip() for line in fd if line.strip()]
        for tweet_id in ids:
            row = labels_by_id.get(tweet_id)
            assert row is not None, f"Missing MMHS label row for id {tweet_id}"
            label_names = [str(label) for label in row.get("labels_str", [])]
            gold = {}
            if label_names:
                gold["label"] = _majority_label(label_names)
            sample = _make_sample(
                dataset_name="MMHS150K",
                task_name=MMHS_SCHEMA.task_name,
                sample_id=tweet_id,
                split="validation" if split_name == "val" else split_name,
                image_path=root / "img_resized" / f"{tweet_id}.jpg",
                meme_text=row.get("tweet_text", ""),
                schema=MMHS_SCHEMA,
                gold_fields=gold,
                raw_fields=row,
            )
            if sample is not None:
                samples.append(sample)
    return samples


def _load_multioff(root: Path, splits: set[str]) -> list[MemeSample]:
    samples: list[MemeSample] = []
    split_to_fname = {
        "train": "Training_meme_dataset.csv",
        "validation": "Validation_meme_dataset.csv",
        "val": "Validation_meme_dataset.csv",
        "test": "Testing_meme_dataset.csv",
    }
    for split_name in splits:
        fname = split_to_fname.get(split_name)
        if fname is None:
            continue
        for row in _read_csv(root / fname):
            sample = _make_sample(
                dataset_name="MultiOFF",
                task_name=MULTIOFF_SCHEMA.task_name,
                sample_id=row["image_name"],
                split="validation" if split_name == "val" else split_name,
                image_path=root / "Labelled Images" / row["image_name"],
                meme_text=row.get("sentence", ""),
                schema=MULTIOFF_SCHEMA,
                gold_fields={"label": row["label"]},
                raw_fields=row,
            )
            if sample is not None:
                samples.append(sample)
    return samples


_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _xlsx_column_index(cell_ref: str) -> int:
    letters = "".join(character for character in cell_ref if character.isalpha())
    value = 0
    for character in letters:
        value = value * 26 + ord(character.upper()) - ord("A") + 1
    return value - 1


def _xlsx_text(cell: ElementTree.Element, shared_strings: list[str]) -> str:
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(f".//{_XLSX_NS}t"))
    value = cell.findtext(f"{_XLSX_NS}v")
    if value is None:
        return ""
    if cell_type == "s":
        return shared_strings[int(value)]
    return value


def _read_xlsx_first_sheet(path: Path) -> list[dict[str, str]]:
    """Read the simple annotation table without requiring openpyxl in Slurm images."""
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(node.text or "" for node in item.findall(f".//{_XLSX_NS}t"))
                for item in shared_root.findall(f"{_XLSX_NS}si")
            ]
        sheet_root = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows: list[list[str]] = []
    for row in sheet_root.findall(f".//{_XLSX_NS}sheetData/{_XLSX_NS}row"):
        values: dict[int, str] = {}
        for cell in row.findall(f"{_XLSX_NS}c"):
            ref = cell.get("r")
            assert ref is not None, "XLSX cell is missing its reference."
            values[_xlsx_column_index(ref)] = _xlsx_text(cell, shared_strings)
        if values:
            rows.append([values.get(index, "") for index in range(max(values) + 1)])
    assert rows, f"No rows found in XLSX sheet: {path}"
    headers = rows[0]
    assert "Img-Name" in headers and "Img-Text" in headers and "Img-Text-Label" in headers, (
        f"Unexpected MultiBully XLSX headers: {headers}"
    )
    return [
        {header: values[index] if index < len(values) else "" for index, header in enumerate(headers) if header}
        for values in rows[1:]
    ]


def _multibully_image_path(image_dpath: Path, image_name: str) -> Path | None:
    direct = image_dpath / image_name
    if direct.exists():
        return direct
    candidates = list(image_dpath.glob(f"{Path(image_name).stem}.*"))
    return candidates[0] if len(candidates) == 1 else None


def _is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError):
        return False
    return True


def _multibully_eval_ids(rows: list[tuple[dict[str, str], Path, str]], n_eval: int, seed: int) -> set[str]:
    by_label: dict[str, list[tuple[dict[str, str], Path, str]]] = {"Nonbully": [], "Bully": []}
    for row in rows:
        by_label[row[2]].append(row)
    n_available = sum(len(items) for items in by_label.values())
    assert n_eval <= n_available, f"MultiBully has only {n_available} valid labeled rows, not {n_eval}."
    target_by_label = {label: n_eval * len(items) // n_available for label, items in by_label.items()}
    remaining = n_eval - sum(target_by_label.values())
    for label in sorted(by_label, key=lambda value: (-(n_eval * len(by_label[value]) % n_available), value)):
        if remaining == 0:
            break
        target_by_label[label] += 1
        remaining -= 1
    selected: set[str] = set()
    for label, items in by_label.items():
        rng = random.Random(f"multibully:{seed}:{label}")
        rng.shuffle(items)
        selected.update(row[0]["Img-Name"] for row in items[: target_by_label[label]])
    assert len(selected) == n_eval, f"MultiBully holdout has {len(selected)}, expected {n_eval}."
    return selected


def _load_multibully(cfg: MemeDatasetConfig, splits: set[str]) -> list[MemeSample]:
    assert cfg.multibully_eval_size > 0, "MultiBully evaluation holdout must contain at least one sample."
    root = Path(cfg.root) / "MultiBully"
    annotation_fpath = root / "Cyberbully_corrected_emotion_sentiment.xlsx"
    image_dpath = root / "bully_data"
    assert annotation_fpath.exists(), f"MultiBully annotation file does not exist: {annotation_fpath}"
    assert image_dpath.exists(), f"MultiBully image directory does not exist: {image_dpath}"
    valid_rows: list[tuple[dict[str, str], Path, str]] = []
    for row in _read_xlsx_first_sheet(annotation_fpath):
        label = row.get("Img-Text-Label", "").strip()
        image_name = row.get("Img-Name", "").strip()
        if label not in {"Nonbully", "Bully"} or not image_name:
            continue
        image_path = _multibully_image_path(image_dpath, image_name)
        if image_path is None:
            _MISSING_IMAGE_COUNTS["MultiBully"] += 1
            _MISSING_IMAGE_EXAMPLES.setdefault("MultiBully", str(image_dpath / image_name))
            continue
        if not _is_readable_image(image_path):
            _UNREADABLE_IMAGE_COUNTS["MultiBully"] += 1
            _UNREADABLE_IMAGE_EXAMPLES.setdefault("MultiBully", str(image_path))
            continue
        valid_rows.append((row, image_path, label))
    assert valid_rows, "No valid MultiBully rows with a label and image were found."
    eval_ids = _multibully_eval_ids(valid_rows, cfg.multibully_eval_size, cfg.multibully_eval_seed)
    include_train = "train" in splits
    include_eval = bool({"validation", "val", "test"} & splits)
    samples: list[MemeSample] = []
    for row, image_path, label in valid_rows:
        sample_id = row["Img-Name"]
        in_eval = sample_id in eval_ids
        if in_eval and not include_eval:
            continue
        if not in_eval and not include_train:
            continue
        samples.append(
            MemeSample(
                dataset_name="MultiBully",
                task_name=MULTIBULLY_SCHEMA.task_name,
                sample_id=sample_id,
                split="validation" if in_eval else "train",
                image_path=str(image_path),
                meme_text=row.get("Img-Text", "").strip(),
                schema=MULTIBULLY_SCHEMA,
                gold_fields={"label": label},
                raw_fields=row,
            )
        )
    assert samples, "No MultiBully samples matched the requested split."
    return samples


def _exist_majority_label(votes: Any) -> str | None:
    if not isinstance(votes, list):
        return None
    normalized = [str(vote).strip().upper() for vote in votes if str(vote).strip().upper() in {"YES", "NO"}]
    if not normalized:
        return None
    counts = Counter(normalized)
    if len(counts) == 2 and counts["YES"] == counts["NO"]:
        return None
    return "sexist" if counts["YES"] > counts["NO"] else "non-sexist"


def _load_exist_translation_map(path: str) -> dict[str, str]:
    translation_fpath = Path(path)
    assert translation_fpath.exists(), f"EXIST translation file does not exist: {translation_fpath}"
    payload = json.loads(translation_fpath.read_text())
    assert isinstance(payload, dict), "EXIST translations must be a JSON object mapping id_EXIST to translated text."
    translations = {str(sample_id): str(text).strip() for sample_id, text in payload.items() if str(text).strip()}
    assert translations, f"EXIST translation file has no non-empty translations: {translation_fpath}"
    return translations


def _exist_eval_ids(rows: list[tuple[dict[str, Any], str]], language: str, n_eval: int, seed: int) -> set[str]:
    by_label: dict[str, list[dict[str, Any]]] = {"non-sexist": [], "sexist": []}
    for row, label in rows:
        if str(row.get("lang", "")).lower() == language:
            by_label[label].append(row)
    n_available = sum(len(items) for items in by_label.values())
    assert n_eval <= n_available, f"EXIST {language} has only {n_available} majority-labeled samples, not {n_eval}."
    target_by_label = {label: n_eval * len(items) // n_available for label, items in by_label.items()}
    n_remaining = n_eval - sum(target_by_label.values())
    for label in sorted(by_label, key=lambda value: (-(n_eval * len(by_label[value]) % n_available), value)):
        if n_remaining == 0:
            break
        target_by_label[label] += 1
        n_remaining -= 1
    selected: set[str] = set()
    for label, items in by_label.items():
        rng = random.Random(f"{seed}:{language}:{label}")
        rng.shuffle(items)
        selected.update(str(row["id_EXIST"]) for row in items[: target_by_label[label]])
    assert len(selected) == n_eval, f"EXIST {language} holdout had {len(selected)}, expected {n_eval}."
    return selected


def _load_exist(cfg: MemeDatasetConfig, splits: set[str]) -> list[MemeSample]:
    train_languages = {language.lower() for language in cfg.exist_train_languages}
    eval_languages = {language.lower() for language in cfg.exist_eval_languages}
    assert train_languages or eval_languages, "Set --exist_train_languages and/or --exist_eval_languages for EXIST."
    requested_languages = train_languages | eval_languages
    assert cfg.exist_eval_per_language > 0, "EXIST evaluation holdout must contain at least one sample per language."
    assert cfg.exist_text_mode == "raw" or cfg.exist_translation_path, "--exist_translation_path is required when --exist_text_mode translation."

    root = Path(cfg.root) / "EXIST_2025_Memes_Dataset"
    records = json.loads((root / "training" / "EXIST2025_training.json").read_text())
    assert isinstance(records, dict), "Expected EXIST2025_training.json to be a JSON object."
    labeled_rows = [(row, label) for row in records.values() if isinstance(row, dict) and (label := _exist_majority_label(row.get("labels_task2_1"))) is not None]
    eval_ids = {
        language: _exist_eval_ids(labeled_rows, language, cfg.exist_eval_per_language, cfg.exist_eval_seed)
        for language in sorted(eval_languages)
    }
    translations = _load_exist_translation_map(cfg.exist_translation_path) if cfg.exist_text_mode == "translation" else {}
    samples: list[MemeSample] = []
    include_train = "train" in splits
    include_eval = bool({"validation", "val", "test"} & splits)
    for row, label in labeled_rows:
        language = str(row.get("lang", "")).lower()
        if language not in requested_languages:
            continue
        sample_id = str(row["id_EXIST"])
        in_eval = sample_id in eval_ids.get(language, set())
        if in_eval and (language not in eval_languages or not include_eval):
            continue
        if not in_eval and (language not in train_languages or not include_train):
            continue
        meme_text = str(row.get("text", ""))
        if cfg.exist_text_mode == "translation" and language in set(cfg.exist_translation_languages):
            assert sample_id in translations, f"Missing EXIST translation for sample {sample_id}."
            meme_text = translations[sample_id]
        sample = _make_sample(
            dataset_name="EXIST_2025_Memes_Dataset",
            task_name=EXIST_SEXISM_SCHEMA.task_name,
            sample_id=sample_id,
            split="validation" if in_eval else "train",
            image_path=root / "training" / str(row["path_memes"]),
            meme_text=meme_text,
            schema=EXIST_SEXISM_SCHEMA,
            gold_fields={"label": label},
            raw_fields=row,
        )
        if sample is not None:
            samples.append(sample)
    assert samples, "No EXIST samples matched the requested language routing."
    return samples


def load_meme_samples(cfg: MemeDatasetConfig) -> list[MemeSample]:
    _MISSING_IMAGE_COUNTS.clear()
    _MISSING_IMAGE_EXAMPLES.clear()
    _UNREADABLE_IMAGE_COUNTS.clear()
    _UNREADABLE_IMAGE_EXAMPLES.clear()
    root = Path(cfg.root)
    assert root.exists(), f"Dataset root does not exist: {root}"
    requested = set(cfg.datasets)
    splits = set(cfg.splits)
    all_samples: list[MemeSample] = []
    if "Crisis_HateMM" in requested:
        all_samples.extend(_load_crisis_hatemm(root / "Crisis_HateMM", cfg.include_unlabeled, splits))
    if "HarMeme" in requested:
        all_samples.extend(_load_harmeme(root / "HarMeme", splits))
    if "Hateful_Memes_Dataset" in requested:
        all_samples.extend(_load_hateful_memes(root / "Hateful_Memes_Dataset", splits, cfg.include_unlabeled, cfg.fhm_labels_path, cfg.fhm_label_split))
    if "MAMI" in requested:
        all_samples.extend(_load_mami(root / "MAMI", splits, cfg.include_unlabeled))
    if "MMHS150K" in requested:
        all_samples.extend(_load_mmhs150k(root / "MMHS150K", splits))
    if "MultiOFF" in requested:
        all_samples.extend(_load_multioff(root / "MultiOFF", splits))
    if "MultiBully" in requested:
        all_samples.extend(_load_multibully(cfg, splits))
    if "EXIST_2025_Memes_Dataset" in requested:
        all_samples.extend(_load_exist(cfg, splits))
    all_samples = _finalize_splits(all_samples, splits, cfg.synthetic_split_seed)
    if cfg.sample_limit is not None and len(all_samples) > cfg.sample_limit:
        rng = random.Random(42)
        rng.shuffle(all_samples)
        all_samples = all_samples[: cfg.sample_limit]
    for dataset_name, count in sorted(_MISSING_IMAGE_COUNTS.items()):
        warnings.warn(
            f"Skipped {count} samples from {dataset_name} because image files were missing. "
            f"Example: {_MISSING_IMAGE_EXAMPLES[dataset_name]}",
            stacklevel=2,
        )
    for dataset_name, count in sorted(_UNREADABLE_IMAGE_COUNTS.items()):
        warnings.warn(
            f"Skipped {count} samples from {dataset_name} because image files were unreadable. "
            f"Example: {_UNREADABLE_IMAGE_EXAMPLES[dataset_name]}",
            stacklevel=2,
        )
    return sorted(all_samples, key=lambda sample: (sample.dataset_name, sample.task_name, sample.split, sample.sample_id))


def build_simple_classify_prompt(sample: MemeSample) -> str:
    """Yes/no classification prompt for constrained binary decoding.

    Returns a direct yes/no question about whether the meme belongs to the
    positive class (last label in the primary field).  Used with logit
    scoring so the model never free-generates JSON, eliminating JSON-collapse
    under SAE interventions.
    """
    schema = sample.schema
    primary_labels = schema.primary_field.labels
    pos_label = primary_labels[-1] if primary_labels else "positive"
    text_ctx = f'\nMeme text: "{sample.meme_text}"' if sample.meme_text else "\nMeme text: [none]"
    return (
        f"This is a meme.{text_ctx}\n"
        f"Task: {schema.task_description}\n"
        f"Does this meme qualify as '{pos_label}'?\n"
        "Answer only with: yes or no."
    )


def build_primary_label_prompt(sample: MemeSample) -> str:
    """Prompt for constrained primary-label selection."""
    schema = sample.schema
    field = schema.primary_field
    labels = ", ".join(f"'{label}'" for label in field.labels)
    text_ctx = f'\nMeme text: "{sample.meme_text}"' if sample.meme_text else "\nMeme text: [none]"
    return (
        f"This is a meme.{text_ctx}\n"
        f"Task: {schema.task_description}\n"
        f"Choose exactly one label for '{field.name}' from: {labels}.\n"
        "Answer only with the label text."
    )


def build_classification_prompt(sample: MemeSample) -> str:
    schema = sample.schema
    required_fields = [schema.primary_field, *schema.secondary_fields]
    lines = [
        "You are analyzing a meme for a structured classification task.",
        schema.task_description,
        "Use both the image and any OCR-style meme text together.",
        "Respond with valid JSON only.",
        "",
        "Required JSON schema:",
        "{",
    ]
    for field in required_fields:
        if field.multi_label:
            lines.append(
                f'  "{field.name}": ["<zero or more of {", ".join(field.labels)}>"],'
            )
        else:
            lines.append(f'  "{field.name}": "<one of {", ".join(field.labels)}>",')
    lines.append('  "reasoning": "<brief explanation grounded in the image and text>"')
    lines.extend(
        [
            "}",
            "",
            "Keep the reasoning short and evidence-based.",
            "",
            f"Meme text: {sample.meme_text or '[no OCR text available]'}",
        ]
    )
    return "\n".join(lines)
