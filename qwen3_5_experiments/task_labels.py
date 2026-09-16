

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


# ── Natively binary tasks: task_name -> positive class label string ──────────
BINARY_TASKS: dict[str, str] = {
    "hateful_memes_binary": "hateful",
    "subtask_a_hate_speech": "Hate Speech",
    "misogyny_and_subtypes": "misogynous",
    "offensive_detection": "offensive",
}

# Dataset name each binary task belongs to (for display / lookups).
BINARY_TASK_DATASET: dict[str, str] = {
    "hateful_memes_binary": "Hateful_Memes_Dataset",
    "subtask_a_hate_speech": "Crisis_HateMM",
    "misogyny_and_subtypes": "MAMI",
    "offensive_detection": "MultiOFF",
}

# gold_fields key for each binary task (all four native binary schemas use the
# primary field name "label" — see gemma3_residual_sae.residual_sae.datasets).
BINARY_TASK_FIELD: dict[str, str] = {
    "hateful_memes_binary": "label",
    "subtask_a_hate_speech": "label",
    "misogyny_and_subtypes": "label",
    "offensive_detection": "label",
}


@dataclass(frozen=True)
class MulticlassTaskSpec:
    dataset_name: str
    task_name: str
    field: str                       # key inside gold_fields
    classes: tuple[str, ...]         # native fine-grained label set
    binary_negative: str             # collapsed negative class name
    binary_positive: str             # collapsed positive class name
    collapse: Callable[[str], str]   # native label -> {binary_negative, binary_positive}


def _harmfulness_collapse(v: str) -> str:
    return "not harmful" if v == "not harmful" else "harmful"


def _hate_category_collapse(v: str) -> str:
    return "NotHate" if v == "NotHate" else "Hate"


# ── Natively multi-class tasks that also get a binary collapse ───────────────
MULTICLASS_TASKS: dict[str, MulticlassTaskSpec] = {
    "harmfulness_and_target": MulticlassTaskSpec(
        dataset_name="HarMeme",
        task_name="harmfulness_and_target",
        field="harmfulness",
        classes=("not harmful", "somewhat harmful", "very harmful"),
        binary_negative="not harmful",
        binary_positive="harmful",
        collapse=_harmfulness_collapse,
    ),
    "hate_category": MulticlassTaskSpec(
        dataset_name="MMHS150K",
        task_name="hate_category",
        field="label",
        classes=("NotHate", "Racist", "Sexist", "Homophobe", "Religion", "OtherHate"),
        binary_negative="NotHate",
        binary_positive="Hate",
        collapse=_hate_category_collapse,
    ),
}

# All six datasets, keyed the same way as they appear in manifests / predictions.
ALL_SIX_DATASETS = (
    "Crisis_HateMM",
    "Hateful_Memes_Dataset",
    "HarMeme",
    "MAMI",
    "MMHS150K",
    "MultiOFF",
)


def extract_gold_value(gold_fields: dict, field: str) -> str | None:
    """Look up a single gold field value by name (robust to extra secondary fields)."""
    if not gold_fields:
        return None
    val = gold_fields.get(field)
    if val is None:
        return None
    if isinstance(val, list):
        return val[0] if val else None
    return str(val)


def compute_metrics(y_true: list[str], y_pred: list[str], labels: list[str] | None = None) -> dict:
    """Accuracy + macro-F1 (the locked protocol metric for binary tasks)."""
    from sklearn.metrics import accuracy_score, f1_score

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "n": len(y_true),
    }


def fit_scaled_logreg(
    X_train,
    y_train,
    C: float = 1.0,
    max_iter: int = 3000,
    class_weight: str | None = "balanced",
):
    pass
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MaxAbsScaler

    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    clf = LogisticRegression(C=C, max_iter=max_iter, class_weight=class_weight, solver="lbfgs")
    clf.fit(X_train_scaled, y_train)
    return clf, scaler


def predict_scaled(clf, scaler, X):
    return clf.predict(scaler.transform(X))

