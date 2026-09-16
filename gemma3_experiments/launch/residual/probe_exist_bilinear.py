

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from simple_parsing import parse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MaxAbsScaler

from gemma3_experiments.launch.residual.probe_validate import (
    _extract_features,
    _extract_features_multi,
    _read_jsonl,
    _row_gold_label,
    _row_sample_key,
    load_residual_sae,
    load_residual_stats,
    resolve_visible_device,
)

logger = logging.getLogger(__name__)


@dataclass
class ExistBilinearConfig:
    residual_sae_path: str = ""
    train_cache_dir: str = ""
    val_cache_dir: str = ""
    output_dir: str = ""
    dataset_name: str = "EXIST_2025_Memes_Dataset"
    positive_label: str = "sexist"
    negative_label: str = "non-sexist"
    aggregation: str = "max"
    batch_size: int = 256
    top_k_features: int = 256
    rank: int = 32
    n_epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 30
    seed: int = 42
    device: str = "cuda:0"


class BilinearReadout(torch.nn.Module):
    def __init__(self, n_features: int, rank: int) -> None:
        super().__init__()
        self.image_linear = torch.nn.Linear(n_features, 1)
        self.prompt_linear = torch.nn.Linear(n_features, 1)
        self.image_factor = torch.nn.Parameter(torch.empty(n_features, rank))
        self.prompt_factor = torch.nn.Parameter(torch.empty(n_features, rank))
        self.bilinear_scale = torch.nn.Parameter(torch.tensor(0.1))
        torch.nn.init.xavier_uniform_(self.image_factor)
        torch.nn.init.xavier_uniform_(self.prompt_factor)

    def forward(self, image: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        linear = self.image_linear(image).squeeze(-1) + self.prompt_linear(prompt).squeeze(-1)
        interaction = ((image @ self.image_factor) * (prompt @ self.prompt_factor)).sum(dim=-1)
        return linear + self.bilinear_scale * interaction


def _aligned_pair(
    feature_sets: dict[str, tuple[list[str], Any]],
) -> tuple[list[str], Any, Any]:
    image_keys, image_csr = feature_sets["image"]
    prompt_keys, prompt_csr = feature_sets["prompt"]
    shared = set(image_keys) & set(prompt_keys)
    keys = [key for key in image_keys if key in shared]
    assert keys, "No samples had both image and prompt SAE features."
    image_i = {key: index for index, key in enumerate(image_keys)}
    prompt_i = {key: index for index, key in enumerate(prompt_keys)}
    return keys, image_csr[[image_i[key] for key in keys]], prompt_csr[[prompt_i[key] for key in keys]]


def _select_features(image, prompt, labels: np.ndarray, top_k: int) -> np.ndarray:
    assert labels.any() and (~labels).any(), "Both classes are required for bilinear feature selection."
    image_delta = np.asarray(image[labels].mean(axis=0) - image[~labels].mean(axis=0)).ravel()
    prompt_delta = np.asarray(prompt[labels].mean(axis=0) - prompt[~labels].mean(axis=0)).ravel()
    scores = np.abs(image_delta) + np.abs(prompt_delta)
    n_select = min(top_k, scores.size)
    return np.argpartition(scores, -n_select)[-n_select:]


def _metrics(gold: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(gold, pred)), 6),
        "macro_f1": round(float(f1_score(gold, pred, average="macro")), 6),
        "predicted_positive_rate": round(float(pred.mean()), 6),
    }


def run(cfg: ExistBilinearConfig) -> None:
    assert cfg.top_k_features > 0, "top_k_features must be positive."
    assert cfg.rank > 0, "rank must be positive."
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_visible_device(cfg.device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sae = load_residual_sae(cfg.residual_sae_path, device=device)
    stats = load_residual_stats(cfg.residual_sae_path)
    n_features = sae.num_latents
    extract_kw = {
        "sae": sae,
        "stats": stats,
        "aggregation": cfg.aggregation,
        "batch_size": cfg.batch_size,
        "device": device,
        "n_features": n_features,
    }

    train_cache = Path(cfg.train_cache_dir)
    train_rows = [
        row for row in _read_jsonl(train_cache / "manifest.jsonl")
        if row.get("dataset_name") == cfg.dataset_name and row.get("split") == "train" and _row_gold_label(row) is not None
    ]
    assert train_rows, f"No {cfg.dataset_name} train rows in {train_cache}."
    train_sets = {
        token_selection: _extract_features(train_rows, train_cache, token_selection=token_selection, **extract_kw)
        for token_selection in ("image", "prompt")
    }
    train_keys, train_image, train_prompt = _aligned_pair(train_sets)
    train_by_key = {_row_sample_key(row): row for row in train_rows}
    train_labels = np.asarray([_row_gold_label(train_by_key[key]) == cfg.positive_label for key in train_keys])

    val_cache = Path(cfg.val_cache_dir)
    val_rows = [
        row for row in _read_jsonl(val_cache / "manifest.jsonl")
        if row.get("dataset_name") == cfg.dataset_name and row.get("split") in {"validation", "val"} and _row_gold_label(row) is not None
    ]
    assert val_rows, f"No {cfg.dataset_name} validation rows in {val_cache}."
    for row in val_rows:
        row["_cache_dir"] = str(val_cache)
    val_sets = {
        token_selection: _extract_features_multi(val_rows, token_selection=token_selection, **extract_kw)
        for token_selection in ("image", "prompt")
    }
    val_keys, val_image, val_prompt = _aligned_pair(val_sets)
    val_by_key = {_row_sample_key(row): row for row in val_rows}
    val_labels = np.asarray([_row_gold_label(val_by_key[key]) == cfg.positive_label for key in val_keys])

    feature_i = _select_features(train_image, train_prompt, train_labels, cfg.top_k_features)
    image_scaler = MaxAbsScaler()
    prompt_scaler = MaxAbsScaler()
    train_image = image_scaler.fit_transform(train_image[:, feature_i]).toarray().astype(np.float32)
    train_prompt = prompt_scaler.fit_transform(train_prompt[:, feature_i]).toarray().astype(np.float32)
    val_image = image_scaler.transform(val_image[:, feature_i]).toarray().astype(np.float32)
    val_prompt = prompt_scaler.transform(val_prompt[:, feature_i]).toarray().astype(np.float32)

    model = BilinearReadout(len(feature_i), cfg.rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pos_weight = torch.tensor((~train_labels).sum() / train_labels.sum(), dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_image_t = torch.from_numpy(train_image).to(device)
    train_prompt_t = torch.from_numpy(train_prompt).to(device)
    train_labels_t = torch.from_numpy(train_labels.astype(np.float32)).to(device)
    best_state = None
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(train_image_t, train_prompt_t), train_labels_t)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= cfg.patience:
            logger.info("Stopped at epoch %d after %d stale training-loss epochs.", epoch, stale_epochs)
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        probability = torch.sigmoid(model(torch.from_numpy(val_image).to(device), torch.from_numpy(val_prompt).to(device))).cpu().numpy()
    pred = probability >= 0.5
    summary = {
        "protocol": "train_only_feature_selection_and_early_stopping",
        "n_train": len(train_keys),
        "n_validation": len(val_keys),
        "rank": cfg.rank,
        "top_k_features": len(feature_i),
        "best_train_loss": round(best_loss, 6),
        "metrics": _metrics(val_labels, pred),
        "feature_indices": [int(index) for index in sorted(feature_i.tolist())],
    }
    (output_dir / "bilinear_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "probe_val_predictions.jsonl").open("w") as fd:
        for key, gold, predicted, score in zip(val_keys, val_labels, pred, probability, strict=True):
            row = val_by_key[key]
            fd.write(json.dumps({
                "sample_key": key,
                "dataset_name": row["dataset_name"],
                "task_name": row["task_name"],
                "sample_id": row["sample_id"],
                "language": row.get("raw_fields", {}).get("lang"),
                "gold_label": cfg.positive_label if gold else cfg.negative_label,
                "probe_pred": cfg.positive_label if predicted else cfg.negative_label,
                "probe_proba": {
                    cfg.positive_label: round(float(score), 6),
                    cfg.negative_label: round(float(1.0 - score), 6),
                },
            }) + "\n")
    torch.save({"state_dict": best_state, "feature_indices": feature_i, "config": vars(cfg)}, output_dir / "bilinear_probe.pt")
    logger.info("Wrote bilinear transfer results to %s", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(parse(ExistBilinearConfig))


if __name__ == "__main__":
    main()