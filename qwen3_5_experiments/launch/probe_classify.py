

import json
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from safetensors.torch import load_file
from simple_parsing import parse, Serializable, list_field

from gemma3_experiments.residual_sae.normalization import load_residual_stats
from ..models import load_residual_sae
from ..task_labels import fit_scaled_logreg, predict_scaled

logger = logging.getLogger(__name__)

POSITIVE_LABELS = {
    "hateful_memes_binary": "hateful",
    "subtask_a_hate_speech": "Hate Speech",
    "misogyny_and_subtypes": "misogynous",
    "offensive_detection": "offensive",
}


@dataclass
class ProbeClassifyConfig(Serializable):
    probe_dir: str = "./qwen_probe_pilot"
    residual_cache_dir: str = "./qwen_residual_sae_cache/residual"
    # Optional separate train cache dir. When set, probes are fitted on rows from
    # this directory (split=="train") and evaluated on residual_cache_dir rows.
    # If empty, both train and eval rows are loaded from residual_cache_dir (legacy).
    train_cache_dir: str = ""
    residual_sae_path: str = "./qwen_residual_sae_ckpts/best_finite"
    output_path: str = "./probe_classification_results.jsonl"
    device: str = "cuda:0"
    aggregation: str = "max"
    splits: list[str] = list_field("validation")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fd:
        for line in fd:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(ProbeClassifyConfig)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    probe_dir = Path(cfg.probe_dir)
    cache_dir = Path(cfg.residual_cache_dir)
    device = cfg.device
    # train_dir: directory that contains split=="train" rows for fitting the probe.
    # eval_dir (cache_dir): directory that contains the held-out evaluation rows.
    train_dir = Path(cfg.train_cache_dir) if cfg.train_cache_dir else cache_dir

    # Load SAE
    sae = load_residual_sae(cfg.residual_sae_path, device=device)
    sae.eval()
    residual_stats = load_residual_stats(cfg.residual_sae_path)
    logger.info("SAE loaded: %d latents", sae.num_latents)
    if train_dir != cache_dir:
        logger.info("Using separate train cache: %s", train_dir)

    # Load manifest for evaluation split(s)
    manifest_rows = _read_jsonl(cache_dir / "manifest.jsonl")
    splits = set(cfg.splits)
    manifest_rows = [r for r in manifest_rows if r.get("split") in splits]
    logger.info("Manifest: %d rows in splits %s", len(manifest_rows), cfg.splits)

    # Load top_features.json to know which tasks have probes
    tf_path = probe_dir / "top_features.json"
    if not tf_path.exists():
        raise FileNotFoundError(f"No top_features.json in {probe_dir}. Run probe_sae_features first.")
    probed_tasks = set(json.load(tf_path.open()).keys())
    logger.info("Probed tasks: %s", probed_tasks)

    # Re-train probes on train split and evaluate on validation
    # (since we need to re-run training since sklearn models aren't serialized)
    # Load probe training results to get hyperparameters
    probe_results = {r["task_name"]: r for r in _read_jsonl(probe_dir / "probe_results.jsonl")}

    # Load train split residuals for probe re-training
    train_manifest = _read_jsonl(train_dir / "manifest.jsonl")
    train_manifest = [r for r in train_manifest if r.get("split") == "train"]

    results_out = []
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for task_name, pos_label in POSITIVE_LABELS.items():
        # Get train rows for this task
        task_train = [(r, 1 if str(list(r.get("gold_fields",{}).values())[0]).lower()==pos_label.lower() else 0)
                      for r in train_manifest if r.get("task_name")==task_name
                      and r.get("gold_fields")]
        # Get val rows for this task
        task_val = [(r, 1 if str(list(r.get("gold_fields",{}).values())[0]).lower()==pos_label.lower() else 0)
                    for r in manifest_rows if r.get("task_name")==task_name
                    and r.get("gold_fields")]

        if not task_train or not task_val:
            continue
        n_pos_train = sum(1 for _,l in task_train if l==1)
        n_neg_train = sum(1 for _,l in task_train if l==0)
        if n_pos_train < 20 or n_neg_train < 20:
            continue

        logger.info("Task %s: n_train=%d n_val=%d", task_name, len(task_train), len(task_val))

        def extract(row_label_pairs, base_dir=None):
            base = base_dir or cache_dir
            by_shard = defaultdict(list)
            for row, label in row_label_pairs:
                by_shard[row["shard_file"]].append((row, label))
            X_list, y_list = [], []
            for shard_name, pairs in sorted(by_shard.items()):
                shard_path = base / shard_name
                if not shard_path.exists():
                    continue
                try:
                    t = load_file(str(shard_path))
                except Exception as e:
                    logger.warning("Failed: %s", e)
                    continue
                res_all = t["residual"].to(device=device, dtype=torch.float32)
                for row, label in pairs:
                    sample_res = res_all[row["offset"]: row["offset"]+row["length"]]
                    if sample_res.shape[0] == 0:
                        continue
                    sae_in = residual_stats.normalize(sample_res) if residual_stats else sample_res
                    with torch.no_grad():
                        acts, _ = sae.encode(sae_in)
                        vec = acts.max(dim=0).values if cfg.aggregation=="max" else acts.mean(dim=0)
                    X_list.append(vec.cpu().float().numpy())
                    y_list.append(label)
            return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)

        X_train, y_train = extract(task_train, train_dir)
        X_val, y_val = extract(task_val)

        if len(X_train) < 10:
            continue

        clf, scaler = fit_scaled_logreg(X_train, y_train, C=1.0, max_iter=3000)
        y_pred = predict_scaled(clf, scaler, X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="binary", zero_division=0)

        ds_name = task_val[0][0]["dataset_name"] if task_val else "?"
        logger.info("  %s/%s: acc=%.4f F1=%.4f (n_val=%d)", ds_name, task_name, acc, f1, len(y_val))

        for (row, label), pred in zip(task_val, y_pred):
            results_out.append({
                **{k: row[k] for k in ("dataset_name","task_name","sample_id","split","gold_fields")},
                "probe_prediction": int(pred),
                "probe_label": pos_label if pred == 1 else "not_" + pos_label,
                "gold_label": list(row.get("gold_fields",{}).values())[0] if row.get("gold_fields") else None,
                "correct": int(pred == label),
                "probe_acc_task": round(float(acc), 4),
                "probe_f1_task": round(float(f1), 4),
            })

    with output_path.open("w") as fd:
        for r in results_out:
            fd.write(json.dumps(r) + "\n")
    logger.info("Wrote %d probe classification results to %s", len(results_out), output_path)


if __name__ == "__main__":
    main()
