

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from simple_parsing import ArgumentGenerationMode, parse

logger = logging.getLogger(__name__)


@dataclass
class BilinearFactorConfig:
    checkpoint_dir: str = ""           # output_dir from probe_fhm_crosscoder_pairwise
    feature_descriptions_path: str = ""  # feature_descriptions.json (NL labels)
    audit_path: str = ""               # confounder audit JSONL (for score/direction context)
    top_k: int = 8                     # top features to show per U/V column
    output_dir: str = ""               # if empty, uses checkpoint_dir


def _load_descriptions(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items() if not isinstance(v, dict)}


def _load_audit_scores(path: Path) -> dict[int, dict]:
    """Return {feature_idx: {score, direction, sign_consistency}} from audit JSONL."""
    if not path.exists():
        return {}
    scores: dict[int, dict] = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        fid = int(r["feature_idx"])
        if fid not in scores or r.get("score", 0) > scores[fid].get("score", 0):
            scores[fid] = {
                "score": r.get("score", 0.0),
                "direction": r.get("direction", ""),
                "sign_consistency": r.get("sign_consistency", 0.0),
            }
    return scores


def run(cfg: BilinearFactorConfig) -> None:
    ckpt_dir = Path(cfg.checkpoint_dir)
    out_dir = Path(cfg.output_dir) if cfg.output_dir else ckpt_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ────────────────────────────────────────────────────────
    state_dict_path = ckpt_dir / "model_checkpoint.pt"
    feat_idx_path   = ckpt_dir / "bilinear_feature_indices.json"
    assert state_dict_path.exists(), f"Checkpoint not found: {state_dict_path}"
    assert feat_idx_path.exists(),   f"Feature indices not found: {feat_idx_path}"

    state = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    feat_indices = json.loads(feat_idx_path.read_text())["feature_indices"]  # list[int], len=K

    # Extract U and V (or W_bil for full-rank)
    if "bil_U" in state:
        U = state["bil_U"].numpy()  # [K, r]
        V = state["bil_V"].numpy()  # [K, r]
        rank = U.shape[1]
        logger.info("Loaded low-rank bilinear: K=%d r=%d", len(feat_indices), rank)
    elif "W_bil" in state:
        # Full-rank: SVD to get dominant directions
        W = state["W_bil"].numpy()  # [K, K]
        U_full, S, Vt = np.linalg.svd(W, full_matrices=False)
        rank = min(cfg.top_k, len(S))
        U = U_full[:, :rank] * S[:rank]   # absorb singular values into U
        V = Vt[:rank, :].T                 # [K, rank]
        logger.info("Full-rank W_bil (%dx%d), SVD top-%d directions", W.shape[0], W.shape[1], rank)
    else:
        raise ValueError("No bilinear parameters found in checkpoint state dict.")

    # ── Load descriptions and audit scores ─────────────────────────────────────
    descriptions = _load_descriptions(Path(cfg.feature_descriptions_path))
    audit_scores = _load_audit_scores(Path(cfg.audit_path)) if cfg.audit_path else {}

    # ── Analyse each direction ─────────────────────────────────────────────────
    lines = [
        "=" * 70,
        "Bilinear Factor Analysis",
        f"  Checkpoint:  {ckpt_dir}",
        f"  K={len(feat_indices)} confounder features,  r={rank} interaction directions",
        "=" * 70, "",
    ]

    results = []
    for d in range(rank):
        u_col = U[:, d]   # image weights for direction d
        v_col = V[:, d]   # prompt weights for direction d
        direction_score = float(np.linalg.norm(u_col) * np.linalg.norm(v_col))

        # Top features by |weight| in each column
        u_top_idx = np.argsort(-np.abs(u_col))[:cfg.top_k]
        v_top_idx = np.argsort(-np.abs(v_col))[:cfg.top_k]

        def feat_line(col_idx: int, weight: float) -> str:
            fid = feat_indices[col_idx]
            desc = descriptions.get(fid, "")[:80] if descriptions else ""
            audit = audit_scores.get(fid, {})
            sc = audit.get("sign_consistency", 0.0)
            return f"    f{fid:5d}  w={weight:+.4f}  sc={sc:.3f}  {desc!r}"

        img_lines = [feat_line(i, u_col[i]) for i in u_top_idx]
        pmt_lines = [feat_line(i, v_col[i]) for i in v_top_idx]

        dir_block = [
            f"── Direction {d+1}/{rank}  (strength={direction_score:.4f}) ──────────────────",
            f"  IMAGE features (U[:,{d}]):",
            *img_lines,
            f"  PROMPT features (V[:,{d}]):",
            *pmt_lines,
            "",
        ]
        lines.extend(dir_block)

        results.append({
            "direction": d,
            "strength": round(direction_score, 5),
            "image_features": [
                {"feature_idx": int(feat_indices[i]), "weight": round(float(u_col[i]), 5),
                 "description": descriptions.get(feat_indices[i], ""),
                 "sign_consistency": audit_scores.get(feat_indices[i], {}).get("sign_consistency", 0.0)}
                for i in u_top_idx
            ],
            "prompt_features": [
                {"feature_idx": int(feat_indices[i]), "weight": round(float(v_col[i]), 5),
                 "description": descriptions.get(feat_indices[i], ""),
                 "sign_consistency": audit_scores.get(feat_indices[i], {}).get("sign_consistency", 0.0)}
                for i in v_top_idx
            ],
        })

    # Sort by strength descending
    results.sort(key=lambda x: -x["strength"])

    # ── Write outputs ──────────────────────────────────────────────────────────
    (out_dir / "bilinear_factor_report.txt").write_text("\n".join(lines))
    (out_dir / "bilinear_factors.json").write_text(json.dumps(results, indent=2))
    logger.info("Wrote factor report to %s", out_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(BilinearFactorConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
