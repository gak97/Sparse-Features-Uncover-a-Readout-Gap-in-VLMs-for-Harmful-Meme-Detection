from dataclasses import dataclass
from typing import Literal

from simple_parsing import Serializable, field, list_field


DEFAULT_DATA_ROOT = "./data"
DEFAULT_DATASETS = (
    "Crisis_HateMM",
    "HarMeme",
    "Hateful_Memes_Dataset",
    "MAMI",
    "MMHS150K",
    "MultiOFF",
)


@dataclass
class MemeDatasetConfig(Serializable):
    root: str = DEFAULT_DATA_ROOT
    datasets: list[str] = list_field(*DEFAULT_DATASETS)
    splits: list[str] = list_field("train")
    include_unlabeled: bool = False
    fhm_labels_path: str = ""
    fhm_label_split: str = ""
    sample_limit: int | None = None
    synthetic_split_seed: int = 42


@dataclass
class QwenResidualConfig(Serializable):
    model_id: str = "Qwen/Qwen3.5-9B-Base"
    sae_repo_id: str = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
    sae_layer: int = 20
    # Qwen3.5-9B-Base uses model.model.layers[N] (nested 'model' submodule)
    hook_module_path: str = "model.language_model.layers.20"
    model_device: str = "cuda:0"
    sae_device: str = "cuda:0"
    max_new_tokens: int = 96
    token_selection: Literal["all", "image", "prompt", "generated", "image_and_generated"] = "all"
    capture_generation: bool = True
    sae_chunk_size: int = 128


@dataclass
class DenseActivationCacheConfig(Serializable):
    dataset: MemeDatasetConfig = field(default_factory=MemeDatasetConfig)
    qwen: QwenResidualConfig = field(default_factory=QwenResidualConfig)
    output_dir: str = "./qwen_residual_sae_cache/dense"
    shard_size: int = 32
    overwrite: bool = False
    resume: bool = False
    skip_mmhs150k: bool = False


@dataclass
class ResidualCacheConfig(Serializable):
    qwen: QwenResidualConfig = field(default_factory=QwenResidualConfig)
    dense_cache_dir: str = "./qwen_residual_sae_cache/dense"
    output_dir: str = "./qwen_residual_sae_cache/residual"
    overwrite: bool = False
    resume: bool = False


@dataclass
class QwenInferenceConfig(Serializable):
    dataset: MemeDatasetConfig = field(default_factory=lambda: MemeDatasetConfig(splits=["validation"]))
    qwen: QwenResidualConfig = field(default_factory=QwenResidualConfig)
    residual_sae_path: str = "./qwen_residual_sae_ckpts/best_finite"
    output_path: str = "./qwen_residual_sae_predictions.jsonl"
    max_samples: int | None = None
    apply_residual_sae: bool = True
    reconstruction_alpha: float = 1.0
    # Feature steering (optional — used by steering_sweep)
    steering_feature_idx: int | None = None
    steering_feature_indices: list[int] = list_field()
    steering_delta: float = 0.0
    steering_deltas: list[float] = list_field()
    steering_mode: Literal["add", "clamp"] = "add"
    binary_only: bool = False
    constrained_decode: bool = True
    decode_strategy: Literal["auto", "yes_no", "label_score"] = "auto"


@dataclass
class JumpReluSaeConfig(Serializable):
    expansion_factor: int = 8
    num_latents: int = 0
    normalize_decoder: bool = True
    threshold_init: float = 0.1
    threshold_bandwidth: float = 0.1
    l0_coefficient: float = 1e-4


@dataclass
class ResidualSaeTrainConfig(Serializable):
    residual_cache_dir: str = "./qwen_residual_sae_cache/residual"
    output_dir: str = "./qwen_residual_sae_ckpts"
    sae_architecture: Literal["topk", "jump_relu"] = "jump_relu"
    jump_relu: JumpReluSaeConfig = field(default_factory=JumpReluSaeConfig)
    device: str = "cuda:0"
    batch_size: int = 4096
    lr: float = 5e-5
    weight_decay: float = 0.0
    lr_warmup_steps: int = 100
    grad_clip_norm: float | None = 1.0
    auxk_alpha: float = 0.0
    dead_feature_threshold: int = 1_000_000
    multi_topk_loss_weight: float = 0.125
    max_epochs: int = 1
    max_steps: int | None = None
    seed: int = 42
    val_fraction: float = 0.05
    eval_every_steps: int = 250
    eval_max_batches: int | None = 64
    save_every_steps: int = 500
    log_every_steps: int = 50
    init_geometric_median_tokens: int = 8192
    normalize_residuals: bool = True
    normalization_eps: float = 1e-5