from dataclasses import dataclass
from typing import Literal

from simple_parsing import Serializable, field, list_field

from gemma3_experiments.vendor_sae.config import SaeConfig

from .models import JumpReluSaeConfig


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
    # EXIST uses one labeled training file. Requested evaluation languages receive
    # deterministic, label-stratified holdouts; all other included rows train.
    exist_train_languages: list[str] = list_field()
    exist_eval_languages: list[str] = list_field()
    exist_eval_per_language: int = 500
    exist_eval_seed: int = 42
    exist_text_mode: Literal["raw", "translation"] = "raw"
    exist_translation_path: str = ""
    exist_translation_languages: list[str] = list_field("es")
    # MultiBully ships one XLSX annotation table without an official evaluation
    # split. Hold out a deterministic label-stratified subset from all valid rows.
    multibully_eval_size: int = 1000
    multibully_eval_seed: int = 42


@dataclass
class GemmaResidualConfig(Serializable):
    model_id: str = "google/gemma-3-4b-it"
    sae_repo_id: str = "google/gemma-scope-2-4b-it"
    sae_hookpoint: str = "resid_post/layer_22_width_65k_l0_medium"
    hook_module_path: str = "model.language_model.layers.22"
    model_device: str = "cuda:0"
    sae_device: str = "cuda:0"
    max_new_tokens: int = 96
    token_selection: Literal["all", "image", "generated", "image_and_generated"] = "all"
    capture_generation: bool = True
    sae_chunk_size: int = 128


@dataclass
class DenseActivationCacheConfig(Serializable):
    dataset: MemeDatasetConfig = field(default_factory=MemeDatasetConfig)
    gemma: GemmaResidualConfig = field(default_factory=GemmaResidualConfig)
    output_dir: str = "./residual_sae_cache/dense"
    shard_size: int = 32
    overwrite: bool = False
    resume: bool = False
    skip_mmhs150k: bool = False


@dataclass
class ResidualCacheConfig(Serializable):
    gemma: GemmaResidualConfig = field(default_factory=GemmaResidualConfig)
    dense_cache_dir: str = "./residual_sae_cache/dense"
    output_dir: str = "./residual_sae_cache/residual"
    overwrite: bool = False
    resume: bool = False


@dataclass
class ResidualSaeTrainConfig(Serializable):
    residual_cache_dir: str = "./residual_sae_cache/residual"
    output_dir: str = "./residual_sae_ckpts"
    sae_architecture: Literal["topk", "jump_relu"] = "topk"
    sae: SaeConfig = field(default_factory=lambda: SaeConfig(expansion_factor=8, k=16))
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


@dataclass
class ResidualInferenceConfig(Serializable):
    dataset: MemeDatasetConfig = field(default_factory=MemeDatasetConfig)
    gemma: GemmaResidualConfig = field(default_factory=GemmaResidualConfig)
    residual_sae_path: str = "./residual_sae_ckpts/final"
    output_path: str = "./residual_sae_predictions.jsonl"
    max_samples: int | None = None
    steering_feature_idx: int | None = None
    steering_feature_indices: list[int] = list_field()
    steering_delta: float = 0.0
    steering_deltas: list[float] = list_field()
    steering_values: list[float] = list_field()
    steering_mode: Literal["add", "clamp"] = "add"
    apply_residual_sae: bool = True
    reconstruction_alpha: float = 1.0
    binary_only: bool = False
    constrained_decode: bool = True
    decode_strategy: Literal["auto", "yes_no", "label_score", "hierarchical_label_score", "free_generate"] = "auto"


@dataclass
class ResidualFeatureDiscoveryConfig(Serializable):
    dataset: MemeDatasetConfig = field(default_factory=lambda: MemeDatasetConfig(splits=["train"]))
    residual_cache_dir: str = "./residual_sae_cache/residual"
    residual_sae_path: str = "./residual_sae_ckpts/final"
    output_dir: str = "./residual_feature_discovery"
    device: str = "cuda:0"
    token_selection: Literal["all", "image", "prompt", "generated", "image_and_generated"] = "image"
    aggregation: Literal["max", "mean"] = "max"
    batch_size: int = 512
    top_k_per_label: int = 100
    max_samples_per_dataset: int | None = None
    min_positive_samples: int = 5


@dataclass
class ResidualSteeringSweepConfig(Serializable):
    dataset: MemeDatasetConfig = field(default_factory=lambda: MemeDatasetConfig(splits=["validation"]))
    gemma: GemmaResidualConfig = field(default_factory=GemmaResidualConfig)
    residual_sae_path: str = "./residual_sae_ckpts/final"
    candidates_path: str = "./residual_feature_discovery/steering_candidates.jsonl"
    baseline_predictions_path: str | None = None
    output_dir: str = "./residual_steering_sweep"
    max_candidates: int | None = 20
    candidate_ids: list[str] = list_field()
    # Filter by a single label name (e.g. "misogynous") and/or direction ("label_up"/"label_down").
    # Avoids shell quoting issues when label names contain spaces.
    candidate_label: str | None = None
    candidate_direction: str | None = None
    steering_delta_values: list[float] = list_field(-2.0, -1.0, 1.0, 2.0)
    steering_mode: Literal["add", "clamp"] = "add"
    reconstruction_alpha: float = 0.10
    binary_only: bool = False
    max_samples: int | None = None
    constrained_decode: bool = True
    decode_strategy: Literal["auto", "yes_no", "label_score", "free_generate"] = "auto"
