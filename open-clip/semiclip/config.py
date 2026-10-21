"""Validated configuration for the two-stage RSICD SemiCLIP workflow."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SemiCLIPConfig:
    model_name: str
    pretrained_tag: str
    checkpoint_path: Path
    image_dir: Path
    json_path: Path
    labeled_split: str
    unlabeled_split: str
    supervised_train_fraction: float
    seed: int
    total_batch_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float
    warmup_steps: int
    stage1_epochs: int
    stage2_epochs: int
    concept_min_frequency: int
    concept_max_frequency_ratio: float
    concept_loss_top_k: int
    surrogate_top_k: int
    selection_ratio: float
    clip_loss_weight: float
    concept_loss_weight: float
    intramodal_loss_weight: float
    crossmodal_loss_weight: float
    output_dir: Path

    @property
    def labeled_batch_size(self) -> int:
        return self.total_batch_size // 2

    @property
    def unlabeled_batch_size(self) -> int:
        return self.total_batch_size // 2


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return (config_path.parent / path).resolve() if not path.is_absolute() else path


def load_semiclip_config(config_path: str | Path) -> SemiCLIPConfig:
    """Load and strictly validate a SemiCLIP YAML configuration."""
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"SemiCLIP config does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        values = yaml.safe_load(config_file)
    if not isinstance(values, dict):
        raise ValueError(f"SemiCLIP config must contain a mapping: {config_path}")

    string_fields = (
        "model_name",
        "pretrained_tag",
        "checkpoint_path",
        "image_dir",
        "json_path",
        "labeled_split",
        "unlabeled_split",
        "output_dir",
    )
    missing = [name for name in string_fields if not isinstance(values.get(name), str)]
    if missing:
        raise ValueError(
            "SemiCLIP config is missing string values for: " + ", ".join(missing)
        )

    integer_fields = (
        "seed",
        "total_batch_size",
        "num_workers",
        "warmup_steps",
        "stage1_epochs",
        "stage2_epochs",
        "concept_min_frequency",
        "concept_loss_top_k",
        "surrogate_top_k",
    )
    float_fields = (
        "supervised_train_fraction",
        "learning_rate",
        "weight_decay",
        "beta1",
        "beta2",
        "epsilon",
        "concept_max_frequency_ratio",
        "selection_ratio",
        "clip_loss_weight",
        "concept_loss_weight",
        "intramodal_loss_weight",
        "crossmodal_loss_weight",
    )
    try:
        numbers = {name: int(values[name]) for name in integer_fields}
        numbers.update({name: float(values[name]) for name in float_fields})
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("SemiCLIP config has invalid numeric settings") from error

    if values["labeled_split"] == values["unlabeled_split"]:
        raise ValueError("labeled_split and unlabeled_split must be distinct")
    if "test" in {values["labeled_split"], values["unlabeled_split"]}:
        raise ValueError("The RSICD test split cannot be used for SemiCLIP training")
    if not 0 < numbers["supervised_train_fraction"] < 1:
        raise ValueError("supervised_train_fraction must be strictly between zero and one")
    if numbers["total_batch_size"] < 2 or numbers["total_batch_size"] % 2:
        raise ValueError("total_batch_size must be an even integer of at least two")
    if numbers["num_workers"] < 0 or numbers["warmup_steps"] < 0:
        raise ValueError("num_workers and warmup_steps cannot be negative")
    positive = (
        "learning_rate",
        "epsilon",
        "stage1_epochs",
        "stage2_epochs",
        "concept_min_frequency",
        "concept_loss_top_k",
        "surrogate_top_k",
    )
    if any(numbers[name] <= 0 for name in positive):
        raise ValueError("Learning, epoch, frequency, and top-k settings must be positive")
    if numbers["weight_decay"] < 0:
        raise ValueError("weight_decay cannot be negative")
    if not 0 < numbers["beta1"] < 1 or not 0 < numbers["beta2"] < 1:
        raise ValueError("AdamW beta values must be strictly between zero and one")
    for name in ("concept_max_frequency_ratio", "selection_ratio"):
        if not 0 < numbers[name] <= 1:
            raise ValueError(f"{name} must be strictly positive and at most one")
    for name in (
        "clip_loss_weight",
        "concept_loss_weight",
        "intramodal_loss_weight",
        "crossmodal_loss_weight",
    ):
        if numbers[name] < 0:
            raise ValueError(f"{name} cannot be negative")
    selected = int((numbers["total_batch_size"] // 2) * numbers["selection_ratio"])
    if selected < 1:
        raise ValueError("selection_ratio must retain at least one unlabeled sample per batch")

    return SemiCLIPConfig(
        model_name=values["model_name"],
        pretrained_tag=values["pretrained_tag"],
        checkpoint_path=_resolve(config_path, values["checkpoint_path"]),
        image_dir=_resolve(config_path, values["image_dir"]),
        json_path=_resolve(config_path, values["json_path"]),
        labeled_split=values["labeled_split"],
        unlabeled_split=values["unlabeled_split"],
        supervised_train_fraction=numbers["supervised_train_fraction"],
        seed=numbers["seed"],
        total_batch_size=numbers["total_batch_size"],
        num_workers=numbers["num_workers"],
        learning_rate=numbers["learning_rate"],
        weight_decay=numbers["weight_decay"],
        beta1=numbers["beta1"],
        beta2=numbers["beta2"],
        epsilon=numbers["epsilon"],
        warmup_steps=numbers["warmup_steps"],
        stage1_epochs=numbers["stage1_epochs"],
        stage2_epochs=numbers["stage2_epochs"],
        concept_min_frequency=numbers["concept_min_frequency"],
        concept_max_frequency_ratio=numbers["concept_max_frequency_ratio"],
        concept_loss_top_k=numbers["concept_loss_top_k"],
        surrogate_top_k=numbers["surrogate_top_k"],
        selection_ratio=numbers["selection_ratio"],
        clip_loss_weight=numbers["clip_loss_weight"],
        concept_loss_weight=numbers["concept_loss_weight"],
        intramodal_loss_weight=numbers["intramodal_loss_weight"],
        crossmodal_loss_weight=numbers["crossmodal_loss_weight"],
        output_dir=_resolve(config_path, values["output_dir"]),
    )
