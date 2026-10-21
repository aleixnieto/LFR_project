"""Run the two-stage SemiCLIP workflow on the isolated RSICD splits."""

import argparse
import json
import random
import shutil
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path


OPEN_CLIP_DIR = Path(__file__).resolve().parents[1]
if str(OPEN_CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(OPEN_CLIP_DIR))

import numpy as np
import open_clip
import torch
from torch.utils.data import DataLoader

from dataloader import collate_supervised_captions, split_rsicd_records
from semiclip.concepts import (
    ConceptMiningResult,
    hash_filenames,
    mine_semantic_concepts,
    require_nltk_resources,
)
from semiclip.config import SemiCLIPConfig, load_semiclip_config
from semiclip.data import (
    SemiCLIPDatasets,
    build_semiclip_datasets,
    collate_concept_labeled,
    collate_unlabeled_images,
    collate_unlabeled_views,
)
from semiclip.model import initialize_concept_classifier
from semiclip.stage1 import fit_stage1
from semiclip.stage2 import fit_stage2, prepare_stage2


DEFAULT_CONFIG_PATH = Path(__file__).with_name("configs") / "semiclip_rsicd.yaml"
CONCEPT_ARTIFACT_NAME = "concepts.json"


@dataclass(frozen=True)
class SemiCLIPLoaders:
    labeled_stage1: DataLoader
    labeled_stage2: DataLoader
    labeled_validation: DataLoader
    unlabeled_stage1: DataLoader
    unlabeled_stage2: DataLoader


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("1", "2", "all"), default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-stage2", type=Path)
    parser.add_argument("--stage1-epochs", type=int)
    parser.add_argument("--stage2-epochs", type=int)
    parser.add_argument("--max-stage1-train-batches", type=int)
    parser.add_argument("--max-stage2-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--max-unlabeled-samples", type=int)
    args = parser.parse_args(argv)
    positive = (
        "stage1_epochs",
        "stage2_epochs",
        "max_stage1_train_batches",
        "max_stage2_train_batches",
        "max_validation_batches",
        "max_unlabeled_samples",
    )
    for name in positive:
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.resume_stage2 is not None and args.stage != "2":
        parser.error("--resume-stage2 requires --stage 2")
    return args


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {name}")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device {name!r}; use auto, cpu, or cuda")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _serialized_config(config: SemiCLIPConfig) -> dict[str, object]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in asdict(config).items()
    }


def apply_cli_overrides(
    config: SemiCLIPConfig, args: argparse.Namespace
) -> SemiCLIPConfig:
    updates = {}
    if args.output_dir is not None:
        updates["output_dir"] = args.output_dir.resolve()
    if args.stage1_epochs is not None:
        updates["stage1_epochs"] = args.stage1_epochs
    if args.stage2_epochs is not None:
        updates["stage2_epochs"] = args.stage2_epochs
    return replace(config, **updates)


def load_openclip_components(config: SemiCLIPConfig, device: torch.device):
    """Load only the configured local checkpoint and its matching transforms."""
    if not config.checkpoint_path.is_file():
        raise FileNotFoundError(
            f"OpenCLIP checkpoint is missing: {config.checkpoint_path}. "
            "Download it manually and update checkpoint_path."
        )
    preprocessing = open_clip.get_pretrained_cfg(
        config.model_name, config.pretrained_tag
    )
    if preprocessing is None:
        raise RuntimeError(
            "Installed OpenCLIP has no preprocessing configuration for "
            f"{config.model_name} / {config.pretrained_tag}"
        )
    model, weak_transform, validation_transform = open_clip.create_model_and_transforms(
        config.model_name,
        pretrained=str(config.checkpoint_path),
        device=device,
        image_mean=preprocessing["mean"],
        image_std=preprocessing["std"],
        image_interpolation=preprocessing["interpolation"],
        image_resize_mode=preprocessing["resize_mode"],
    )
    return (
        model,
        open_clip.get_tokenizer(config.model_name),
        weak_transform,
        validation_transform,
    )


def write_workflow_concepts(
    path: Path,
    result: ConceptMiningResult,
    validation_records: list[dict],
    config: SemiCLIPConfig,
) -> None:
    artifact = result.to_json_dict()
    validation_filenames = [record["filename"] for record in validation_records]
    artifact.update(
        {
            "labeled_training_filename_sha256": hash_filenames(
                result.source_filenames
            ),
            "labeled_validation_filenames": validation_filenames,
            "labeled_validation_filename_sha256": hash_filenames(
                validation_filenames
            ),
            "config": _serialized_config(config),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(artifact, file, indent=2)
        file.write("\n")


def load_workflow_concepts(
    path: Path,
    training_records: list[dict],
    validation_records: list[dict],
) -> ConceptMiningResult:
    if not path.is_file():
        raise FileNotFoundError(
            f"SemiCLIP concept artifact is missing: {path}. Run --stage 1 first."
        )
    with path.open(encoding="utf-8") as file:
        artifact = json.load(file)
    training_filenames = tuple(record["filename"] for record in training_records)
    validation_filenames = [record["filename"] for record in validation_records]
    if artifact.get("source_filenames") != list(training_filenames):
        raise ValueError("Concept artifact labeled-training filenames do not match")
    if artifact.get("source_filename_sha256") != hash_filenames(training_filenames):
        raise ValueError("Concept artifact labeled-training hash does not match")
    if artifact.get("labeled_validation_filenames") != validation_filenames:
        raise ValueError("Concept artifact labeled-validation filenames do not match")
    if artifact.get("labeled_validation_filename_sha256") != hash_filenames(
        validation_filenames
    ):
        raise ValueError("Concept artifact labeled-validation hash does not match")
    concepts = artifact.get("concepts")
    counts = artifact.get("counts")
    raw_labels = artifact.get("labels_by_filename")
    if not isinstance(concepts, list) or not concepts:
        raise ValueError("Concept artifact has no vocabulary")
    if not isinstance(counts, dict) or not isinstance(raw_labels, dict):
        raise ValueError("Concept artifact has invalid counts or labels")
    labels = {
        filename: tuple(int(index) for index in indices)
        for filename, indices in raw_labels.items()
    }
    if set(labels) != set(training_filenames):
        raise ValueError("Concept artifact labels do not cover labeled training images")
    return ConceptMiningResult(
        concepts=tuple(str(concept) for concept in concepts),
        counts={str(name): int(count) for name, count in counts.items()},
        labels_by_filename=labels,
        source_filenames=training_filenames,
    )


def build_loaders(
    datasets: SemiCLIPDatasets,
    config: SemiCLIPConfig,
    device: torch.device,
) -> SemiCLIPLoaders:
    common = {
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
    }

    def generator(offset: int) -> torch.Generator:
        return torch.Generator().manual_seed(config.seed + offset)

    labeled_stage1 = DataLoader(
        datasets.labeled_train,
        batch_size=config.labeled_batch_size,
        shuffle=True,
        generator=generator(1),
        drop_last=True,
        collate_fn=collate_concept_labeled,
        **common,
    )
    labeled_stage2 = DataLoader(
        datasets.labeled_train,
        batch_size=config.labeled_batch_size,
        shuffle=True,
        generator=generator(3),
        drop_last=True,
        collate_fn=collate_concept_labeled,
        **common,
    )
    labeled_validation = DataLoader(
        datasets.labeled_validation,
        batch_size=config.total_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_supervised_captions,
        **common,
    )
    unlabeled_stage1 = DataLoader(
        datasets.unlabeled_stage1,
        batch_size=config.unlabeled_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_unlabeled_images,
        **common,
    )
    unlabeled_stage2 = DataLoader(
        datasets.unlabeled_stage2,
        batch_size=config.unlabeled_batch_size,
        shuffle=True,
        generator=generator(2),
        drop_last=True,
        collate_fn=collate_unlabeled_views,
        **common,
    )
    if not labeled_stage1 or not labeled_stage2 or not labeled_validation:
        raise ValueError("Configured batch sizes leave an empty labeled loader")
    if not unlabeled_stage1 or not unlabeled_stage2:
        raise ValueError("Configured batch sizes leave an empty unlabeled loader")
    return SemiCLIPLoaders(
        labeled_stage1,
        labeled_stage2,
        labeled_validation,
        unlabeled_stage1,
        unlabeled_stage2,
    )


def prepare_output(
    output_dir: Path,
    config: SemiCLIPConfig,
    source_config_path: Path | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if source_config_path is not None:
        source = source_config_path.resolve()
        destination = (output_dir / "config.yaml").resolve()
        if source != destination:
            shutil.copy2(source, destination)
    with (output_dir / "resolved_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(_serialized_config(config), file, indent=2)
        file.write("\n")


def run_semiclip(
    args: argparse.Namespace,
    config: SemiCLIPConfig,
    source_config_path: Path | None = None,
) -> None:
    config = apply_cli_overrides(config, args)
    device = resolve_device(args.device)
    set_seed(config.seed)
    prepare_output(config.output_dir, config, source_config_path)
    model, tokenizer, weak_transform, validation_transform = (
        load_openclip_components(config, device)
    )
    training_records, validation_records = split_rsicd_records(
        config.json_path,
        config.labeled_split,
        config.supervised_train_fraction,
        config.seed,
    )
    concept_path = config.output_dir / CONCEPT_ARTIFACT_NAME
    if args.stage in {"1", "all"}:
        require_nltk_resources()
        concepts = mine_semantic_concepts(
            training_records,
            min_frequency=config.concept_min_frequency,
            max_frequency_ratio=config.concept_max_frequency_ratio,
        )
        write_workflow_concepts(
            concept_path, concepts, validation_records, config
        )
    else:
        concepts = load_workflow_concepts(
            concept_path, training_records, validation_records
        )

    datasets = build_semiclip_datasets(
        config,
        weak_transform,
        validation_transform,
        tokenizer,
        concepts.labels_by_filename,
        len(concepts.concepts),
        max_unlabeled_samples=args.max_unlabeled_samples,
    )
    loaders = build_loaders(datasets, config, device)
    classifier = initialize_concept_classifier(
        model, tokenizer, concepts.concepts, device
    )

    if args.stage in {"1", "all"}:
        fit_stage1(
            model,
            classifier,
            loaders.labeled_stage1,
            loaders.labeled_validation,
            loaders.unlabeled_stage1,
            device,
            config,
            concepts.concepts,
            config.output_dir,
            max_train_batches=args.max_stage1_train_batches,
            max_validation_batches=args.max_validation_batches,
        )

    if args.stage in {"2", "all"}:
        stage1_checkpoint = config.output_dir / "stage1_final.pt"
        stage1_cache = config.output_dir / "stage1_unlabeled_logits.pt"
        prepared = prepare_stage2(
            model,
            classifier,
            tokenizer,
            config,
            stage1_checkpoint,
            stage1_cache,
            device,
        )
        fit_stage2(
            model,
            classifier,
            prepared.prompt,
            loaders.labeled_stage2,
            loaders.unlabeled_stage2,
            loaders.labeled_validation,
            prepared.cached_logits,
            device,
            config,
            concepts.concepts,
            prepared.stage1_provenance,
            config.output_dir,
            max_train_batches=args.max_stage2_train_batches,
            max_validation_batches=args.max_validation_batches,
            resume_path=args.resume_stage2,
        )


def main() -> None:
    args = parse_args()
    config = load_semiclip_config(args.config)
    run_semiclip(args, config, args.config)


if __name__ == "__main__":
    main()
