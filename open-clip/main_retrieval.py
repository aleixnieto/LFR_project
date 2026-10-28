"""Evaluate configurable image-to-text retrieval Accuracy@k on RSICD."""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import open_clip
import torch
import torch.nn.functional as functional
import yaml
from torch.utils.data import DataLoader

from dataloader import RSICDSupervisedDataset, collate_supervised_captions

from main_zeroshot import (
    apply_model_overrides,
    load_dataset_config,
    load_model_config,
    require_checkpoint,
    resolve_device,
)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("configs") / "retrieval_config.yaml"
DEFAULT_OUTPUT_DIR = Path("evaluations/open_clip_retrieval")


@dataclass(frozen=True)
class RetrievalConfig:
    num_captions: int = 10
    log_every: int = 50
    ks: tuple[int, ...] = (1, 3, 5, 10)


def load_retrieval_config(config_path: Path) -> RetrievalConfig:
    with config_path.open(encoding="utf-8") as config_file:
        values = yaml.safe_load(config_file)
    if not isinstance(values, dict):
        raise ValueError(f"Retrieval config must contain a mapping: {config_path}")
    defaults = RetrievalConfig()
    num_captions = values.get("num_captions", defaults.num_captions)
    log_every = values.get("log_every", defaults.log_every)
    ks = values.get("ks", list(defaults.ks))
    for name, value in (("num_captions", num_captions), ("log_every", log_every)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        not isinstance(ks, list)
        or not ks
        or any(type(k) is not int or k <= 0 for k in ks)
    ):
        raise ValueError("ks must be a non-empty list of positive integers")
    if len(set(ks)) != len(ks):
        raise ValueError("ks must contain distinct values")
    return RetrievalConfig(num_captions=num_captions, log_every=log_every, ks=tuple(ks))


def configure_logger(output_dir: Path, log_filename: str = "retrieval.log") -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("open_clip_retrieval")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(levelname)s %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(output_dir / log_filename, encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def build_caption_features(
    captions: list[str],
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """Return one unit-length embedding per caption, shaped [num_captions, dim]."""
    if not captions:
        raise ValueError("At least one caption is required")
    text_tokens = tokenizer(captions).to(device)
    with torch.no_grad():
        caption_features = model.encode_text(text_tokens)
    return functional.normalize(caption_features, dim=-1)


def print_retrieval_examples(
    predictions: list[dict[str, object]],
    captions_by_image: dict[str, list[str]],
    logger: logging.Logger,
    num_captions: int = 10,
    log_every: int = 50,
) -> None:
    """Log up to num_captions ranked captions for every log_every-th image."""
    if num_captions <= 0:
        raise ValueError("num_captions must be positive")
    if log_every <= 0:
        raise ValueError("log_every must be positive")
    for image_number, prediction in enumerate(predictions, start=1):
        if image_number % log_every != 0:
            continue
        filename = prediction["filename"]
        top_captions = prediction["top_captions"]
        top_similarities = prediction["top_similarities"]
        logger.info("Image %d: %s", image_number, filename)
        for rank, (caption, score) in enumerate(
            zip(top_captions[:num_captions], top_similarities[:num_captions]), start=1
        ):
            status = "✅" if caption in captions_by_image[filename] else "❌"
            logger.info("  %d. [%s] %r (similarity: %.4f)", rank, status, caption, score)


@torch.no_grad()
def evaluate(
    data_loader: DataLoader,
    model: torch.nn.Module,
    caption_features: torch.Tensor,
    captions: list[str],
    captions_by_image: dict[str, list[str]],
    device: torch.device,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    num_captions: int = 10,
) -> list[dict[str, object]]:
    """Check whether the retrieved text is one of the query image's captions."""
    if not captions or len(captions) != caption_features.shape[0]:
        raise ValueError("Each caption embedding must have caption text")
    if not ks or any(type(k) is not int or k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")
    if type(num_captions) is not int or num_captions <= 0:
        raise ValueError("num_captions must be a positive integer")
    retrieval_depth = max(max(ks), num_captions)
    model.eval()
    predictions: list[dict[str, object]] = []
    for batch in data_loader:
        image_features = functional.normalize(
            model.encode_image(batch["images"].to(device)), dim=-1
        )
        similarities = image_features @ caption_features.T
        top_indices = similarities.argsort(dim=1, descending=True, stable=True)[:, :retrieval_depth]
        top_scores = similarities.gather(1, top_indices)
        for filename, caption_indices, scores in zip(
            batch["filenames"], top_indices.tolist(), top_scores.tolist()
        ):
            retrieved_captions = [captions[index] for index in caption_indices]
            matches = [
                caption in captions_by_image[filename]
                for caption in retrieved_captions
            ]
            predictions.append({
                "filename": filename,
                "caption_index": caption_indices[0],
                "caption": retrieved_captions[0],
                "similarity": scores[0],
                "correct": matches[0],
                "top_captions": retrieved_captions,
                "top_similarities": scores,
                **{f"correct_at_{k}": any(matches[:k]) for k in ks},
            })
    if not predictions:
        raise ValueError("No image queries were evaluated")
    return predictions



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML model/checkpoint configuration (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="Override the config local checkpoint_path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Evaluate only the first N test images; omit to evaluate the full test split.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Evaluation device (default: auto).",
    )
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    return args


def main() -> None:
    args = parse_args()
    retrieval_config = load_retrieval_config(args.config)
    dataset_config = load_dataset_config(args.config)
    model_config = apply_model_overrides(args, load_model_config(args.config))
    checkpoint_path = require_checkpoint(model_config)
    pretrained_config = open_clip.get_pretrained_cfg(
        model_config.model_name, model_config.pretrained_tag
    )
    if pretrained_config is None:
        raise RuntimeError(
            f"No preprocessing configuration for "
            f"{model_config.model_name} / {model_config.pretrained_tag}."
        )
    device = resolve_device(args.device)
    logger = configure_logger(args.output_dir.resolve())
    logger.info("Device: %s", device)

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_config.model_name,
        pretrained=str(checkpoint_path),
        device=device,
        image_mean=pretrained_config["mean"],
        image_std=pretrained_config["std"],
        image_interpolation=pretrained_config["interpolation"],
        image_resize_mode=pretrained_config["resize_mode"],
    )
    tokenizer = open_clip.get_tokenizer(model_config.model_name)
    model.eval()

    with dataset_config.json_path.open(encoding="utf-8") as metadata_file:
        records = json.load(metadata_file)["images"]
    records = [
        record for record in records
        if record.get("split") == dataset_config.dataset_split
    ]
    if args.max_samples is not None:
        records = records[:args.max_samples]
    dataset = RSICDSupervisedDataset(
        image_dir=dataset_config.image_dir,
        records=records,
        transform=preprocess,
        tokenizer=tokenizer,
        caption_policy="all_unique",
    )
    captions = list(dict.fromkeys(
        caption for image_captions in dataset.captions for caption in image_captions
    ))
    captions_by_image = dict(zip(dataset.filenames, dataset.captions))

    caption_features = torch.cat([
        build_caption_features(
            captions[start:start + args.batch_size],
            model, tokenizer, device,
        )
        for start in range(0, len(captions), args.batch_size)
    ])
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_supervised_captions,
    )
    predictions = evaluate(
        data_loader, model, caption_features, captions, captions_by_image, device,
        ks=retrieval_config.ks, num_captions=retrieval_config.num_captions,
    )
    print_retrieval_examples(
        predictions, captions_by_image, logger,
        num_captions=retrieval_config.num_captions,
        log_every=retrieval_config.log_every,
    )

    logger.info("Model: %s / %s", model_config.model_name, model_config.pretrained_tag)
    logger.info("Checkpoint: %s", model_config.checkpoint_path)
    logger.info("Images: %s", dataset_config.image_dir)
    logger.info("Metadata: %s", dataset_config.json_path)
    logger.info("Dataset split: %s", dataset_config.dataset_split)
    logger.info("Loaded image records: %d", len(dataset))
    logger.info("Candidate captions: %d", len(captions))
    for k in retrieval_config.ks:
        correct = sum(prediction[f"correct_at_{k}"] for prediction in predictions)
        logger.info(
            "Accuracy@%d: %.4f (%d/%d)",
            k,
            correct / len(predictions),
            correct,
            len(predictions),
        )
    if args.max_samples is not None:
        logger.info("Subset evaluation: the caption pool is limited to the selected images.")
    logger.info("Log: %s", args.output_dir.resolve() / "retrieval.log")



if __name__ == "__main__":
    main()
