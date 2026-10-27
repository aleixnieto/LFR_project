"""Evaluate image-to-text retrieval Accuracy@1, @3, and @5 on RSICD."""

import argparse
import json
from pathlib import Path
import open_clip
import torch
import torch.nn.functional as functional
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


@torch.no_grad()
def evaluate(
    data_loader: DataLoader,
    model: torch.nn.Module,
    caption_features: torch.Tensor,
    captions: list[str],
    captions_by_image: dict[str, list[str]],
    device: torch.device,
) -> list[dict[str, object]]:
    """Check whether the retrieved text is one of the query image's captions."""
    if not captions or len(captions) != caption_features.shape[0]:
        raise ValueError("Each caption embedding must have caption text")
    model.eval()
    predictions: list[dict[str, object]] = []
    for batch in data_loader:
        image_features = functional.normalize(
            model.encode_image(batch["images"].to(device)), dim=-1
        )
        similarities = image_features @ caption_features.T
        top_indices = similarities.argsort(dim=1, descending=True, stable=True)[:, :5]
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
                "correct_at_1": any(matches[:1]),
                "correct_at_3": any(matches[:3]),
                "correct_at_5": any(matches[:5]),
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
    parser.add_argument("--model-name", help="Override the config model_name.")
    parser.add_argument(
        "--pretrained",
        help="Override the config pretrained_tag; weights remain local.",
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
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Evaluation device (default: auto).",
    )
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
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
    captions = []
    for image_captions in dataset.captions:
        for caption in image_captions:
            captions.append(caption)
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
        data_loader, model, caption_features, captions, captions_by_image, device
    )

    print(f"Model: {model_config.model_name} / {model_config.pretrained_tag}")
    print(f"Checkpoint: {model_config.checkpoint_path}")
    print(f"Images: {dataset_config.image_dir}")
    print(f"Metadata: {dataset_config.json_path}")
    print(f"Dataset split: {dataset_config.dataset_split}")
    print(f"Loaded image records: {len(dataset)}")
    print(f"Candidate captions: {len(captions)}")
    for k in (1, 3, 5):
        correct = sum(prediction[f"correct_at_{k}"] for prediction in predictions)
        print(f"Accuracy@{k}: {correct / len(predictions):.4f} ({correct}/{len(predictions)})")
    if args.max_samples is not None:
        print("Subset evaluation: the caption pool is limited to the selected images.")
    # for prediction in predictions:
    #     print(
    #         f"{prediction['filename']} -> {prediction['caption']!r} "
    #         f"(correct: {prediction['correct']}, similarity: {prediction['similarity']:.4f})"
    #     )


if __name__ == "__main__":
    main()
