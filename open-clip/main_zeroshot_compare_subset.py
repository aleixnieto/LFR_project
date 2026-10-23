"""Compare configured checkpoints on an explicit RSICD filename subset."""

import argparse
import json
from pathlib import Path

import open_clip
import torch
from torch.utils.data import DataLoader, Subset

from dataloader import RSICDZeroShotDataset
from main_zeroshot import build_text_features, write_results
from main_zeroshot_compare import (
    calculate_comparison_metrics,
    evaluate_with_progress,
    load_comparison_dataset,
    load_model,
    load_models,
    resolve_device,
    select_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-config", type=Path, required=True)
    parser.add_argument("--subset-json", type=Path, required=True)
    parser.add_argument("--subset-key", default="labeled_validation_filenames")
    parser.add_argument("--model", action="append")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def load_filenames(path: Path, key: str) -> list[str]:
    with path.open(encoding="utf-8") as file:
        artifact = json.load(file)
    filenames = artifact.get(key) if isinstance(artifact, dict) else None
    if not isinstance(filenames, list) or not filenames or not all(
        isinstance(filename, str) and filename for filename in filenames
    ):
        raise ValueError(f"{path} must contain a non-empty string list at {key!r}")
    if len(set(filenames)) != len(filenames):
        raise ValueError(f"{path} contains duplicate filenames at {key!r}")
    return filenames


def subset_indices(dataset: RSICDZeroShotDataset, filenames: list[str]) -> list[int]:
    by_filename = {
        record["filename"]: index for index, record in enumerate(dataset.images_info)
    }
    missing = [filename for filename in filenames if filename not in by_filename]
    if missing:
        raise ValueError(
            "Subset contains filenames outside the configured dataset split: "
            f"{missing[:3]}"
        )
    return [by_filename[filename] for filename in filenames]


def main() -> None:
    args = parse_args()
    dataset_config = load_comparison_dataset(args.models_config)
    selections = select_models(load_models(args.models_config), args.model)
    filenames = load_filenames(args.subset_json, args.subset_key)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output_dir}")
    output_dir.mkdir(parents=True)
    device = resolve_device(args.device)
    summaries: list[dict[str, object]] = []

    for selection in selections:
        model, preprocess = load_model(selection, device)
        dataset = RSICDZeroShotDataset(
            image_dir=dataset_config.image_dir,
            json_path=dataset_config.json_path,
            class_dir=dataset_config.class_dir,
            transform=preprocess,
            split=dataset_config.dataset_split,
        )
        evaluation_dataset = Subset(dataset, subset_indices(dataset, filenames))
        loader = DataLoader(evaluation_dataset, batch_size=args.batch_size, shuffle=False)
        tokenizer = open_clip.get_tokenizer(selection.model_name)
        text_features = build_text_features(dataset.class_names, model, tokenizer, device)
        ground_truths, predictions = evaluate_with_progress(
            loader, dataset.class_names, model, text_features, device, selection.label
        )
        metric_values, _ = calculate_comparison_metrics(
            ground_truths,
            [record["prediction"] for record in predictions],
            dataset.class_names,
        )
        metrics = {
            "label": selection.label,
            "checkpoint_path": str(selection.checkpoint_path),
            "checkpoint_type": selection.checkpoint_type,
            "num_samples": len(predictions),
            "num_classes": len(dataset.class_names),
            **metric_values,
        }
        write_results(output_dir / selection.label, metrics, predictions)
        summaries.append(
            {
                key: metrics[key]
                for key in (
                    "label",
                    "checkpoint_path",
                    "checkpoint_type",
                    "num_samples",
                    "accuracy",
                    "weighted_precision",
                    "weighted_recall",
                    "weighted_f1",
                )
            }
        )
        print(
            f"{selection.label}: accuracy={metrics['accuracy']:.4f} "
            f"weighted_f1={metrics['weighted_f1']:.4f}"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump({"models": summaries}, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
