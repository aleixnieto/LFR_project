"""Compare checkpoints on the explicit filename subset declared in their YAML."""

import argparse
from pathlib import Path

import yaml

from main_zeroshot_compare_subset import load_filenames, main as _unused, subset_indices
from main_zeroshot_compare_subset import (  # Reuse the subset-safe evaluator pieces.
    DataLoader,
    RSICDZeroShotDataset,
    build_text_features,
    calculate_comparison_metrics,
    evaluate_with_progress,
    json,
    load_comparison_dataset,
    load_model,
    load_models,
    open_clip,
    resolve_device,
    select_models,
    torch,
    write_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-config", type=Path, required=True)
    parser.add_argument("--model", action="append")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def load_subset_source(config_path: Path) -> tuple[Path, str]:
    config_path = config_path.resolve()
    with config_path.open(encoding="utf-8") as file:
        values = yaml.safe_load(file)
    if not isinstance(values, dict):
        raise ValueError(f"Comparison config must contain a mapping: {config_path}")
    source = values.get("subset_filenames_json")
    key = values.get("subset_filenames_key", "labeled_validation_filenames")
    if not isinstance(source, str) or not source:
        raise ValueError("Comparison config requires subset_filenames_json")
    if not isinstance(key, str) or not key:
        raise ValueError("subset_filenames_key must be a non-empty string")
    path = Path(source)
    return ((config_path.parent / path).resolve() if not path.is_absolute() else path, key)


def main() -> None:
    args = parse_args()
    subset_path, subset_key = load_subset_source(args.models_config)
    filenames = load_filenames(subset_path, subset_key)
    dataset_config = load_comparison_dataset(args.models_config)
    selections = select_models(load_models(args.models_config), args.model)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output_dir}")
    output_dir.mkdir(parents=True)
    device = resolve_device(args.device)
    summaries: list[dict[str, object]] = []

    for selection in selections:
        model, preprocess = load_model(selection, device)
        dataset = RSICDZeroShotDataset(
            dataset_config.image_dir, dataset_config.json_path, dataset_config.class_dir,
            preprocess, dataset_config.dataset_split,
        )
        loader = DataLoader(
            __import__("torch").utils.data.Subset(dataset, subset_indices(dataset, filenames)),
            batch_size=args.batch_size, shuffle=False,
        )
        text_features = build_text_features(
            dataset.class_names, model, open_clip.get_tokenizer(selection.model_name), device
        )
        ground_truths, predictions = evaluate_with_progress(
            loader, dataset.class_names, model, text_features, device, selection.label
        )
        metric_values, _ = calculate_comparison_metrics(
            ground_truths, [item["prediction"] for item in predictions], dataset.class_names
        )
        metrics = {
            "label": selection.label, "checkpoint_path": str(selection.checkpoint_path),
            "checkpoint_type": selection.checkpoint_type, "num_samples": len(predictions),
            "num_classes": len(dataset.class_names), **metric_values,
        }
        write_results(output_dir / selection.label, metrics, predictions)
        summaries.append({key: metrics[key] for key in (
            "label", "checkpoint_path", "checkpoint_type", "num_samples", "accuracy",
            "weighted_precision", "weighted_recall", "weighted_f1",
        )})
        print(f"{selection.label}: accuracy={metrics['accuracy']:.4f} weighted_f1={metrics['weighted_f1']:.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump({"models": summaries}, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
