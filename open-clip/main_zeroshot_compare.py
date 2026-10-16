"""Compare explicitly configured baseline and fine-tuned OpenCLIP models on RSICD."""

import argparse
import json
import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from main_zeroshot import (
    DatasetConfig,
    build_text_features,
    resolve_device,
    write_results,
)


DEFAULT_MODELS_CONFIG_PATH = Path(__file__).with_name("zeroshot_compare_config.yaml")
DEFAULT_OUTPUT_DIR = Path("evaluations/open_clip_zeroshot_comparison")


@dataclass(frozen=True)
class ComparisonModel:
    label: str
    model_name: str
    pretrained_tag: str
    checkpoint_path: Path
    checkpoint_type: str


def resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_comparison_dataset(models_config_path: Path) -> DatasetConfig:
    models_config_path = models_config_path.resolve()
    with models_config_path.open(encoding="utf-8") as config_file:
        values = yaml.safe_load(config_file)
    if not isinstance(values, dict):
        raise ValueError(f"Comparison config must contain a mapping: {models_config_path}")

    required_keys = ("image_dir", "json_path", "class_dir")
    missing = [key for key in required_keys if not isinstance(values.get(key), str)]
    if missing:
        raise ValueError(f"Comparison config is missing string values for: {', '.join(missing)}")
    return DatasetConfig(
        image_dir=resolve_path(models_config_path, values["image_dir"]),
        json_path=resolve_path(models_config_path, values["json_path"]),
        class_dir=resolve_path(models_config_path, values["class_dir"]),
    )


def load_models(models_config_path: Path) -> list[ComparisonModel]:
    models_config_path = models_config_path.resolve()
    with models_config_path.open(encoding="utf-8") as config_file:
        values = yaml.safe_load(config_file)
    entries = values.get("models") if isinstance(values, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{models_config_path} must contain a non-empty models list")

    models: list[ComparisonModel] = []
    labels: set[str] = set()
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Model entry {index} must be a mapping")
        label = entry.get("label")
        checkpoint_path = entry.get("checkpoint_path")
        checkpoint_type = entry.get("checkpoint_type")
        if not all(isinstance(value, str) and value for value in (label, checkpoint_path, checkpoint_type, entry.get("model_name"), entry.get("pretrained_tag"))):
            raise ValueError(f"Model entry {index} needs label, model_name, pretrained_tag, checkpoint_path, and checkpoint_type strings")
        if label in labels:
            raise ValueError(f"Duplicate model label: {label}")
        if checkpoint_type not in {"openclip", "finetune"}:
            raise ValueError(f"Model {label} has checkpoint_type {checkpoint_type!r}; use openclip or finetune")
        model_name = entry["model_name"]
        pretrained_tag = entry["pretrained_tag"]
        models.append(
            ComparisonModel(
                label=label,
                model_name=model_name,
                pretrained_tag=pretrained_tag,
                checkpoint_path=resolve_path(models_config_path, checkpoint_path),
                checkpoint_type=checkpoint_type,
            )
        )
        labels.add(label)
    return models


def select_models(models: list[ComparisonModel], labels: list[str] | None) -> list[ComparisonModel]:
    if labels is None:
        return models
    by_label = {model.label: model for model in models}
    missing = [label for label in labels if label not in by_label]
    if missing:
        raise ValueError(f"Unknown configured model label(s): {', '.join(missing)}")
    return [by_label[label] for label in labels]


def safe_name(label: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._")
    if not name:
        raise ValueError("Model labels must contain at least one letter or number")
    return name


def load_model(selection: ComparisonModel, device: torch.device):
    if not selection.checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint for {selection.label} does not exist: {selection.checkpoint_path}")
    pretrained_config = open_clip.get_pretrained_cfg(selection.model_name, selection.pretrained_tag)
    if pretrained_config is None:
        raise ValueError(f"No preprocessing configuration for {selection.model_name} / {selection.pretrained_tag}")

    creation_kwargs = {
        "device": device,
        "image_mean": pretrained_config["mean"],
        "image_std": pretrained_config["std"],
        "image_interpolation": pretrained_config["interpolation"],
        "image_resize_mode": pretrained_config["resize_mode"],
    }
    if selection.checkpoint_type == "openclip":
        model, _, preprocess = open_clip.create_model_and_transforms(
            selection.model_name, pretrained=str(selection.checkpoint_path), **creation_kwargs
        )
    else:
        previous_disable_level = logging.root.manager.disable
        logging.disable(logging.WARNING)
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                selection.model_name, pretrained=None, **creation_kwargs
            )
        finally:
            logging.disable(previous_disable_level)
        state = torch.load(selection.checkpoint_path, map_location=device, weights_only=False)
        if not isinstance(state, dict) or "model_state_dict" not in state:
            raise ValueError(f"Fine-tuning checkpoint lacks model_state_dict: {selection.checkpoint_path}")
        model.load_state_dict(state["model_state_dict"])
    return model.eval(), preprocess


def calculate_comparison_metrics(
    ground_truths: list[str], predictions: list[str], class_names: list[str]
) -> tuple[dict[str, object], list[list[int]]]:
    precision, recall, f1_scores, support = precision_recall_fscore_support(
        ground_truths, predictions, labels=class_names, zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        ground_truths, predictions, average="weighted", zero_division=0
    )
    matrix = confusion_matrix(ground_truths, predictions, labels=class_names)
    accuracy = sum(expected == predicted for expected, predicted in zip(ground_truths, predictions))
    return (
        {
            "accuracy": accuracy / len(ground_truths),
            "weighted_precision": float(weighted_precision),
            "weighted_recall": float(weighted_recall),
            "weighted_f1": float(weighted_f1),
            "per_class": {
                name: {
                    "precision": float(class_precision),
                    "recall": float(class_recall),
                    "f1": float(class_f1),
                    "support": int(class_support),
                }
                for name, class_precision, class_recall, class_f1, class_support in zip(
                    class_names, precision, recall, f1_scores, support
                )
            },
        },
        matrix.tolist(),
    )


def save_confusion_matrix(matrix: list[list[int]], class_names: list[str], output_path: Path, label: str) -> None:
    figure, axis = plt.subplots(figsize=(14, 12))
    display = ConfusionMatrixDisplay(confusion_matrix=np.asarray(matrix), display_labels=class_names)
    display.plot(ax=axis, cmap="Blues", colorbar=False)
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    plt.setp(axis.get_yticklabels(), fontsize=9)
    axis.set_title(f"Confusion matrix: {label}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_model_comparison_plot(summaries: list[dict[str, object]], output_path: Path) -> None:
    metric_keys = ("accuracy", "weighted_precision", "weighted_recall", "weighted_f1")
    metric_labels = ("Accuracy", "Weighted precision", "Weighted recall", "Weighted F1")
    positions = np.arange(len(summaries))
    width = 0.8 / len(metric_keys)

    figure, axis = plt.subplots(figsize=(max(9, len(summaries) * 2.5), 6))
    for index, (key, metric_label) in enumerate(zip(metric_keys, metric_labels)):
        values = [float(summary[key]) for summary in summaries]
        offset = (index - (len(metric_keys) - 1) / 2) * width
        axis.bar(positions + offset, values, width, label=metric_label)

    axis.set_title("Zero-shot evaluation comparison")
    axis.set_xlabel("Model")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.set_xticks(positions, [str(summary["label"]) for summary in summaries], rotation=20, ha="right")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

def save_confusion_matrices_comparison(
    matrices: list[tuple[str, list[list[int]]]],
    class_names: list[str],
    output_path: Path,
) -> None:
    columns = min(3, len(matrices))
    rows = (len(matrices) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(6 * columns, 6 * rows), squeeze=False, layout="constrained"
    )
    flat_axes = axes.flat
    maximum = max(int(np.asarray(matrix).max()) for _, matrix in matrices)
    displays = []

    for axis, (label, matrix) in zip(flat_axes, matrices):
        display = ConfusionMatrixDisplay(
            confusion_matrix=np.asarray(matrix), display_labels=class_names
        )
        display.plot(
            ax=axis,
            cmap="Blues",
            colorbar=False,
            include_values=False,
            im_kw={"vmin": 0, "vmax": maximum},
        )
        axis.set_title(label)
        plt.setp(axis.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        plt.setp(axis.get_yticklabels(), fontsize=7)
        displays.append(display)

    for axis in flat_axes:
        if not axis.has_data():
            axis.set_visible(False)
    figure.colorbar(displays[0].im_, ax=axes.ravel().tolist(), shrink=0.8, label="Count")
    figure.suptitle("Confusion-matrix comparison", y=1.01)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

def configure_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("open_clip_zeroshot_compare")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(output_dir / "comparison.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def evaluate_with_progress(
    data_loader: DataLoader,
    class_names: list[str],
    model: torch.nn.Module,
    text_features: torch.Tensor,
    device: torch.device,
    label: str,
) -> tuple[list[str], list[dict[str, object]]]:
    ground_truths: list[str] = []
    predictions: list[dict[str, object]] = []
    with torch.no_grad():
        for image_batch, label_batch, filename_batch in tqdm(
            data_loader, desc=f"{label}: evaluating", unit="batch"
        ):
            image_features = torch.nn.functional.normalize(model.encode_image(image_batch.to(device)), dim=-1)
            similarities = image_features @ text_features.T
            predicted_indices = similarities.argmax(dim=-1).tolist()
            for ground_truth, filename, predicted_index, scores in zip(
                label_batch, filename_batch, predicted_indices, similarities
            ):
                prediction = class_names[predicted_index]
                ground_truths.append(ground_truth)
                predictions.append(
                    {
                        "filename": filename,
                        "ground_truth": ground_truth,
                        "prediction": prediction,
                        "similarity": float(scores[predicted_index].item()),
                    }
                )
    return ground_truths, predictions

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-config", type=Path, default=DEFAULT_MODELS_CONFIG_PATH, help="YAML list of models to compare.")
    parser.add_argument("--model", action="append", help="Evaluate only this configured label; repeat to select several.")
    parser.add_argument("--max-samples", type=int, help="Evaluate only the first N test images.")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    dataset_config = load_comparison_dataset(args.models_config)
    models = select_models(load_models(args.models_config), args.model)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    logger = configure_logger(output_dir)
    logger.info("Device: %s", device)
    logger.info("Selected models: %s", ", ".join(model.label for model in models))
    summaries: list[dict[str, object]] = []
    confusion_matrices: list[tuple[str, list[list[int]]]] = []

    for selection in models:
        logger.info(
            "[%s] Loading %s checkpoint (%s / %s): %s",
            selection.label,
            selection.checkpoint_type,
            selection.model_name,
            selection.pretrained_tag,
            selection.checkpoint_path,
        )
        model, preprocess = load_model(selection, device)
        from dataloader import RSICDZeroShotDataset

        dataset = RSICDZeroShotDataset(
            image_dir=dataset_config.image_dir,
            json_path=dataset_config.json_path,
            class_dir=dataset_config.class_dir,
            transform=preprocess,
        )
        evaluation_dataset = dataset if args.max_samples is None else Subset(dataset, range(min(args.max_samples, len(dataset))))
        if len(evaluation_dataset) == 0:
            raise ValueError("No RSICD test samples selected for evaluation.")
        logger.info(
            "[%s] Test images: %d; classes: %d; batch size: %d",
            selection.label,
            len(evaluation_dataset),
            len(dataset.class_names),
            args.batch_size,
        )
        data_loader = DataLoader(evaluation_dataset, batch_size=args.batch_size, shuffle=False)
        logger.info("[%s] Encoding class prompts", selection.label)
        tokenizer = open_clip.get_tokenizer(selection.model_name)
        text_features = build_text_features(dataset.class_names, model, tokenizer, device)
        logger.info("[%s] Running image evaluation", selection.label)
        ground_truths, predictions = evaluate_with_progress(
            data_loader, dataset.class_names, model, text_features, device, selection.label
        )
        metric_values, matrix = calculate_comparison_metrics(
            ground_truths, [record["prediction"] for record in predictions], dataset.class_names
        )
        model_dir = output_dir / safe_name(selection.label)
        metrics = {
            "label": selection.label,
            "model_name": selection.model_name,
            "pretrained_tag": selection.pretrained_tag,
            "checkpoint_path": str(selection.checkpoint_path),
            "checkpoint_type": selection.checkpoint_type,
            "num_samples": len(predictions),
            "num_classes": len(dataset.class_names),
            **metric_values,
        }
        write_results(model_dir, metrics, predictions)
        save_confusion_matrix(matrix, dataset.class_names, model_dir / "confusion_matrix.png", selection.label)
        confusion_matrices.append((selection.label, matrix))
        summaries.append({key: metrics[key] for key in ("label", "checkpoint_path", "checkpoint_type", "num_samples", "accuracy", "weighted_precision", "weighted_recall", "weighted_f1")})
        print(f"  accuracy={metrics['accuracy']:.4f} weighted_f1={metrics['weighted_f1']:.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump({"models": summaries}, summary_file, indent=2)
        summary_file.write("\n")
    comparison_plot_path = output_dir / "model_comparison.png"
    save_model_comparison_plot(summaries, comparison_plot_path)
    confusion_comparison_path = output_dir / "confusion_matrices_comparison.png"
    save_confusion_matrices_comparison(confusion_matrices, dataset.class_names, confusion_comparison_path)
    logger.info(
        "Comparison complete. Summary: %s; metrics plot: %s; confusion plot: %s",
        output_dir / "summary.json",
        comparison_plot_path,
        confusion_comparison_path,
    )


if __name__ == "__main__":
    main()
