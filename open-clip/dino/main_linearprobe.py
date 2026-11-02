"""Fit a linear probe on RSICD VAL labels and evaluate once on TEST."""

import argparse
import logging
import math
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import open_clip
import torch
import torch.nn.functional as functional
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm


OPEN_CLIP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPEN_CLIP_DIR))

from dataloader import RSICDZeroShotDataset  # noqa: E402
from main_zeroshot import (  # noqa: E402
    load_dataset_config,
    load_model_config,
    require_checkpoint,
    resolve_device,
    write_results,
)
from main_zeroshot_compare import (  # noqa: E402
    calculate_comparison_metrics,
    save_confusion_matrices_comparison,
    save_model_comparison_plot,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("configs") / "linearprobe.yaml"


def save_evaluation_plots(
    ground_truths: list[str],
    predictions: list[str],
    class_names: list[str],
    model_name: str,
    output_dir: Path,
) -> None:
    """Save weighted metric bars and a class-ordered TEST confusion matrix."""
    metrics, matrix = calculate_comparison_metrics(ground_truths, predictions, class_names)
    label = f"{model_name} linear probe"
    save_model_comparison_plot(
        [{"label": label, **metrics}],
        output_dir / "model_comparison.png",
        title="Linear-probe TEST evaluation",
    )
    save_confusion_matrices_comparison(
        [(label, matrix)], class_names, output_dir / "confusion_matrices_comparison.png"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluations/open_clip_linearprobe"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--c", type=float, default=1.0, help="Inverse L2 regularization strength.")
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_iter <= 0:
        parser.error("--batch-size and --max-iter must be positive")
    if not math.isfinite(args.c) or args.c <= 0:
        parser.error("--c must be finite and positive")
    return args


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    dataset: RSICDZeroShotDataset,
    batch_size: int,
    device: torch.device,
    split: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Extract L2-normalized, projected OpenCLIP image features, without captions."""
    model.eval()
    features: list[torch.Tensor] = []
    labels: list[str] = []
    filenames: list[str] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for images, batch_labels, batch_filenames in tqdm(loader, desc=f"Encoding {split}"):
        encoded = model.encode_image(images.to(device))
        features.append(functional.normalize(encoded.float(), dim=-1).cpu())
        labels.extend(batch_labels)
        filenames.extend(batch_filenames)
    return torch.cat(features).numpy(), labels, filenames


def main() -> None:
    args = parse_args()
    dataset_config = load_dataset_config(args.config)
    if dataset_config.dataset_split != "val":
        raise ValueError("Linear probe training requires dataset_split: val")
    model_config = load_model_config(args.config)
    checkpoint_path = require_checkpoint(model_config)
    pretrained_config = open_clip.get_pretrained_cfg(
        model_config.model_name, model_config.pretrained_tag
    )
    if pretrained_config is None:
        raise ValueError("No OpenCLIP preprocessing configuration for the selected model/tag")
    device = resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("open_clip_linearprobe")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "linearprobe.log", encoding="utf-8"),
    ):
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(handler)

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_config.model_name,
        pretrained=str(checkpoint_path),
        device=device,
        image_mean=pretrained_config["mean"],
        image_std=pretrained_config["std"],
        image_interpolation=pretrained_config["interpolation"],
        image_resize_mode=pretrained_config["resize_mode"],
    )
    model.requires_grad_(False)
    model.eval()
    train_dataset = RSICDZeroShotDataset(
        dataset_config.image_dir, dataset_config.json_path,
        dataset_config.class_dir, preprocess, split="val",
    )
    test_dataset = RSICDZeroShotDataset(
        dataset_config.image_dir, dataset_config.json_path,
        dataset_config.class_dir, preprocess, split="test",
    )
    train_names = {record["filename"] for record in train_dataset.images_info}
    test_names = {record["filename"] for record in test_dataset.images_info}
    if train_names & test_names:
        raise ValueError("VAL and TEST contain overlapping image filenames")
    train_features, train_labels, _ = extract_features(
        model, train_dataset, args.batch_size, device, "val"
    )
    missing_classes = set(train_dataset.class_names) - set(train_labels)
    if missing_classes:
        raise ValueError(f"VAL has no training examples for classes: {sorted(missing_classes)}")
    logger.info(
        "Frozen %s; features=%d; VAL training images=%d; C=%g",
        model_config.model_name, train_features.shape[1], len(train_labels), args.c,
    )
    classifier = LogisticRegression(C=args.c, max_iter=args.max_iter, solver="lbfgs")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(train_features, train_labels)

    test_features, test_labels, test_filenames = extract_features(
        model, test_dataset, args.batch_size, device, "test"
    )
    predicted_labels = classifier.predict(test_features).tolist()
    report = classification_report(
        test_labels, predicted_labels, labels=test_dataset.class_names,
        output_dict=True, zero_division=0,
    )
    metrics = {
        "model_name": model_config.model_name,
        "pretrained_tag": model_config.pretrained_tag,
        "checkpoint_path": str(checkpoint_path),
        "train_split": "val",
        "evaluation_split": "test",
        "num_train_samples": len(train_labels),
        "num_samples": len(test_labels),
        "num_classes": len(train_dataset.class_names),
        "class_names": train_dataset.class_names,
        "feature_type": "L2-normalized projected OpenCLIP image embeddings",
        "feature_dim": train_features.shape[1],
        "c": args.c,
        "max_iter": args.max_iter,
        "device": str(device),
        "accuracy": accuracy_score(test_labels, predicted_labels),
        "classification_report": report,
    }
    predictions = [
        {"filename": filename, "ground_truth": truth, "prediction": prediction}
        for filename, truth, prediction in zip(test_filenames, test_labels, predicted_labels)
    ]
    write_results(output_dir, metrics, predictions)
    save_evaluation_plots(
        test_labels, predicted_labels, test_dataset.class_names,
        model_config.model_name, output_dir,
    )
    joblib.dump(
        {"classifier": classifier, "metrics": metrics}, output_dir / "linearprobe.joblib"
    )
    logger.info("TEST images=%d; accuracy=%.4f", len(test_labels), metrics["accuracy"])
    logger.info("\n%s", classification_report(
        test_labels, predicted_labels, labels=test_dataset.class_names,
        digits=4, zero_division=0,
    ))
    logger.info("Results: %s", output_dir)


if __name__ == "__main__":
    main()
