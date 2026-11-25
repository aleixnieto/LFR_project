"""Evaluate the original and the DINO-adapted image encoder on the RSICD test split.

Default: weighted k-NN as in DINO's eval_knn.py. The labeled split supplies the
reference images, the test split is classified once, and no classifier is trained.
The earlier logistic-regression probe is still available with ``--method linear``
or ``--method both``.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

# Allow imports from the parent open-clip directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloader import RSICDZeroShotDataset
from main_zeroshot import load_dataset_config, load_model_config
from knn_utils import (
    DEFAULT_K,
    DEFAULT_TEMPERATURE,
    encode_labels,
    exact_mcnemar,
    knn_predict,
)
from validate_features import SPLIT_MODES, extract_features, labeled_partition


DEFAULT_CONFIG = (
    Path(__file__).parent / "configs" / "dino_config.yaml"
)

DEFAULT_CHECKPOINT = Path("results/dino/best_model.pth")
DEFAULT_OUTPUT_DIR = Path("evaluations/dino_knn")

ENCODER_SLUGS = {
    "Original CLIP": "original_clip",
    "DINO-adapted CLIP": "dino_adapted",
}


def load_encoder(config_path, checkpoint_path, device):
    """Load original CLIP and the DINO-adapted student encoder."""

    model_config = load_model_config(config_path)

    pretrained_config = open_clip.get_pretrained_cfg(
        model_config.model_name,
        model_config.pretrained_tag,
    )

    clip_model, _, val_transform = (
        open_clip.create_model_and_transforms(
            model_config.model_name,
            pretrained=str(model_config.checkpoint_path),
            device=device,
            image_mean=pretrained_config["mean"],
            image_std=pretrained_config["std"],
            image_interpolation=pretrained_config["interpolation"],
            image_resize_mode=pretrained_config["resize_mode"],
        )
    )

    # Original pretrained encoder
    original_encoder = clip_model.visual

    # Load saved DINO checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Extract ONLY the student encoder parameters.
    # Ignore teacher, projection heads, and center.
    student_state_dict = {
        key.removeprefix("student."): value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith("student.")
    }

    if not student_state_dict:
        raise ValueError(
            "No student encoder weights found in checkpoint."
        )

    # Create an independent copy for adapted features
    adapted_encoder = copy.deepcopy(original_encoder)

    adapted_encoder.load_state_dict(
        student_state_dict,
        strict=True,
    )

    adapted_encoder.to(device).eval()
    original_encoder.eval()

    return original_encoder, adapted_encoder, val_transform, int(checkpoint["epoch"])


def make_loader(
    dataset_config,
    transform,
    split,
    batch_size,
    num_workers,
    device,
):
    dataset = RSICDZeroShotDataset(
        image_dir=dataset_config.image_dir,
        json_path=dataset_config.json_path,
        class_dir=dataset_config.class_dir,
        transform=transform,
        split=split,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def summarize(y_true, y_pred):
    """Accuracy, macro-F1 and counts for one set of test predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "correct": int(np.sum(y_pred == y_true)),
        "total": int(len(y_true)),
    }


def evaluate_encoder(
    encoder,
    train_loader,
    test_loader,
    device,
    method="knn",
    k=DEFAULT_K,
    temperature=DEFAULT_TEMPERATURE,
    validation_k=(),
    split_mode="project",
    split_seed=42,
    json_path=None,
):
    """
    Evaluate frozen features of one encoder.

    knn: the labeled images are the reference set and the test images are
    classified once with a weighted k-NN vote. The held-out part of the labeled
    split is also classified (reference: its training part) to report a
    validation accuracy for several k without touching the test split.
    linear: fit a logistic regression on the labeled images, evaluate on test.
    """

    # Extract frozen representations
    X_train, y_train, train_filenames = extract_features(
        encoder, train_loader, device, return_filenames=True
    )

    X_test, y_test, test_filenames = extract_features(
        encoder, test_loader, device, return_filenames=True
    )

    print(f"Labeled images: {len(y_train)}")
    print(f"Test images: {len(y_test)}")
    print(f"Embedding dimension: {X_train.shape[1]}")

    results = {
        "test_filenames": list(test_filenames),
        "test_labels": [str(label) for label in y_test],
    }

    if method in ("knn", "both"):
        class_names = list(train_loader.dataset.class_names)
        train_targets, _ = encode_labels(y_train, class_names)

        # Validation inside the labeled split (no test image is used here)
        reference_idx, heldout_idx = labeled_partition(
            train_filenames,
            y_train,
            split_mode=split_mode,
            json_path=json_path,
            seed=split_seed,
        )
        validation_accuracy_by_k = {}
        for candidate in sorted(set(validation_k) | {k}):
            predictions = knn_predict(
                X_train[reference_idx],
                train_targets[reference_idx],
                X_train[heldout_idx],
                num_classes=len(class_names),
                k=candidate,
                temperature=temperature,
            )
            validation_accuracy_by_k[int(candidate)] = float(
                np.mean(predictions == train_targets[heldout_idx])
            )

        # Test: every labeled image is a reference image
        predictions = knn_predict(
            X_train,
            train_targets,
            X_test,
            num_classes=len(class_names),
            k=k,
            temperature=temperature,
        )
        predicted_labels = [class_names[index] for index in predictions]
        results["knn"] = {
            **summarize(results["test_labels"], predicted_labels),
            "k": int(k),
            "temperature": float(temperature),
            "reference_images": int(len(y_train)),
            "validation_split": split_mode,
            "validation_reference_images": int(len(reference_idx)),
            "validation_images": int(len(heldout_idx)),
            "validation_accuracy_by_k": validation_accuracy_by_k,
            "predictions": predicted_labels,
        }

    if method in ("linear", "both"):
        # Train linear classifier
        classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
        )

        classifier.fit(X_train, y_train)

        # Evaluate on test set
        predicted_labels = [str(label) for label in classifier.predict(X_test)]
        results["linear_probe"] = {
            **summarize(results["test_labels"], predicted_labels),
            "c": 1.0,
            "max_iter": 1000,
            "train_images": int(len(y_train)),
            "predictions": predicted_labels,
        }

    return results


def compare_encoders(results, method):
    """Exact McNemar test between the two encoders on the test split."""
    truth = np.asarray(results["Original CLIP"]["test_labels"])
    original = np.asarray(results["Original CLIP"][method]["predictions"]) == truth
    adapted = np.asarray(results["DINO-adapted CLIP"][method]["predictions"]) == truth
    original_only = int(np.sum(original & ~adapted))
    adapted_only = int(np.sum(adapted & ~original))
    return {
        "original_only_correct": original_only,
        "dino_only_correct": adapted_only,
        "exact_mcnemar_p": exact_mcnemar(original_only, adapted_only),
    }


def write_outputs(output_dir, results, comparisons, run_info):
    """Save metrics and per-image test predictions so the numbers can be checked."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"run": run_info, "encoders": {}, "comparison": comparisons}

    for name, result in results.items():
        slug = ENCODER_SLUGS[name]
        metrics["encoders"][slug] = {}
        for method in ("knn", "linear_probe"):
            if method not in result:
                continue
            metrics["encoders"][slug][method] = {
                key: value for key, value in result[method].items() if key != "predictions"
            }
            prediction_dir = output_dir / method / slug
            prediction_dir.mkdir(parents=True, exist_ok=True)
            rows = [
                {"filename": filename, "ground_truth": truth, "prediction": prediction}
                for filename, truth, prediction in zip(
                    result["test_filenames"], result["test_labels"], result[method]["predictions"]
                )
            ]
            with (prediction_dir / "predictions.json").open("w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics and predictions saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--method",
        choices=("knn", "linear", "both"),
        default="knn",
        help="knn: weighted k-NN as in DINO (default). linear: logistic-regression probe.",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Neighbors used on the test split.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--validation-k",
        type=int,
        nargs="*",
        default=[1, 5, 10, 20],
        help="Values of k also reported on the held-out labeled images (never on test).",
    )
    parser.add_argument(
        "--split",
        choices=SPLIT_MODES,
        default="project",
        help="Partition of the labeled split used for the validation accuracy.",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    args = parser.parse_args()
    device = torch.device(args.device)

    dataset_config = load_dataset_config(args.config)

    original_encoder, adapted_encoder, val_transform, checkpoint_epoch = load_encoder(
        args.config,
        args.checkpoint,
        device,
    )

    # Labeled development data: k-NN reference set (or training set of the linear probe)
    train_loader = make_loader(
        dataset_config,
        val_transform,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    # Held-out evaluation data
    test_loader = make_loader(
        dataset_config,
        val_transform,
        split="test",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    results = {}

    for name, encoder in [
        ("Original CLIP", original_encoder),
        ("DINO-adapted CLIP", adapted_encoder),
    ]:
        print(f"\nEvaluating {name}...")

        results[name] = evaluate_encoder(
            encoder,
            train_loader,
            test_loader,
            device,
            method=args.method,
            k=args.k,
            temperature=args.temperature,
            validation_k=args.validation_k,
            split_mode=args.split,
            split_seed=args.split_seed,
            json_path=dataset_config.json_path,
        )

    methods = [m for m in ("knn", "linear_probe") if m in results["Original CLIP"]]
    comparisons = {}

    for method in methods:
        title = f"k-NN (k={args.k}, T={args.temperature})" if method == "knn" else "LINEAR PROBE"
        print(f"\n=== FINAL TEST RESULTS: {title} ===")

        for name, result in results.items():
            print(f"\n{name}")
            print(
                f"Accuracy: {result[method]['accuracy']:.4%} "
                f"({result[method]['correct']}/{result[method]['total']})"
            )
            print(f"Macro-F1: {result[method]['macro_f1']:.4f}")
            if method == "knn":
                by_k = ", ".join(
                    f"k={candidate}: {accuracy:.4%}"
                    for candidate, accuracy in result[method]["validation_accuracy_by_k"].items()
                )
                print(f"Validation accuracy ({result[method]['validation_images']} held-out labeled images): {by_k}")

        improvement = (
            results["DINO-adapted CLIP"][method]["accuracy"]
            - results["Original CLIP"][method]["accuracy"]
        )
        comparisons[method] = compare_encoders(results, method)

        print(
            f"\nDINO accuracy improvement: "
            f"{improvement * 100:+.2f} percentage points"
        )
        print(
            f"Only DINO correct: {comparisons[method]['dino_only_correct']}, "
            f"only original correct: {comparisons[method]['original_only_correct']}, "
            f"exact McNemar p = {comparisons[method]['exact_mcnemar_p']:.4g}"
        )

    write_outputs(
        args.output_dir,
        results,
        comparisons,
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint_epoch,
            "method": args.method,
            "labeled_split": "val",
            "evaluation_split": "test",
            "feature_type": "L2-normalized projected OpenCLIP image embeddings",
        },
    )


if __name__ == "__main__":
    main()
