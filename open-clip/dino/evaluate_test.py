
import argparse
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
from validate_features import extract_features


DEFAULT_CONFIG = (
    Path(__file__).parent / "configs" / "dino_config.yaml"
)

DEFAULT_CHECKPOINT = Path("results/dino/best_model.pth")


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
    import copy
    adapted_encoder = copy.deepcopy(original_encoder)

    adapted_encoder.load_state_dict(
        student_state_dict,
        strict=True,
    )

    adapted_encoder.to(device).eval()
    original_encoder.eval()

    return original_encoder, adapted_encoder, val_transform


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


def evaluate_encoder(encoder, train_loader, test_loader, device):
    """
    Fit a linear probe on labeled validation data.
    Evaluate once on the held-out test data.
    """

    # Extract frozen representations
    X_train, y_train = extract_features(
        encoder, train_loader, device
    )

    X_test, y_test = extract_features(
        encoder, test_loader, device
    )

    print(f"Probe training images: {len(y_train)}")
    print(f"Test images: {len(y_test)}")
    print(f"Embedding dimension: {X_train.shape[1]}")

    # Train linear classifier
    classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
    )

    classifier.fit(X_train, y_train)

    # Evaluate on test set
    predictions = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    n_correct = int(np.sum(predictions == y_test))
    n_total = len(y_test)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "correct": n_correct,
        "total": n_total,
    }


def main():
    parser = argparse.ArgumentParser()

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

    args = parser.parse_args()
    device = torch.device(args.device)

    dataset_config = load_dataset_config(args.config)

    original_encoder, adapted_encoder, val_transform = load_encoder(
        args.config,
        args.checkpoint,
        device,
    )

    # Labeled development data: fit the final linear probe
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
        )

    print("\n=== FINAL TEST RESULTS ===")

    for name, result in results.items():
        print(f"\n{name}")
        print(
            f"Accuracy: {result['accuracy']:.4%} "
            f"({result['correct']}/{result['total']})"
        )
        print(f"Macro-F1: {result['macro_f1']:.4f}")

    improvement = (
        results["DINO-adapted CLIP"]["accuracy"]
        - results["Original CLIP"]["accuracy"]
    )

    print(
        f"\nDINO accuracy improvement: "
        f"{improvement * 100:+.2f} percentage points"
    )


if __name__ == "__main__":
    main()
