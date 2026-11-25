"""Select a saved DINO checkpoint with the weighted k-NN monitor, without retraining.

Every given checkpoint (and the original CLIP encoder, shown as epoch 0) is scored
by k-NN accuracy on the held-out part of the labeled split. The test split is never
loaded here: run ``evaluate_test.py`` once on the selected checkpoint afterwards.

Example, from the repository root:

    python open-clip/dino/select_checkpoint.py \
        --checkpoints results/dino/checkpoint_epoch_*.pth
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Allow imports from the parent open-clip directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main_zeroshot import load_dataset_config
from evaluate_test import DEFAULT_CONFIG, load_encoder, make_loader
from knn_utils import DEFAULT_K, DEFAULT_TEMPERATURE, encode_labels, knn_predict
from validate_features import SPLIT_MODES, extract_features, labeled_partition


def heldout_knn_accuracy(encoder, loader, device, k_values, temperature, split_mode, split_seed, json_path):
    """k-NN accuracy on the held-out labeled images for each k (reference: the other labeled images)."""
    features, labels, filenames = extract_features(encoder, loader, device, return_filenames=True)
    reference_idx, heldout_idx = labeled_partition(
        filenames, labels, split_mode=split_mode, json_path=json_path, seed=split_seed
    )
    class_names = list(loader.dataset.class_names)
    targets, _ = encode_labels(labels, class_names)
    accuracies = {}
    for k in k_values:
        predictions = knn_predict(
            features[reference_idx],
            targets[reference_idx],
            features[heldout_idx],
            num_classes=len(class_names),
            k=k,
            temperature=temperature,
        )
        accuracies[int(k)] = float(np.mean(predictions == targets[heldout_idx]))
    return accuracies, int(len(reference_idx)), int(len(heldout_idx))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Neighbors used for the selection.")
    parser.add_argument("--extra-k", type=int, nargs="*", default=[1, 5, 10], help="Other values of k to report.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--split", choices=SPLIT_MODES, default="project")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("results/dino/knn_selection.json"))
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset_config = load_dataset_config(args.config)
    k_values = sorted(set(args.extra_k) | {args.k})

    rows = []
    loader = None
    for checkpoint_path in args.checkpoints:
        original_encoder, adapted_encoder, val_transform, epoch = load_encoder(
            args.config, checkpoint_path, device
        )
        if loader is None:
            # Labeled split only; the test split is not used for the selection
            loader = make_loader(
                dataset_config, val_transform, split="val",
                batch_size=args.batch_size, num_workers=args.num_workers, device=device,
            )
            accuracies, n_reference, n_heldout = heldout_knn_accuracy(
                original_encoder, loader, device, k_values, args.temperature,
                args.split, args.split_seed, dataset_config.json_path,
            )
            rows.append({"epoch": 0, "checkpoint": "original CLIP", "accuracy_by_k": accuracies})
        accuracies, n_reference, n_heldout = heldout_knn_accuracy(
            adapted_encoder, loader, device, k_values, args.temperature,
            args.split, args.split_seed, dataset_config.json_path,
        )
        rows.append({"epoch": epoch, "checkpoint": str(checkpoint_path), "accuracy_by_k": accuracies})

    rows.sort(key=lambda row: row["epoch"])
    candidates = [row for row in rows if row["epoch"] > 0]
    # Highest accuracy at the selection k; the earliest epoch wins a tie, as in main_dino.py
    best = max(candidates, key=lambda row: (row["accuracy_by_k"][args.k], -row["epoch"]))
    baseline = rows[0]

    print(f"\nk-NN accuracy on the {n_heldout} held-out labeled images "
          f"(reference: {n_reference} labeled images, split: {args.split}, T={args.temperature})")
    print("epoch  " + "  ".join(f"k={k:<3d}" for k in k_values) + "  checkpoint")
    for row in rows:
        marker = "  <- selected" if row is best else ""
        print(f"{row['epoch']:5d}  " + "  ".join(f"{row['accuracy_by_k'][k]:.4f}" for k in k_values)
              + f"  {row['checkpoint']}{marker}")
    print(f"\nSelected with k={args.k}: {best['checkpoint']} (epoch {best['epoch']}, "
          f"accuracy {best['accuracy_by_k'][args.k]:.4f}; original CLIP {baseline['accuracy_by_k'][args.k]:.4f})")
    if baseline["accuracy_by_k"][args.k] >= best["accuracy_by_k"][args.k]:
        print("Note: no checkpoint beats the original CLIP encoder on the held-out images.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump({
            "selection_k": args.k,
            "temperature": args.temperature,
            "split": args.split,
            "split_seed": args.split_seed,
            "reference_images": n_reference,
            "heldout_images": n_heldout,
            "selected_checkpoint": best["checkpoint"],
            "selected_epoch": best["epoch"],
            "checkpoints": rows,
        }, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
