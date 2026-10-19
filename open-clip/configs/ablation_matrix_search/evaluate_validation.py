"""Evaluate matrix checkpoints on the shared derived validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import open_clip
import torch
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset


CONFIG_DIR = Path(__file__).resolve().parent
OPEN_CLIP_DIR = CONFIG_DIR.parents[1]
PROJECT_ROOT = CONFIG_DIR.parents[2]
sys.path.insert(0, str(OPEN_CLIP_DIR))

from dataloader import RSICDZeroShotDataset  # noqa: E402
from main_zeroshot import build_text_features, evaluate, resolve_device, write_results  # noqa: E402
from main_zeroshot_compare import ComparisonModel, load_model  # noqa: E402
from report_results import load_expected_runs  # noqa: E402


DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "finetune_ablation_matrix_search"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "validation_zero_shot"
DEFAULT_CLASS_DIR = PROJECT_ROOT / "txtclasses_rsicd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--class-dir", type=Path, default=DEFAULT_CLASS_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def load_yaml(path: Path) -> dict[str, object]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return document


def resolve_from_config(config_path: Path, value: object, key: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{config_path} lacks string {key}")
    path = Path(value)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def common_settings(config_dir: Path = CONFIG_DIR) -> dict[str, object]:
    paths = sorted(config_dir.glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"No matrix configs in {config_dir}")

    settings: list[dict[str, object]] = []
    for path in paths:
        config = load_yaml(path)
        settings.append(
            {
                "model_name": config["model_name"],
                "pretrained_tag": config["pretrained_tag"],
                "image_dir": resolve_from_config(path, config.get("image_dir"), "image_dir"),
                "json_path": resolve_from_config(path, config.get("json_path"), "json_path"),
                "dataset_split": config["dataset_split"],
                "seed": int(config["seed"]),
                "train_fraction": float(config["supervised_train_fraction"]),
            }
        )
    first = settings[0]
    if any(setting != first for setting in settings[1:]):
        raise ValueError("Matrix configs do not share one dataset, split, seed, and model")
    return first


def validation_indices(dataset_size: int, train_fraction: float, seed: int) -> list[int]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between zero and one")
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(dataset_size, generator=generator).tolist()
    train_size = int(dataset_size * train_fraction)
    return shuffled[train_size:]


def classification_metrics(
    ground_truths: list[str],
    predictions: list[dict[str, object]],
    class_names: list[str],
) -> dict[str, object]:
    predicted_labels = [str(record["prediction"]) for record in predictions]
    weighted = precision_recall_fscore_support(
        ground_truths,
        predicted_labels,
        labels=class_names,
        average="weighted",
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        ground_truths,
        predicted_labels,
        labels=class_names,
        average="macro",
        zero_division=0,
    )
    per_class = precision_recall_fscore_support(
        ground_truths,
        predicted_labels,
        labels=class_names,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(ground_truths, predicted_labels)),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "per_class": {
            name: {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
            }
            for name, precision, recall, f1, support in zip(
                class_names, per_class[0], per_class[1], per_class[2], per_class[3]
            )
        },
    }



def evaluate_all(
    results_dir: Path,
    output_dir: Path,
    class_dir: Path,
    batch_size: int,
    device_name: str,
) -> tuple[list[dict[str, object]], list[str]]:
    expected_runs = load_expected_runs()
    settings = common_settings()
    device = resolve_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    missing_checkpoints: list[str] = []
    for index, run in enumerate(expected_runs, start=1):
        checkpoint_path = results_dir / run.name / "ckp_best.pth"
        model_output_dir = output_dir / run.name
        if not checkpoint_path.is_file():
            print(f"[{index}/{len(expected_runs)}] Missing checkpoint: {run.name}")
            missing_checkpoints.append(run.name)
            continue

        print(f"[{index}/{len(expected_runs)}] Evaluating: {run.name}")
        selection = ComparisonModel(
            label=run.name,
            model_name=str(settings["model_name"]),
            pretrained_tag=str(settings["pretrained_tag"]),
            checkpoint_path=checkpoint_path,
            checkpoint_type="finetune",
        )
        model, preprocess = load_model(selection, device)
        dataset = RSICDZeroShotDataset(
            image_dir=Path(settings["image_dir"]),
            json_path=Path(settings["json_path"]),
            class_dir=class_dir,
            transform=preprocess,
            split=str(settings["dataset_split"]),
        )
        indices = validation_indices(
            len(dataset), float(settings["train_fraction"]), int(settings["seed"])
        )
        subset = Subset(dataset, indices)
        data_loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        tokenizer = open_clip.get_tokenizer(selection.model_name)
        text_features = build_text_features(
            dataset.class_names, model, tokenizer, device
        )
        ground_truths, predictions = evaluate(
            data_loader, dataset.class_names, model, text_features, device
        )
        filenames = [str(record["filename"]) for record in predictions]
        metrics = {
            "label": run.name,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "model_name": selection.model_name,
            "pretrained_tag": selection.pretrained_tag,
            "source_split": settings["dataset_split"],
            "split_seed": settings["seed"],
            "supervised_train_fraction": settings["train_fraction"],
            "num_samples": len(predictions),
            "validation_filename_sha256": hashlib.sha256(
                "\n".join(filenames).encode("utf-8")
            ).hexdigest(),
            **classification_metrics(
                ground_truths, predictions, dataset.class_names
            ),
        }
        write_results(model_output_dir, metrics, predictions)
        summaries.append(metrics)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    hashes = {
        str(summary["validation_filename_sha256"])
        for summary in summaries
        if "validation_filename_sha256" in summary
    }
    if len(hashes) > 1:
        raise RuntimeError("Evaluation runs did not use the same validation filenames")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "models": summaries,
                "missing_checkpoints": missing_checkpoints,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summaries, missing_checkpoints


def main() -> None:
    args = parse_args()
    summaries, missing = evaluate_all(
        results_dir=args.results_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        class_dir=args.class_dir.resolve(),
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(f"Evaluated {len(summaries)} runs; missing {len(missing)}")
    print(f"Summary: {(args.output_dir.resolve() / 'summary.json')}")


if __name__ == "__main__":
    main()
