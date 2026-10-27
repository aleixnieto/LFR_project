"""Compare configured checkpoints on image-to-text retrieval Accuracy@1, @3, and @5."""

import argparse
import json
from pathlib import Path

import open_clip
import torch
from torch.utils.data import DataLoader

from dataloader import RSICDSupervisedDataset, collate_supervised_captions
from main_retrieval import build_caption_features, evaluate
from main_zeroshot import resolve_device, write_results
from main_zeroshot_compare import (
    load_comparison_dataset,
    load_model,
    load_models,
    safe_name,
    select_models,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("configs") / "retrieval_compare_config.yaml"
DEFAULT_OUTPUT_DIR = Path("evaluations/open_clip_retrieval_comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", action="append", help="Select a label; repeat for several.")
    parser.add_argument("--max-samples", type=int, help="Limit both images and candidate captions.")
    parser.add_argument("--batch-size", type=int, default=32)
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
    selections = select_models(load_models(args.models_config), args.model)
    output_names = [safe_name(selection.label) for selection in selections]
    if len(set(output_names)) != len(output_names):
        raise ValueError("Selected model labels must produce distinct output directory names")
    missing = [
        f"{selection.label}: {selection.checkpoint_path}"
        for selection in selections if not selection.checkpoint_path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing selected checkpoints:\n" + "\n".join(missing))

    with dataset_config.json_path.open(encoding="utf-8") as metadata_file:
        records = json.load(metadata_file)["images"]
    records = [
        record for record in records
        if record.get("split") == dataset_config.dataset_split
    ]
    if args.max_samples is not None:
        records = records[:args.max_samples]
    if not records:
        raise ValueError("No image queries selected for retrieval")

    device = resolve_device(args.device)
    summaries: list[dict[str, object]] = []
    for selection in selections:
        print(f"[{selection.label}] Loading {selection.checkpoint_path}", flush=True)
        model, preprocess = load_model(selection, device)
        tokenizer = open_clip.get_tokenizer(selection.model_name)
        dataset = RSICDSupervisedDataset(
            dataset_config.image_dir, records, preprocess, tokenizer,
            caption_policy="all_unique",
        )
        captions = [caption for image_captions in dataset.captions for caption in image_captions]
        captions_by_image = dict(zip(dataset.filenames, dataset.captions))
        print(f"[{selection.label}] Encoding {len(captions)} captions", flush=True)
        caption_features = torch.cat([
            build_caption_features(
                captions[start:start + args.batch_size], model, tokenizer, device
            )
            for start in range(0, len(captions), args.batch_size)
        ])
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_supervised_captions,
        )
        print(f"[{selection.label}] Evaluating {len(dataset)} images", flush=True)
        predictions = evaluate(
            loader, model, caption_features, captions, captions_by_image, device
        )
        metrics = {
            "label": selection.label,
            "model_name": selection.model_name,
            "pretrained_tag": selection.pretrained_tag,
            "checkpoint_path": str(selection.checkpoint_path),
            "checkpoint_type": selection.checkpoint_type,
            "dataset_split": dataset_config.dataset_split,
            "image_dir": str(dataset_config.image_dir),
            "json_path": str(dataset_config.json_path),
            "device": str(device),
            "num_samples": len(predictions),
            "num_captions": len(captions),
            "max_samples": args.max_samples,
            "matching_rule": "retrieved text is in the query image's caption list",
            **{
                f"accuracy_at_{k}": sum(
                    prediction[f"correct_at_{k}"] for prediction in predictions
                ) / len(predictions)
                for k in (1, 3, 5)
            },
        }
        write_results(args.output_dir / safe_name(selection.label), metrics, predictions)
        summaries.append(metrics)
        del model, caption_features
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump({"models": summaries}, summary_file, indent=2)
        summary_file.write("\n")
    print(f"\n{'Model':40} {'Accuracy@1':>12} {'Accuracy@3':>12} {'Accuracy@5':>12}")
    for metrics in summaries:
        print(
            f"{metrics['label']:40} "
            f"{metrics['accuracy_at_1']:12.4f} "
            f"{metrics['accuracy_at_3']:12.4f} "
            f"{metrics['accuracy_at_5']:12.4f}"
        )
    if args.max_samples is not None:
        print("Subset evaluation: the caption pool is limited to the selected images.")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
