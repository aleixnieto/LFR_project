"""Create headless training plots from a completed SemiCLIP run."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_epoch_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"SemiCLIP metrics file does not exist: {path}")
    with path.open(encoding="utf-8") as file:
        document = json.load(file)
    records = document.get("epochs") if isinstance(document, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError(f"No epoch records found in: {path}")
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("epoch"), int):
            raise ValueError(f"Invalid epoch record in: {path}")
        if not isinstance(record.get("train"), dict):
            raise ValueError(f"Epoch {record.get('epoch')} has no training metrics: {path}")
    return records


def metric_series(
    records: list[dict[str, object]],
    section: str,
    metric: str,
) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    values: list[float] = []
    for record in records:
        metrics = record.get(section)
        value = metrics.get(metric) if isinstance(metrics, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        epochs.append(int(record["epoch"]))
        values.append(float(value))
    return epochs, values


def _finish_plot(axis, title: str, ylabel: str, output_path: Path) -> Path:
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend()
    axis.figure.tight_layout()
    axis.figure.savefig(output_path, dpi=150)
    plt.close(axis.figure)
    return output_path


def plot_stage1_losses(records: list[dict[str, object]], output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(10, 6))
    for section, metric, label, style in (
        ("train", "clip_loss", "Train CLIP", "-"),
        ("train", "concept_loss", "Train concept", "-"),
        ("validation", "clip_loss", "Validation CLIP", "--"),
    ):
        epochs, values = metric_series(records, section, metric)
        if values:
            axis.plot(epochs, values, style, label=label)
    return _finish_plot(
        axis, "SemiCLIP Stage 1 losses", "Loss", output_dir / "stage1_losses.png"
    )


def plot_stage2_losses(records: list[dict[str, object]], output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(10, 6))
    for metric, label in (
        ("clip_loss", "CLIP"),
        ("concept_loss", "Semantic"),
        ("intramodal_loss", "Intra-modal"),
        ("crossmodal_loss", "Cross-modal"),
    ):
        epochs, values = metric_series(records, "train", metric)
        if values:
            axis.plot(epochs, values, label=label)
    validation_epochs, validation = metric_series(
        records, "validation", "clip_loss"
    )
    if validation:
        axis.plot(validation_epochs, validation, "--", label="Validation CLIP")
    return _finish_plot(
        axis, "SemiCLIP Stage 2 losses", "Loss", output_dir / "stage2_losses.png"
    )


def plot_learning_rates(
    stage1: list[dict[str, object]],
    stage2: list[dict[str, object]],
    output_dir: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(10, 6))
    for records, label in ((stage1, "Stage 1"), (stage2, "Stage 2")):
        epochs, values = metric_series(records, "train", "learning_rate")
        if values:
            axis.plot(epochs, values, label=label)
    return _finish_plot(
        axis, "SemiCLIP learning rate", "Learning rate", output_dir / "learning_rate.png"
    )


def plot_selection_counts(
    records: list[dict[str, object]], output_dir: Path
) -> Path:
    figure, axis = plt.subplots(figsize=(10, 6))
    epochs, values = metric_series(records, "train", "selected_pairs_total")
    if values:
        axis.plot(epochs, values, marker="o", label="Selected pairs")
    return _finish_plot(
        axis,
        "SemiCLIP Stage 2 batch-local selection",
        "Selected image–surrogate pairs",
        output_dir / "selection_counts.png",
    )


def plot_logit_scales(
    stage1: list[dict[str, object]],
    stage2: list[dict[str, object]],
    output_dir: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(10, 6))
    for records, stage in ((stage1, "Stage 1"), (stage2, "Stage 2")):
        for metric, name, style in (
            ("logit_scale", "CLIP", "-"),
            ("concept_logit_scale", "Concept", "--"),
        ):
            epochs, values = metric_series(records, "train", metric)
            if values:
                axis.plot(epochs, values, style, label=f"{stage} {name}")
    return _finish_plot(
        axis, "SemiCLIP logit scales", "Scale", output_dir / "logit_scales.png"
    )


def create_plots(run_dir: str | Path, output_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1 = load_epoch_records(run_dir / "stage1_metrics.json")
    stage2 = load_epoch_records(run_dir / "stage2_metrics.json")
    return [
        plot_stage1_losses(stage1, output_dir),
        plot_stage2_losses(stage2, output_dir),
        plot_learning_rates(stage1, stage2, output_dir),
        plot_selection_counts(stage2, output_dir),
        plot_logit_scales(stage1, stage2, output_dir),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = args.run_dir / "plots"
    return args


def main() -> None:
    args = parse_args()
    print("Saved plots:")
    for path in create_plots(args.run_dir, args.output_dir):
        print(f"  {path}")


if __name__ == "__main__":
    main()
