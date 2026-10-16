"""Plot progress from one or more OpenCLIP fine-tuning runs."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RunHistory:
    label: str
    epochs: list[int]
    train: dict[str, list[float]]
    validation: dict[str, list[float]]


METRICS = {
    "loss": "Loss",
    "positive_text_to_image_logit": "Positive text-to-image logit",
    "positive_image_to_text_logit": "Positive image-to-text logit",
    "logit_scale": "Logit scale",
    "learning_rate": "Learning rate",
}


def metrics_path(path: Path) -> Path:
    """Resolve a run directory or direct metrics.json path."""
    path = path.resolve()
    if path.is_dir():
        path /= "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Metrics file does not exist: {path}")
    return path


def numeric_value(record: dict[str, object], key: str) -> float | None:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def load_history(path: Path, label: str) -> RunHistory:
    path = metrics_path(path)
    with path.open(encoding="utf-8") as metrics_file:
        document = json.load(metrics_file)
    records = document.get("epochs") if isinstance(document, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError(f"No epoch records found in: {path}")

    epochs: list[int] = []
    train = {metric: [] for metric in METRICS}
    validation = {metric: [] for metric in METRICS}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("epoch"), int):
            raise ValueError(f"Invalid epoch record in: {path}")
        train_record = record.get("train")
        validation_record = record.get("validation")
        if not isinstance(train_record, dict) or not isinstance(validation_record, dict):
            raise ValueError(f"Epoch {record['epoch']} lacks train or validation metrics in: {path}")
        epochs.append(record["epoch"])
        for metric in METRICS:
            train_value = numeric_value(train_record, metric)
            validation_value = numeric_value(validation_record, metric)
            train[metric].append(float("nan") if train_value is None else train_value)
            validation[metric].append(float("nan") if validation_value is None else validation_value)
    return RunHistory(label=label, epochs=epochs, train=train, validation=validation)


def has_values(values: list[float]) -> bool:
    return any(value == value for value in values)


def plot_metric(histories: list[RunHistory], metric: str, output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(10, 6))
    for history in histories:
        for split, values, style in (
            ("train", history.train[metric], "-"),
            ("validation", history.validation[metric], "--"),
        ):
            if has_values(values):
                axis.plot(history.epochs, values, style, label=f"{history.label} — {split}")
    axis.set_title(f"{METRICS[metric]} vs epoch")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(METRICS[metric])
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output_path = output_dir / f"{metric}.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="Run directory or metrics.json file.")
    parser.add_argument("--label", action="append", help="Legend label; repeat once per run.")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"), help="Directory for PNG charts (default: plots).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.label is not None and len(args.label) != len(args.runs):
        raise ValueError("Provide either no --label options or exactly one per run")
    labels = args.label or [run.name if run.is_dir() else run.parent.name for run in args.runs]
    histories = [load_history(run, label) for run, label in zip(args.runs, labels)]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Saved plots:")
    for metric in METRICS:
        print(f"  {plot_metric(histories, metric, output_dir)}")


if __name__ == "__main__":
    main()
