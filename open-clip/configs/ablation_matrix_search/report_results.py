"""Build a comparison report for the full-factorial fine-tuning matrix."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "finetune_ablation_matrix_search"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "comparison_report"
BATCH_SIZES = (16, 32, 64)
OPTIMIZERS = ("adafactor", "adamw")
LEARNING_RATES = (5e-6, 1e-5, 3e-5, 5e-5, 1e-4)


@dataclass(frozen=True)
class ExpectedRun:
    name: str
    batch_size: int
    optimizer: str
    learning_rate: float


@dataclass(frozen=True)
class RunResult:
    name: str
    batch_size: int
    optimizer: str
    learning_rate: float
    best_epoch: int
    configured_epochs: int
    completed_epochs: int
    best_validation_loss: float
    last_validation_loss: float
    best_epoch_train_loss: float
    last_train_loss: float

    @property
    def peaked_before_last(self) -> bool:
        return self.best_epoch < self.completed_epochs

    def csv_row(self) -> dict[str, object]:
        row = asdict(self)
        row["peaked_before_last"] = self.peaked_before_last
        return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Matrix output root containing one directory per run.",
    )
    parser.add_argument(
        "--classification-dir",
        type=Path,
        help="Validation zero-shot results (default: RESULTS_DIR/validation_zero_shot).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination for report.md, summary.csv, and plots.",
    )
    return parser.parse_args()


def read_mapping(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".json":
        document = json.loads(path.read_text(encoding="utf-8"))
    else:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return document


def load_expected_runs(config_dir: Path = CONFIG_DIR) -> list[ExpectedRun]:
    runs: list[ExpectedRun] = []
    for config_path in sorted(config_dir.glob("*.yaml")):
        config = read_mapping(config_path)
        output_dir = config.get("output_dir")
        if not isinstance(output_dir, str):
            raise ValueError(f"Missing output_dir in {config_path}")
        runs.append(
            ExpectedRun(
                name=Path(output_dir).name,
                batch_size=int(config["batch_size"]),
                optimizer=str(config["optimizer"]),
                learning_rate=float(config["learning_rate"]),
            )
        )

    expected_matrix = {
        (batch_size, optimizer, learning_rate)
        for batch_size in BATCH_SIZES
        for optimizer in OPTIMIZERS
        for learning_rate in LEARNING_RATES
    }
    actual_matrix = {
        (run.batch_size, run.optimizer, run.learning_rate) for run in runs
    }
    if actual_matrix != expected_matrix or len(runs) != len(expected_matrix):
        raise ValueError("Config directory does not contain the complete unique matrix")
    return runs


def numeric(record: dict[str, object], key: str, context: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} lacks numeric {key}")
    return float(value)


def load_result(expected: ExpectedRun, results_dir: Path) -> RunResult | None:
    metrics_path = results_dir / expected.name / "metrics.json"
    if not metrics_path.is_file():
        return None

    metrics = read_mapping(metrics_path)
    records = metrics.get("epochs")
    if not isinstance(records, list) or not records:
        raise ValueError(f"No epoch records in {metrics_path}")

    parsed: list[tuple[int, dict[str, object], dict[str, object]]] = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("epoch"), int):
            raise ValueError(f"Invalid epoch record in {metrics_path}")
        train = record.get("train")
        validation = record.get("validation")
        if not isinstance(train, dict) or not isinstance(validation, dict):
            raise ValueError(f"Epoch {record['epoch']} lacks metrics in {metrics_path}")
        parsed.append((record["epoch"], train, validation))

    best = min(
        parsed,
        key=lambda row: numeric(
            row[2], "selection_loss", f"{expected.name} epoch {row[0]}"
        ),
    )
    last = parsed[-1]
    return RunResult(
        name=expected.name,
        batch_size=expected.batch_size,
        optimizer=expected.optimizer,
        learning_rate=expected.learning_rate,
        best_epoch=best[0],
        configured_epochs=int(metrics.get("configured_epochs", len(parsed))),
        completed_epochs=len(parsed),
        best_validation_loss=numeric(
            best[2], "selection_loss", f"{expected.name} best validation"
        ),
        last_validation_loss=numeric(
            last[2], "selection_loss", f"{expected.name} last validation"
        ),
        best_epoch_train_loss=numeric(
            best[1], "loss", f"{expected.name} best-epoch train"
        ),
        last_train_loss=numeric(last[1], "loss", f"{expected.name} last train"),
    )


def load_classifications(
    expected: list[ExpectedRun], classification_dir: Path
) -> tuple[dict[str, dict[str, float]], list[str]]:
    classifications: dict[str, dict[str, float]] = {}
    missing: list[str] = []
    split_hashes: set[str] = set()
    for run in expected:
        metrics_path = classification_dir / run.name / "metrics.json"
        if not metrics_path.is_file():
            missing.append(run.name)
            continue
        metrics = read_mapping(metrics_path)
        classifications[run.name] = {
            "accuracy": numeric(metrics, "accuracy", run.name),
            "macro_f1": numeric(metrics, "macro_f1", run.name),
            "weighted_f1": numeric(metrics, "weighted_f1", run.name),
        }
        split_hash = metrics.get("validation_filename_sha256")
        if not isinstance(split_hash, str) or not split_hash:
            raise ValueError(f"{metrics_path} lacks validation split hash")
        split_hashes.add(split_hash)

    if len(split_hashes) > 1:
        raise ValueError(
            "Classification results do not use the same validation filenames"
        )
    return classifications, missing


def lr_label(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def write_csv(
    results: list[RunResult],
    classifications: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    rows = []
    for result in results:
        row = result.csv_row()
        classification = classifications.get(result.name, {})
        row["accuracy"] = classification.get("accuracy", "")
        row["macro_f1"] = classification.get("macro_f1", "")
        row["weighted_f1"] = classification.get("weighted_f1", "")
        rows.append(row)
    fieldnames = list(rows[0])
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def lookup_results(
    results: list[RunResult],
) -> dict[tuple[int, str, float], RunResult]:
    return {
        (result.batch_size, result.optimizer, result.learning_rate): result
        for result in results
    }


def plot_metric(
    results: list[RunResult],
    attribute: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    lookup = lookup_results(results)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5), squeeze=False)
    for axis, batch_size in zip(axes[0], BATCH_SIZES):
        for optimizer in OPTIMIZERS:
            available = [
                lookup[(batch_size, optimizer, learning_rate)]
                for learning_rate in LEARNING_RATES
                if (batch_size, optimizer, learning_rate) in lookup
            ]
            if available:
                axis.plot(
                    [result.learning_rate for result in available],
                    [getattr(result, attribute) for result in available],
                    marker="o",
                    label=optimizer,
                )
        axis.set_xscale("log")
        axis.set_title(f"Batch size {batch_size}")
        axis.set_xlabel("Learning rate")
        axis.set_ylabel(ylabel)
        if attribute == "best_epoch":
            axis.set_ylim(0.5, 8.5)
            axis.set_yticks(range(1, 9))
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_classification_metrics(
    results: list[RunResult],
    classifications: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    lookup = lookup_results(results)
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), squeeze=False)
    for row_index, (metric, label) in enumerate(
        (("accuracy", "Accuracy"), ("macro_f1", "Macro F1"))
    ):
        for axis, batch_size in zip(axes[row_index], BATCH_SIZES):
            for optimizer in OPTIMIZERS:
                available = [
                    lookup[(batch_size, optimizer, learning_rate)]
                    for learning_rate in LEARNING_RATES
                    if (batch_size, optimizer, learning_rate) in lookup
                    and lookup[(batch_size, optimizer, learning_rate)].name
                    in classifications
                ]
                if available:
                    axis.plot(
                        [result.learning_rate for result in available],
                        [classifications[result.name][metric] for result in available],
                        marker="o",
                        label=optimizer,
                    )
            axis.set_xscale("log")
            axis.set_ylim(0, 1)
            axis.set_title(f"{label}, batch size {batch_size}")
            axis.set_xlabel("Learning rate")
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.3)
            axis.legend()
    figure.suptitle("Zero-shot classification on the shared validation split")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def winner_for_batch(
    results: list[RunResult], batch_size: int
) -> RunResult | None:
    candidates = [result for result in results if result.batch_size == batch_size]
    if not candidates:
        return None
    return min(candidates, key=lambda result: result.best_validation_loss)


def markdown_matrix(results: list[RunResult], batch_size: int) -> list[str]:
    lookup = lookup_results(results)
    candidates = [result for result in results if result.batch_size == batch_size]
    minimum = min(
        (result.best_validation_loss for result in candidates), default=None
    )
    lines = [
        "| Learning rate | Adafactor loss | Peak epoch | AdamW loss | Peak epoch |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for learning_rate in LEARNING_RATES:
        cells = [lr_label(learning_rate)]
        for optimizer in OPTIMIZERS:
            result = lookup.get((batch_size, optimizer, learning_rate))
            if result is None:
                cells.extend(("missing", "missing"))
                continue
            loss = f"{result.best_validation_loss:.6f}"
            if minimum is not None and result.best_validation_loss == minimum:
                loss = f"**{loss}**"
            cells.extend((loss, str(result.best_epoch)))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def markdown_classification_table(
    results: list[RunResult], classifications: dict[str, dict[str, float]]
) -> list[str]:
    lines = [
        "| Run | Accuracy | Macro F1 | Weighted F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    if not classifications:
        lines.append("| No evaluated runs | — | — | — |")
        return lines
    max_accuracy = max(values["accuracy"] for values in classifications.values())
    max_macro_f1 = max(values["macro_f1"] for values in classifications.values())
    for result in results:
        values = classifications.get(result.name)
        if values is None:
            lines.append(f"| {result.name} | missing | missing | missing |")
            continue
        accuracy = f"{values['accuracy']:.6f}"
        macro_f1 = f"{values['macro_f1']:.6f}"
        if values["accuracy"] == max_accuracy:
            accuracy = f"**{accuracy}**"
        if values["macro_f1"] == max_macro_f1:
            macro_f1 = f"**{macro_f1}**"
        lines.append(
            f"| {result.name} | {accuracy} | {macro_f1} | "
            f"{values['weighted_f1']:.6f} |"
        )
    return lines


def conclusion(
    results: list[RunResult],
    expected_count: int,
    missing: list[str],
    classifications: dict[str, dict[str, float]],
    missing_classifications: list[str],
) -> list[str]:
    status = "Preliminary" if missing or missing_classifications else "Completed"
    lines = [f"{status} conclusions based on {len(results)}/{expected_count} runs:", ""]
    winners: list[RunResult] = []
    for batch_size in BATCH_SIZES:
        winner = winner_for_batch(results, batch_size)
        if winner is None:
            lines.append(f"- Batch size {batch_size}: no completed runs.")
            continue
        winners.append(winner)
        lines.append(
            f"- Batch size {batch_size}: {winner.optimizer} at "
            f"{lr_label(winner.learning_rate)} had the lowest validation loss "
            f"({winner.best_validation_loss:.6f}) at epoch {winner.best_epoch}."
        )
    if len(winners) == len(BATCH_SIZES):
        choices = {(winner.optimizer, winner.learning_rate) for winner in winners}
        if len(choices) == 1:
            winner = winners[0]
            lines.append(
                f"- {winner.optimizer} at {lr_label(winner.learning_rate)} won "
                "within all three batch sizes, indicating a stable trend."
            )
        else:
            lines.append(
                "- The within-batch winner changes with batch size, indicating "
                "an interaction between the matrix dimensions."
            )
    early = sum(result.peaked_before_last for result in results)
    final = sum(result.best_epoch == result.completed_epochs for result in results)
    lines.extend((
        f"- {early} runs peaked before their final completed epoch; "
        f"{final} were still best at their final completed epoch.",
        "",
        "**No global batch-size winner is selected from contrastive loss.** "
        "Its scale changes with the number of in-batch negatives.",
    ))
    if classifications:
        best_accuracy = max(values["accuracy"] for values in classifications.values())
        best_macro_f1 = max(values["macro_f1"] for values in classifications.values())
        accuracy_leaders = sorted(
            name for name, values in classifications.items()
            if values["accuracy"] == best_accuracy
        )
        macro_f1_leaders = sorted(
            name for name, values in classifications.items()
            if values["macro_f1"] == best_macro_f1
        )
        joint_leaders = sorted(set(accuracy_leaders) & set(macro_f1_leaders))
        lines.extend((
            "",
            f"- Accuracy leader(s): {', '.join(accuracy_leaders)} ({best_accuracy:.6f}).",
            f"- Macro-F1 leader(s): {', '.join(macro_f1_leaders)} ({best_macro_f1:.6f}).",
        ))
        if len(joint_leaders) == 1 and not missing and not missing_classifications:
            lines.append(
                f"- **Global validation winner: {joint_leaders[0]}.** It leads "
                "both zero-shot accuracy and macro F1 on the shared split."
            )
        elif len(joint_leaders) == 1:
            lines.append(
                f"- Current joint leader: {joint_leaders[0]}. The evaluation "
                "matrix is incomplete, so no global winner is declared yet."
            )
        else:
            lines.append(
                "- Accuracy and macro F1 do not identify one unique joint leader; "
                "no unique global winner is declared. Carry the metric leaders "
                "into same-split, multi-seed confirmation."
            )
    else:
        lines.append(
            "Run the shared-split zero-shot evaluation before selecting across "
            "batch sizes."
        )
    if missing or missing_classifications:
        lines.extend(("", "The matrix is incomplete, so these conclusions are provisional."))
    return lines


def write_report(
    expected: list[ExpectedRun],
    results: list[RunResult],
    missing: list[str],
    classifications: dict[str, dict[str, float]],
    missing_classifications: list[str],
    output_path: Path,
) -> None:
    lines = [
        "# Matrix-ablation comparison report", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
        "## Completion", "",
        f"- Expected runs: {len(expected)}",
        f"- Runs with metrics: {len(results)}",
        f"- Missing runs: {len(missing)}",
        f"- Runs with classification metrics: {len(classifications)}",
        f"- Missing classification runs: {len(missing_classifications)}",
    ]
    if missing:
        lines.extend(("", "Missing run directories or metrics:", ""))
        lines.extend(f"- {name}" for name in missing)
    if missing_classifications:
        lines.extend(("", "Missing zero-shot classification metrics:", ""))
        lines.extend(f"- {name}" for name in missing_classifications)
    lines.extend((
        "", "## Comparison rules", "",
        "All runs use the same seed-42 875/219 train/validation split and "
        "the same all-unique/per-caption validation objective. Best epoch "
        "means minimum validation.selection_loss.", "",
        "Loss values are compared only within a fixed batch size. Cross-"
        "batch selection uses zero-shot accuracy and macro F1 on the same "
        "held-out validation images.",
    ))
    for batch_size in BATCH_SIZES:
        lines.extend(("", f"## Batch size {batch_size}", ""))
        lines.extend(markdown_matrix(results, batch_size))
    lines.extend((
        "", "## Peak timing", "",
        "| Run | Best epoch | Completed epochs | Best validation loss | Last validation loss |",
        "| --- | ---: | ---: | ---: | ---: |",
    ))
    for result in results:
        lines.append(
            f"| {result.name} | {result.best_epoch} | "
            f"{result.completed_epochs}/{result.configured_epochs} | "
            f"{result.best_validation_loss:.6f} | {result.last_validation_loss:.6f} |"
        )
    lines.extend(("", "## Shared-split zero-shot classification", ""))
    lines.extend(markdown_classification_table(results, classifications))
    lines.extend(("", "## Conclusion", ""))
    lines.extend(conclusion(
        results, len(expected), missing, classifications, missing_classifications
    ))
    lines.extend((
        "", "## Artifacts", "", "- summary.csv",
        "- validation_loss_by_learning_rate.png",
        "- best_epoch_by_learning_rate.png",
        "- classification_metrics_by_learning_rate.png (when available)",
    ))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_report(
    results_dir: Path,
    output_dir: Path,
    config_dir: Path = CONFIG_DIR,
    classification_dir: Path | None = None,
) -> tuple[list[RunResult], list[str]]:
    expected = load_expected_runs(config_dir)
    results: list[RunResult] = []
    missing: list[str] = []
    for run in expected:
        result = load_result(run, results_dir)
        if result is None:
            missing.append(run.name)
        else:
            results.append(result)
    if not results:
        raise FileNotFoundError(f"No metrics.json files found beneath {results_dir}")
    results.sort(key=lambda result: (
        result.batch_size, result.learning_rate, result.optimizer
    ))
    if classification_dir is None:
        classification_dir = results_dir / "validation_zero_shot"
    classifications, _ = load_classifications(expected, classification_dir)
    result_names = {result.name for result in results}
    classifications = {
        name: values for name, values in classifications.items()
        if name in result_names
    }
    missing_classifications = [
        run.name for run in expected if run.name not in classifications
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(results, classifications, output_dir / "summary.csv")
    plot_metric(
        results, "best_validation_loss", "Best validation loss",
        "Best held-out contrastive loss within each batch size",
        output_dir / "validation_loss_by_learning_rate.png",
    )
    plot_metric(
        results, "best_epoch", "Best epoch",
        "Epoch with minimum held-out contrastive loss",
        output_dir / "best_epoch_by_learning_rate.png",
    )
    if classifications:
        plot_classification_metrics(
            results, classifications,
            output_dir / "classification_metrics_by_learning_rate.png",
        )
    write_report(
        expected, results, missing, classifications, missing_classifications,
        output_dir / "report.md",
    )
    return results, missing

def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    classification_dir = (
        args.classification_dir.resolve()
        if args.classification_dir is not None
        else results_dir / "validation_zero_shot"
    )
    output_dir = args.output_dir.resolve()
    results, missing = generate_report(
        results_dir=results_dir,
        output_dir=output_dir,
        classification_dir=classification_dir,
    )
    print(f"Compared {len(results)} runs; missing {len(missing)}")
    print(f"Report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
