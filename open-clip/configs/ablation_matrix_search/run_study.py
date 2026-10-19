"""Run all or one stage of the matrix ablation study."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).resolve().parent
OPEN_CLIP_DIR = CONFIG_DIR.parents[1]
PROJECT_ROOT = CONFIG_DIR.parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "finetune_ablation_matrix_search"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "train", "evaluate", "report"),
        default="all",
        help="Run the full workflow or exactly one stage (default: all).",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--evaluation-batch-size", type=int, default=32)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    if args.evaluation_batch_size <= 0:
        parser.error("--evaluation-batch-size must be positive")
    return args


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_name(config_path: Path) -> str:
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("output_dir"), str):
        raise ValueError(f"Missing output_dir in {config_path}")
    return Path(document["output_dir"]).name


def train_all(config_paths: list[Path], results_dir: Path, device: str) -> None:
    for index, config_path in enumerate(config_paths, start=1):
        print(f"[{index}/{len(config_paths)}] Training {config_path.stem}", flush=True)
        run([
            sys.executable,
            str(OPEN_CLIP_DIR / "main_finetune.py"),
            "--config",
            str(config_path),
            "--device",
            device,
            "--output-dir",
            str(results_dir / run_name(config_path)),
        ])


def evaluate_all(results_dir: Path, batch_size: int, device: str) -> None:
    run([
        sys.executable,
        str(CONFIG_DIR / "evaluate_validation.py"),
        "--results-dir",
        str(results_dir),
        "--output-dir",
        str(results_dir / "validation_zero_shot"),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
    ])


def generate_report(results_dir: Path) -> None:
    run([
        sys.executable,
        str(CONFIG_DIR / "report_results.py"),
        "--results-dir",
        str(results_dir),
        "--classification-dir",
        str(results_dir / "validation_zero_shot"),
        "--output-dir",
        str(results_dir / "comparison_report"),
    ])


def main() -> None:
    args = parse_args()
    config_paths = sorted(CONFIG_DIR.glob("*.yaml"))
    if not config_paths:
        raise FileNotFoundError(f"No YAML configs found in {CONFIG_DIR}")

    results_dir = args.results_dir.resolve()
    if args.stage in {"all", "train"}:
        train_all(config_paths, results_dir, args.device)
    if args.stage in {"all", "evaluate"}:
        evaluate_all(results_dir, args.evaluation_batch_size, args.device)
    if args.stage in {"all", "report"}:
        generate_report(results_dir)
        print(f"Study report: {results_dir / 'comparison_report' / 'report.md'}")


if __name__ == "__main__":
    main()
