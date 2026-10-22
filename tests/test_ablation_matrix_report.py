"""Regression tests for the full-factorial ablation report."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "open-clip" / "configs" / "ablation_matrix_search"
sys.path.insert(0, str(REPORT_DIR))

from report_results import generate_report, load_expected_runs  # noqa: E402


class AblationMatrixReportTests(unittest.TestCase):
    def test_complete_matrix_generates_report_csv_plots_and_conclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            results_dir = root / "results"
            output_dir = root / "report"

            expected_runs = load_expected_runs()
            for run in expected_runs:
                run_dir = results_dir / run.name
                run_dir.mkdir(parents=True)
                optimizer_penalty = 0.1 if run.optimizer == "adafactor" else 0.0
                learning_rate_penalties = {
                    5e-6: 0.08,
                    1e-5: 0.0,
                    3e-5: 0.04,
                    5e-5: 0.07,
                    1e-4: 0.12,
                }
                base_loss = (
                    1.0
                    + run.batch_size / 100
                    + optimizer_penalty
                    + learning_rate_penalties[run.learning_rate]
                )
                epochs = []
                for epoch in range(1, 9):
                    validation_loss = base_loss + abs(epoch - 4) * 0.01
                    epochs.append(
                        {
                            "epoch": epoch,
                            "train": {"loss": validation_loss - 0.2},
                            "validation": {
                                "loss": validation_loss,
                                "selection_loss": validation_loss,
                            },
                            "is_best": epoch <= 4,
                        }
                    )
                (run_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "configured_epochs": 8,
                            "epochs": epochs,
                        }
                    ),
                    encoding="utf-8",
                )
                evaluation_dir = (
                    results_dir / "validation_zero_shot" / run.name
                )
                evaluation_dir.mkdir(parents=True)
                is_winner = run.name == "bs32_adamw_lr_1e-5"
                (evaluation_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "accuracy": 0.9 if is_winner else 0.6,
                            "macro_f1": 0.88 if is_winner else 0.55,
                            "weighted_f1": 0.89 if is_winner else 0.58,
                            "validation_filename_sha256": "shared-split-hash",
                        }
                    ),
                    encoding="utf-8",
                )

            results, missing = generate_report(results_dir, output_dir)

            self.assertEqual(len(results), 30)
            self.assertEqual(missing, [])
            self.assertEqual(
                len((output_dir / "summary.csv").read_text().splitlines()),
                31,
            )
            report = (output_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("Completed conclusions", report)
            self.assertEqual(report.count("adamw at 1e-5 had the lowest"), 3)
            self.assertIn("Global validation winner: bs32_adamw_lr_1e-5", report)
            self.assertTrue(
                (output_dir / "validation_loss_by_learning_rate.png").is_file()
            )
            self.assertTrue(
                (output_dir / "best_epoch_by_learning_rate.png").is_file()
            )
            self.assertTrue(
                (output_dir / "classification_metrics_by_learning_rate.png").is_file()
            )
            header = (output_dir / "summary.csv").read_text().splitlines()[0]
            self.assertIn("accuracy", header)
            self.assertIn("macro_f1", header)


if __name__ == "__main__":
    unittest.main()
