"""Tests for SemiCLIP plots and zero-shot comparison configuration."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_CLIP_DIR = PROJECT_ROOT / "open-clip"
sys.path.insert(0, str(OPEN_CLIP_DIR))

from main_zeroshot_compare import load_comparison_dataset, load_models  # noqa: E402
from plot_semiclip import create_plots, load_epoch_records  # noqa: E402


COMPARISON_CONFIG = (
    OPEN_CLIP_DIR / "configs" / "semiclip" / "zeroshot_compare_config.yaml"
)


class SemiCLIPReportingTests(unittest.TestCase):
    def test_plotter_creates_all_stage_specific_charts(self) -> None:
        stage1 = {
            "epochs": [
                {
                    "epoch": 0,
                    "train": {
                        "clip_loss": 1.2,
                        "concept_loss": 0.8,
                        "learning_rate": 5e-5,
                        "logit_scale": 14.0,
                        "concept_logit_scale": 12.0,
                    },
                    "validation": {"clip_loss": 1.1},
                },
                {
                    "epoch": 1,
                    "train": {
                        "clip_loss": 1.0,
                        "concept_loss": 0.7,
                        "learning_rate": 4e-5,
                        "logit_scale": 14.2,
                        "concept_logit_scale": 12.2,
                    },
                    "validation": {"clip_loss": 0.9},
                },
            ]
        }
        stage2 = {
            "epochs": [
                {
                    "epoch": 0,
                    "train": {
                        "clip_loss": 0.9,
                        "concept_loss": 0.6,
                        "intramodal_loss": 0.3,
                        "crossmodal_loss": 0.2,
                        "learning_rate": 5e-5,
                        "selected_pairs_total": 20,
                        "logit_scale": 14.3,
                        "concept_logit_scale": 12.4,
                    },
                    "validation": {"clip_loss": 0.8},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "stage1_metrics.json").write_text(
                json.dumps(stage1), encoding="utf-8"
            )
            (run_dir / "stage2_metrics.json").write_text(
                json.dumps(stage2), encoding="utf-8"
            )

            paths = create_plots(run_dir, root / "plots")

            self.assertEqual(
                {path.name for path in paths},
                {
                    "stage1_losses.png",
                    "stage2_losses.png",
                    "learning_rate.png",
                    "selection_counts.png",
                    "logit_scales.png",
                },
            )
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths))

    def test_metrics_loader_rejects_missing_or_empty_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "metrics.json"
            with self.assertRaises(FileNotFoundError):
                load_epoch_records(path)
            path.write_text(json.dumps({"epochs": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No epoch"):
                load_epoch_records(path)

    def test_comparison_config_lists_baseline_supervised_and_semiclip(self) -> None:
        dataset = load_comparison_dataset(COMPARISON_CONFIG)
        models = load_models(COMPARISON_CONFIG)

        self.assertEqual(dataset.dataset_split, "test")
        self.assertEqual([model.label for model in models], ["baseline", "supervised", "semiclip"])
        self.assertEqual(
            [model.checkpoint_type for model in models],
            ["openclip", "finetune", "finetune"],
        )
        self.assertEqual(models[-1].checkpoint_path.name, "stage2_final.pt")


if __name__ == "__main__":
    unittest.main()
