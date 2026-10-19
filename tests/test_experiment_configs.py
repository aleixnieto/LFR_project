"""Tests for the three full-run caption-strategy configurations."""

import sys
import unittest
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_CLIP_DIR = PROJECT_ROOT / "open-clip"
sys.path.insert(0, str(OPEN_CLIP_DIR))

from main_finetune import load_config  # noqa: E402


EXPERIMENT_CONFIGS = {
    "finetune_full_config.yaml": ("all_unique", "per_caption"),
    "finetune_full_random_one_config.yaml": ("random_one", "per_caption"),
    "finetune_full_per_image_config.yaml": ("all_unique", "per_image"),
}


class ExperimentConfigTests(unittest.TestCase):
    def test_configs_define_the_expected_strategy_matrix(self) -> None:
        loaded = {
            filename: load_config(
                OPEN_CLIP_DIR / "configs" / "experiment_caption_strategy" / filename
            )
            for filename in EXPERIMENT_CONFIGS
        }

        for filename, expected_strategy in EXPERIMENT_CONFIGS.items():
            with self.subTest(config=filename):
                config = loaded[filename]
                self.assertEqual(
                    (config.caption_policy, config.text_to_image_weighting),
                    expected_strategy,
                )
                self.assertEqual(config.dataset_split, "val")
                self.assertEqual(config.seed, 42)
                self.assertEqual(config.num_workers, 4)
                self.assertEqual(config.epochs, 5)

    def test_non_strategy_controls_are_identical(self) -> None:
        configs = [
            load_config(
                OPEN_CLIP_DIR / "configs" / "experiment_caption_strategy" / filename
            )
            for filename in EXPERIMENT_CONFIGS
        ]
        excluded = {
            "caption_policy",
            "text_to_image_weighting",
            "output_dir",
        }
        controls = [
            {
                key: value
                for key, value in asdict(config).items()
                if key not in excluded
            }
            for config in configs
        ]

        self.assertTrue(all(control == controls[0] for control in controls[1:]))
        self.assertEqual(len({config.output_dir for config in configs}), 3)


if __name__ == "__main__":
    unittest.main()
