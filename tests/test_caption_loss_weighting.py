"""Tests for configurable text-to-image loss weighting."""

import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import torch
import torch.nn.functional as functional
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_CLIP_DIR = PROJECT_ROOT / "open-clip"
sys.path.insert(0, str(OPEN_CLIP_DIR))

import main_finetune  # noqa: E402
from main_finetune import (  # noqa: E402
    load_config,
    supervised_contrastive_loss,
    validate_one_epoch,
)
from test_caption_training_baseline import LookupModel  # noqa: E402


class TextToImageWeightingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = LookupModel()
        self.images = torch.tensor([[[[0.0]]], [[[1.0]]]])

    def test_per_caption_and_per_image_match_their_formulas(self) -> None:
        texts = torch.tensor([[0], [1], [2]])
        caption_to_image = torch.tensor([0, 0, 1])
        logits = self.model.text_features @ self.model.image_features.T
        caption_losses = functional.cross_entropy(
            logits, caption_to_image, reduction="none"
        )
        expected_per_caption = caption_losses.mean()
        expected_per_image = torch.stack(
            [caption_losses[:2].mean(), caption_losses[2:].mean()]
        ).mean()

        _, caption_statistics = supervised_contrastive_loss(
            self.model,
            self.images,
            texts,
            caption_to_image,
            text_to_image_weighting="per_caption",
        )
        _, image_statistics = supervised_contrastive_loss(
            self.model,
            self.images,
            texts,
            caption_to_image,
            text_to_image_weighting="per_image",
        )

        self.assertAlmostEqual(
            caption_statistics["text_to_image_loss"],
            expected_per_caption.item(),
            places=6,
        )
        self.assertAlmostEqual(
            image_statistics["text_to_image_loss"],
            expected_per_image.item(),
            places=6,
        )
        self.assertNotAlmostEqual(
            caption_statistics["text_to_image_loss"],
            image_statistics["text_to_image_loss"],
            places=6,
        )
        self.assertAlmostEqual(
            caption_statistics["image_to_text_loss"],
            image_statistics["image_to_text_loss"],
            places=6,
        )

    def test_weightings_are_equal_with_one_caption_per_image(self) -> None:
        texts = torch.tensor([[0], [2]])
        caption_to_image = torch.tensor([0, 1])

        caption_loss, caption_statistics = supervised_contrastive_loss(
            self.model,
            self.images,
            texts,
            caption_to_image,
            text_to_image_weighting="per_caption",
        )
        image_loss, image_statistics = supervised_contrastive_loss(
            self.model,
            self.images,
            texts,
            caption_to_image,
            text_to_image_weighting="per_image",
        )

        torch.testing.assert_close(caption_loss, image_loss)
        self.assertAlmostEqual(
            caption_statistics["text_to_image_loss"],
            image_statistics["text_to_image_loss"],
        )

    def test_weightings_are_equal_with_equal_caption_counts(self) -> None:
        texts = torch.tensor([[0], [1], [1], [2]])
        caption_to_image = torch.tensor([0, 0, 1, 1])

        caption_loss, _ = supervised_contrastive_loss(
            self.model,
            self.images,
            texts,
            caption_to_image,
            text_to_image_weighting="per_caption",
        )
        image_loss, _ = supervised_contrastive_loss(
            self.model,
            self.images,
            texts,
            caption_to_image,
            text_to_image_weighting="per_image",
        )

        torch.testing.assert_close(caption_loss, image_loss)

    def test_invalid_weighting_and_unrepresented_image_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "text_to_image_weighting"):
            supervised_contrastive_loss(
                self.model,
                self.images,
                torch.tensor([[0], [2]]),
                torch.tensor([0, 1]),
                text_to_image_weighting="uniform",
            )

        with self.assertRaisesRegex(ValueError, "Every image"):
            supervised_contrastive_loss(
                self.model,
                self.images,
                torch.tensor([[0], [1]]),
                torch.tensor([0, 0]),
                text_to_image_weighting="per_image",
            )

    def test_config_rejects_unknown_weighting(self) -> None:
        source_path = OPEN_CLIP_DIR / "configs" / "finetune_config.yaml"
        values = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        values["text_to_image_weighting"] = "uniform"
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "invalid.yaml"
            config_path.write_text(yaml.safe_dump(values), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "text_to_image_weighting"):
                load_config(config_path)

    def test_validation_explicitly_uses_common_per_caption_objective(self) -> None:
        batch = {
            "images": self.images,
            "texts": torch.tensor([[0], [1], [2]]),
            "caption_to_image": torch.tensor([0, 0, 1]),
            "filenames": ["a.jpg", "b.jpg"],
        }
        observed_weightings: list[str] = []

        def fake_loss(
            _model,
            _images,
            _texts,
            _caption_to_image,
            text_to_image_weighting="per_caption",
        ):
            observed_weightings.append(text_to_image_weighting)
            return torch.tensor(1.25), {
                "text_to_image_loss": 1.0,
                "image_to_text_loss": 1.5,
            }

        with mock.patch.object(
            main_finetune, "supervised_contrastive_loss", side_effect=fake_loss
        ):
            metrics = validate_one_epoch(
                self.model, [batch], torch.device("cpu"), max_batches=None
            )

        self.assertEqual(observed_weightings, ["per_caption"])
        self.assertEqual(metrics["selection_loss"], 1.25)
        self.assertEqual(metrics["text_to_image_loss"], 1.0)
        self.assertEqual(metrics["image_to_text_loss"], 1.5)
        self.assertEqual(metrics["images"], 2)
        self.assertEqual(metrics["captions"], 3)
        self.assertEqual(metrics["captions_per_image"], 1.5)
        self.assertEqual(
            metrics["image_order_sha256"],
            main_finetune.hash_filename_order(["a.jpg", "b.jpg"]),
        )


if __name__ == "__main__":
    unittest.main()
