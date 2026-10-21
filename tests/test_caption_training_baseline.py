"""Regression tests for the existing supervised image-caption training path."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as functional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_CLIP_DIR = PROJECT_ROOT / "open-clip"
sys.path.insert(0, str(OPEN_CLIP_DIR))

from dataloader import (  # noqa: E402
    RSICDSupervisedDataset,
    build_supervised_train_val_datasets,
    collate_supervised_captions,
)
from main_finetune import load_config, supervised_contrastive_loss  # noqa: E402


class RecordingTokenizer:
    """Minimal tokenizer that records the reconstructed caption strings."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, captions: list[str]) -> torch.Tensor:
        self.calls.append(list(captions))
        rows = len(captions)
        return torch.arange(rows * 3, dtype=torch.long).reshape(rows, 3)


class LookupModel(torch.nn.Module):
    """Small deterministic model exposing predefined normalized features."""

    def __init__(self) -> None:
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))
        self.register_buffer(
            "image_features",
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        )
        self.register_buffer(
            "text_features",
            torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]]),
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        indices = images[:, 0, 0, 0].long()
        return self.image_features[indices]

    def encode_text(self, texts: torch.Tensor) -> torch.Tensor:
        return self.text_features[texts[:, 0].long()]


def make_record(filename: str, split: str, captions: list[list[str]]) -> dict:
    return {
        "filename": filename,
        "split": split,
        "sentences": [{"tokens": tokens} for tokens in captions],
    }


class CaptionDatasetBaselineTests(unittest.TestCase):
    def test_exact_caption_duplicates_are_removed_in_first_seen_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_dir = Path(temporary_directory)
            (image_dir / "airport.jpg").touch()
            tokenizer = RecordingTokenizer()
            record = make_record(
                "airport.jpg",
                "val",
                [
                    ["an", "airport"],
                    ["aerial", "airport", "view"],
                    ["an", "airport"],
                    ["airport", "near", "roads"],
                ],
            )

            dataset = RSICDSupervisedDataset(
                image_dir=image_dir,
                records=[record],
                transform=lambda image: image,
                tokenizer=tokenizer,
            )

            self.assertEqual(
                tokenizer.calls,
                [["an airport", "aerial airport view", "airport near roads"]],
            )
            self.assertEqual(dataset.caption_tokens[0].shape, (3, 3))

    def test_source_val_split_is_partitioned_by_whole_image_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "images"
            image_dir.mkdir()
            records = [
                make_record(
                    f"labeled_{index}.jpg",
                    "val",
                    [["caption", str(index), "one"], ["caption", str(index), "two"]],
                )
                for index in range(4)
            ]
            records.extend(
                [
                    make_record("unlabeled.jpg", "train", [["must", "not", "load"]]),
                    make_record("held_out.jpg", "test", [["must", "not", "load"]]),
                ]
            )
            for record in records:
                (image_dir / record["filename"]).touch()
            json_path = root / "dataset_rsicd.json"
            json_path.write_text(json.dumps({"images": records}), encoding="utf-8")

            tokenizer = RecordingTokenizer()
            train_dataset, validation_dataset = build_supervised_train_val_datasets(
                image_dir=image_dir,
                json_path=json_path,
                train_transform=lambda image: image,
                validation_transform=lambda image: image,
                tokenizer=tokenizer,
                train_fraction=0.5,
                seed=42,
                source_split="val",
            )

            train_filenames = set(train_dataset.filenames)
            validation_filenames = set(validation_dataset.filenames)
            expected_filenames = {f"labeled_{index}.jpg" for index in range(4)}
            self.assertEqual(train_filenames | validation_filenames, expected_filenames)
            self.assertTrue(train_filenames.isdisjoint(validation_filenames))
            self.assertNotIn("unlabeled.jpg", train_filenames | validation_filenames)
            self.assertNotIn("held_out.jpg", train_filenames | validation_filenames)
            self.assertTrue(
                all(tokens.shape[0] == 2 for tokens in train_dataset.caption_tokens)
            )
            self.assertTrue(
                all(tokens.shape[0] == 2 for tokens in validation_dataset.caption_tokens)
            )

    def test_collate_flattens_captions_and_maps_them_to_source_images(self) -> None:
        batch = [
            (torch.full((1, 2, 2), 1.0), torch.tensor([[10, 11], [12, 13]]), "a.jpg"),
            (torch.full((1, 2, 2), 2.0), torch.tensor([[20, 21]]), "b.jpg"),
        ]

        collated = collate_supervised_captions(batch)

        torch.testing.assert_close(
            collated["texts"], torch.tensor([[10, 11], [12, 13], [20, 21]])
        )
        torch.testing.assert_close(
            collated["caption_to_image"], torch.tensor([0, 0, 1])
        )
        self.assertEqual(collated["filenames"], ["a.jpg", "b.jpg"])
        self.assertEqual(tuple(collated["images"].shape), (2, 1, 2, 2))

    def test_finetune_config_uses_captioned_val_source_split(self) -> None:
        config_path = OPEN_CLIP_DIR / "configs" / "finetune_config.yaml"

        config = load_config(config_path)

        self.assertEqual(config.dataset_split, "val")

    def test_control_config_matches_semiclip_stage1_optimization(self) -> None:
        config_path = (
            OPEN_CLIP_DIR
            / "semiclip"
            / "configs"
            / "supervised_random_control_paper_1gpu.yaml"
        )

        config = load_config(config_path)

        self.assertEqual(config.caption_policy, "random_one")
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.epochs, 25)
        self.assertEqual(config.optimizer, "adamw")
        self.assertEqual(config.scheduler, "cosine")
        self.assertEqual(config.warmup_steps, 10)
        self.assertEqual(config.validation_batch_size, 64)
        self.assertEqual(config.shuffle_seed_offset, 1)
        self.assertAlmostEqual(config.learning_rate, 5e-5)
        self.assertAlmostEqual(config.weight_decay, 5e-4)
        self.assertTrue(config.drop_last)


class SupervisedLossBaselineTests(unittest.TestCase):
    def test_loss_matches_caption_mean_and_multi_positive_image_targets(self) -> None:
        model = LookupModel()
        images = torch.tensor([[[[0.0]]], [[[1.0]]]])
        texts = torch.tensor([[0], [1], [2]])
        caption_to_image = torch.tensor([0, 0, 1])

        actual_loss, statistics = supervised_contrastive_loss(
            model, images, texts, caption_to_image
        )

        logits_text_to_image = model.text_features @ model.image_features.T
        expected_text_to_image = functional.cross_entropy(
            logits_text_to_image, caption_to_image
        )
        expected_image_targets = torch.tensor(
            [[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]
        )
        logits_image_to_text = logits_text_to_image.T
        expected_image_to_text = -(
            expected_image_targets
            * functional.log_softmax(logits_image_to_text, dim=1)
        ).sum(dim=1).mean()
        expected_loss = (expected_text_to_image + expected_image_to_text) / 2

        torch.testing.assert_close(actual_loss, expected_loss)
        expected_positive_text_logit = logits_text_to_image[
            torch.arange(3), caption_to_image
        ].mean()
        expected_positive_image_logit = (
            logits_image_to_text * expected_image_targets
        ).sum(dim=1).mean()
        self.assertAlmostEqual(
            statistics["positive_text_to_image_logit"],
            expected_positive_text_logit.item(),
        )
        self.assertAlmostEqual(
            statistics["positive_image_to_text_logit"],
            expected_positive_image_logit.item(),
        )


if __name__ == "__main__":
    unittest.main()
