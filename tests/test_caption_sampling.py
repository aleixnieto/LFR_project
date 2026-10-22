"""Tests for deterministic one-caption-per-image sampling."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_CLIP_DIR = PROJECT_ROOT / "open-clip"
sys.path.insert(0, str(OPEN_CLIP_DIR))

from dataloader import (  # noqa: E402
    RSICDSupervisedDataset,
    build_supervised_train_val_datasets,
    collate_supervised_captions,
)
from main_finetune import hash_filename_order, load_config  # noqa: E402
from semiclip.data import ConceptLabeledDataset, collate_concept_labeled  # noqa: E402


class IndexTokenizer:
    """Represent a caption by its position in an image's caption list."""

    def __call__(self, captions: list[str]) -> torch.Tensor:
        return torch.arange(len(captions), dtype=torch.long).unsqueeze(1)


def constant_transform(_image: Image.Image) -> torch.Tensor:
    return torch.zeros((3, 2, 2))


def make_records(count: int, caption_count: int = 5) -> list[dict]:
    return [
        {
            "filename": f"image_{image_index}.png",
            "split": "val",
            "sentences": [
                {"tokens": ["caption", str(caption_index)]}
                for caption_index in range(caption_count)
            ],
        }
        for image_index in range(count)
    ]


def write_images(image_dir: Path, records: list[dict]) -> None:
    image_dir.mkdir()
    for record in records:
        Image.new("RGB", (2, 2)).save(image_dir / record["filename"])


def collect_samples(
    dataset: RSICDSupervisedDataset, num_workers: int
) -> dict[str, int]:
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
        drop_last=False,
        collate_fn=collate_supervised_captions,
    )
    selected: dict[str, int] = {}
    for batch in loader:
        for filename, token_row in zip(batch["filenames"], batch["texts"]):
            selected[filename] = int(token_row[0])
    return selected


class RandomCaptionSamplingTests(unittest.TestCase):
    def make_dataset(
        self,
        image_dir: Path,
        records: list[dict],
        policy: str = "random_one",
        seed: int = 42,
    ) -> RSICDSupervisedDataset:
        return RSICDSupervisedDataset(
            image_dir=image_dir,
            records=records,
            transform=constant_transform,
            tokenizer=IndexTokenizer(),
            caption_policy=policy,
            sampling_seed=seed,
        )

    def test_random_one_is_deterministic_by_seed_epoch_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            records = make_records(8)
            image_dir = root / "images"
            write_images(image_dir, records)
            first = self.make_dataset(image_dir, records)
            second = self.make_dataset(image_dir, records)

            first.set_epoch(3)
            second.set_epoch(3)
            forward = {
                first.filenames[index]: int(first.selected_caption_tokens(index)[0, 0])
                for index in range(len(first))
            }
            reverse = {
                first.filenames[index]: int(first.selected_caption_tokens(index)[0, 0])
                for index in reversed(range(len(first)))
            }
            repeated = {
                second.filenames[index]: int(second.selected_caption_tokens(index)[0, 0])
                for index in range(len(second))
            }

            self.assertEqual(forward, reverse)
            self.assertEqual(forward, repeated)
            self.assertTrue(all(0 <= choice < 5 for choice in forward.values()))

            first.set_epoch(4)
            next_epoch = {
                first.filenames[index]: int(first.selected_caption_tokens(index)[0, 0])
                for index in range(len(first))
            }
            self.assertNotEqual(forward, next_epoch)

    def test_random_one_matches_with_zero_and_four_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            records = make_records(8)
            image_dir = root / "images"
            write_images(image_dir, records)
            dataset = self.make_dataset(image_dir, records)
            dataset.set_epoch(2)

            without_workers = collect_samples(dataset, num_workers=0)
            with_workers = collect_samples(dataset, num_workers=4)

            self.assertEqual(without_workers, with_workers)
            self.assertEqual(len(with_workers), len(records))

    def test_all_unique_ignores_epoch_and_returns_every_caption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            records = make_records(1)
            image_dir = root / "images"
            write_images(image_dir, records)
            dataset = self.make_dataset(image_dir, records, policy="all_unique")

            dataset.set_epoch(9)

            torch.testing.assert_close(
                dataset.selected_caption_tokens(0), torch.arange(5).unsqueeze(1)
            )

    def test_semiclip_collate_preserves_all_unique_caption_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            records = make_records(2, caption_count=2)
            records[0]["sentences"].append({"tokens": ["caption", "0"]})
            image_dir = root / "images"
            write_images(image_dir, records)
            base = self.make_dataset(image_dir, records, policy="all_unique")
            dataset = ConceptLabeledDataset(
                base, {filename: (0,) for filename in base.filenames}, num_concepts=1
            )
            batch = collate_concept_labeled([dataset[0], dataset[1]])

            self.assertEqual(tuple(batch["texts"].shape), (4, 1))
            torch.testing.assert_close(batch["caption_to_image"], torch.tensor([0, 0, 1, 1]))
            self.assertEqual(tuple(batch["concept_targets"].shape), (2, 1))

    def test_validation_always_uses_all_unique_captions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            records = make_records(4)
            image_dir = root / "images"
            write_images(image_dir, records)
            json_path = root / "dataset_rsicd.json"
            json_path.write_text(json.dumps({"images": records}), encoding="utf-8")

            train_dataset, validation_dataset = build_supervised_train_val_datasets(
                image_dir=image_dir,
                json_path=json_path,
                train_transform=constant_transform,
                validation_transform=constant_transform,
                tokenizer=IndexTokenizer(),
                train_fraction=0.5,
                seed=42,
                source_split="val",
                train_caption_policy="random_one",
            )

            self.assertEqual(train_dataset.caption_policy, "random_one")
            self.assertEqual(validation_dataset.caption_policy, "all_unique")
            self.assertEqual(train_dataset.selected_caption_tokens(0).shape[0], 1)
            self.assertEqual(validation_dataset.selected_caption_tokens(0).shape[0], 5)

    def test_invalid_policy_and_negative_epoch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            records = make_records(1)
            image_dir = root / "images"
            write_images(image_dir, records)

            with self.assertRaisesRegex(ValueError, "caption_policy"):
                self.make_dataset(image_dir, records, policy="first")

            dataset = self.make_dataset(image_dir, records)
            with self.assertRaisesRegex(ValueError, "epoch"):
                dataset.set_epoch(-1)

    def test_config_rejects_unknown_caption_policy(self) -> None:
        source_path = OPEN_CLIP_DIR / "configs" / "finetune_config.yaml"
        values = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        values["caption_policy"] = "first"
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "invalid.yaml"
            config_path.write_text(yaml.safe_dump(values), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "caption_policy"):
                load_config(config_path)

    def test_filename_order_hash_is_order_sensitive_and_reproducible(self) -> None:
        filenames = ["a.jpg", "b.jpg", "c.jpg"]

        first = hash_filename_order(filenames)

        self.assertEqual(first, hash_filename_order(list(filenames)))
        self.assertNotEqual(first, hash_filename_order(list(reversed(filenames))))
        self.assertEqual(len(first), 64)


if __name__ == "__main__":
    unittest.main()
