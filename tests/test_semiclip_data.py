"""Tests for SemiCLIP dataset isolation, labels, collation, and strong views."""

import json
import random
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_CLIP_DIR = PROJECT_ROOT / "open-clip"
sys.path.insert(0, str(OPEN_CLIP_DIR))

from semiclip.augment import RandAugmentMC  # noqa: E402
from semiclip.config import load_semiclip_config  # noqa: E402
from semiclip.data import (  # noqa: E402
    ConceptLabeledDataset,
    WeakStrongTransform,
    build_semiclip_datasets,
    collate_concept_labeled,
    collate_unlabeled_images,
    collate_unlabeled_views,
)


DEFAULT_CONFIG = OPEN_CLIP_DIR / "semiclip" / "configs" / "semiclip_rsicd.yaml"


def tensor_transform(image: Image.Image) -> torch.Tensor:
    return transforms.ToTensor()(image.resize((8, 8)))


class RecordingTokenizer:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, texts: list[str]) -> torch.Tensor:
        self.texts.extend(texts)
        rows = torch.zeros(len(texts), 6, dtype=torch.long)
        for index, text in enumerate(texts):
            rows[index, 0] = len(text)
        return rows


def record(filename: str, split: str, captions) -> dict:
    return {
        "filename": filename,
        "split": split,
        "sentences": captions,
    }


class SemiCLIPAugmentationTests(unittest.TestCase):
    def test_strong_transform_matches_reference_structure(self) -> None:
        transform = WeakStrongTransform(tensor_transform)
        strong = transform.strong_transform.transforms

        self.assertIsInstance(strong[0], transforms.Resize)
        self.assertEqual(strong[0].size, 256)
        self.assertIsInstance(strong[1], transforms.RandomCrop)
        self.assertEqual(strong[1].size, (224, 224))
        self.assertEqual(strong[1].padding, 28)
        self.assertEqual(strong[1].padding_mode, "reflect")
        self.assertIsInstance(strong[2], transforms.RandomHorizontalFlip)
        self.assertIsInstance(strong[3], RandAugmentMC)
        self.assertEqual((strong[3].n, strong[3].m), (2, 10))

    def test_weak_and_strong_views_have_expected_shapes(self) -> None:
        random.seed(4)
        np.random.seed(4)
        image = Image.new("RGB", (300, 260), color=(20, 40, 60))

        weak, strong = WeakStrongTransform(tensor_transform)(image)

        self.assertEqual(tuple(weak.shape), (3, 8, 8))
        self.assertEqual(tuple(strong.shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(strong).all())

    def test_randaugment_validates_parameters_and_preserves_image_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "n"):
            RandAugmentMC(0, 10)
        with self.assertRaisesRegex(ValueError, "m"):
            RandAugmentMC(2, 11)
        with self.assertRaisesRegex(TypeError, "PIL"):
            RandAugmentMC(2, 10)(torch.zeros(3, 4, 4))

        output = RandAugmentMC(2, 10)(Image.new("RGB", (32, 24), color="blue"))
        self.assertEqual(output.size, (32, 24))


class SemiCLIPDatasetTests(unittest.TestCase):
    def _resources(self, root: Path):
        image_dir = root / "images"
        image_dir.mkdir()
        records = []
        labels = {}
        for index in range(5):
            filename = f"val_{index}.jpg"
            Image.new("RGB", (16, 16), color=(index, 0, 0)).save(image_dir / filename)
            records.append(
                record(
                    filename,
                    "val",
                    [
                        {"tokens": ["caption", str(index), "first"]},
                        {"tokens": ["caption", str(index), "second"]},
                    ],
                )
            )
            labels[filename] = (index % 3,)
        for index in range(3):
            filename = f"train_{index}.jpg"
            Image.new("RGB", (16, 16), color=(0, index, 0)).save(image_dir / filename)
            records.append(
                record(filename, "train", [{"malformed": "must never tokenize"}])
            )
        Image.new("RGB", (16, 16), color="white").save(image_dir / "test.jpg")
        records.append(
            record("test.jpg", "test", [{"tokens": ["reserved", "test"]}])
        )
        json_path = root / "dataset.json"
        json_path.write_text(json.dumps({"images": records}), encoding="utf-8")
        return image_dir, json_path, labels

    def test_builder_keeps_unlabeled_captions_and_test_split_out(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir, json_path, labels = self._resources(root)
            tokenizer = RecordingTokenizer()
            config = replace(
                load_semiclip_config(DEFAULT_CONFIG),
                image_dir=image_dir,
                json_path=json_path,
            )

            datasets = build_semiclip_datasets(
                config,
                tensor_transform,
                tensor_transform,
                tokenizer,
                labels,
                num_concepts=3,
            )

            self.assertEqual(len(datasets.labeled_train), 4)
            self.assertEqual(len(datasets.labeled_validation), 1)
            self.assertEqual(len(datasets.unlabeled_stage1), 3)
            self.assertEqual(len(datasets.unlabeled_stage2), 3)
            self.assertEqual(
                datasets.unlabeled_stage1.filenames,
                ["train_0.jpg", "train_1.jpg", "train_2.jpg"],
            )
            self.assertEqual(
                datasets.unlabeled_stage1.filenames,
                datasets.unlabeled_stage2.filenames,
            )
            self.assertNotIn("test.jpg", datasets.labeled_train.filenames)
            self.assertNotIn("test.jpg", datasets.labeled_validation.filenames)
            self.assertFalse(any("must never tokenize" in text for text in tokenizer.texts))
            self.assertFalse(any("reserved test" in text for text in tokenizer.texts))

    def test_concept_targets_epoch_forwarding_and_collation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir, json_path, labels = self._resources(root)
            config = replace(
                load_semiclip_config(DEFAULT_CONFIG),
                image_dir=image_dir,
                json_path=json_path,
            )
            datasets = build_semiclip_datasets(
                config,
                tensor_transform,
                tensor_transform,
                RecordingTokenizer(),
                labels,
                num_concepts=3,
                max_unlabeled_samples=2,
            )
            datasets.labeled_train.set_epoch(7)
            first = datasets.labeled_train[0]
            second = datasets.labeled_train[1]

            batch = collate_concept_labeled([first, second])

            self.assertEqual(datasets.labeled_train.dataset.epoch, 7)
            self.assertEqual(tuple(batch["images"].shape), (2, 3, 8, 8))
            self.assertEqual(tuple(batch["texts"].shape), (2, 6))
            self.assertEqual(tuple(batch["concept_targets"].shape), (2, 3))
            self.assertEqual(batch["concept_targets"].sum(dim=1).tolist(), [1.0, 1.0])
            self.assertEqual(len(datasets.unlabeled_stage1), 2)

            weak_batch = collate_unlabeled_images(
                [datasets.unlabeled_stage1[0], datasets.unlabeled_stage1[1]]
            )
            view_batch = collate_unlabeled_views(
                [datasets.unlabeled_stage2[0], datasets.unlabeled_stage2[1]]
            )
            self.assertEqual(tuple(weak_batch["images"].shape), (2, 3, 8, 8))
            self.assertEqual(tuple(view_batch["weak_images"].shape), (2, 3, 8, 8))
            self.assertEqual(tuple(view_batch["strong_images"].shape), (2, 3, 224, 224))

    def test_concept_adapter_rejects_missing_and_out_of_range_labels(self) -> None:
        class StubDataset:
            filenames = ["one.jpg"]

            def __len__(self):
                return 1

        with self.assertRaisesRegex(ValueError, "missing"):
            ConceptLabeledDataset(StubDataset(), {}, 2)
        with self.assertRaisesRegex(ValueError, "out of range"):
            ConceptLabeledDataset(StubDataset(), {"one.jpg": (2,)}, 2)


if __name__ == "__main__":
    unittest.main()
