"""Regression tests for the RSICD image-only dataset reused from RS-CLIP."""

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_CLIP_DIR = PROJECT_ROOT / "open-clip"
sys.path.insert(0, str(OPEN_CLIP_DIR))

from dataloader import RSICDImageOnlyDataset  # noqa: E402


def tensor_transform(_image):
    return torch.zeros(3, 2, 2)


def paired_transform(_image):
    return torch.zeros(3, 2, 2), torch.ones(3, 2, 2)


class RSICDImageOnlyDatasetTests(unittest.TestCase):
    def _write_image(self, path: Path) -> None:
        Image.new("RGB", (3, 3), color="red").save(path)

    def test_image_only_split_ignores_caption_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "images"
            image_dir.mkdir()
            self._write_image(image_dir / "train.jpg")
            self._write_image(image_dir / "test.jpg")
            metadata = {
                "images": [
                    {
                        "filename": "train.jpg",
                        "split": "train",
                        "sentences": [{"tokens": []}, {"unexpected": "content"}],
                    },
                    {
                        "filename": "test.jpg",
                        "split": "test",
                        "sentences": [{"tokens": ["must", "not", "load"]}],
                    },
                ]
            }
            json_path = root / "dataset.json"
            json_path.write_text(json.dumps(metadata), encoding="utf-8")

            dataset = RSICDImageOnlyDataset(
                image_dir, json_path, tensor_transform, "train"
            )

            self.assertEqual(dataset.filenames, ["train.jpg"])
            image, filename = dataset[0]
            self.assertEqual(filename, "train.jpg")
            self.assertEqual(tuple(image.shape), (3, 2, 2))
            self.assertNotIn(
                "tokenizer", inspect.signature(RSICDImageOnlyDataset).parameters
            )

    def test_transform_may_return_weak_and_strong_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "images"
            image_dir.mkdir()
            self._write_image(image_dir / "train.jpg")
            json_path = root / "dataset.json"
            json_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "filename": "train.jpg",
                                "split": "train",
                                "sentences": [],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            dataset = RSICDImageOnlyDataset(
                image_dir, json_path, paired_transform, "train"
            )

            (weak, strong), filename = dataset[0]
            self.assertEqual(filename, "train.jpg")
            torch.testing.assert_close(weak, torch.zeros(3, 2, 2))
            torch.testing.assert_close(strong, torch.ones(3, 2, 2))

    def test_max_samples_and_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "images"
            image_dir.mkdir()
            records = []
            for index in range(3):
                filename = f"train_{index}.jpg"
                self._write_image(image_dir / filename)
                records.append({"filename": filename, "split": "train"})
            json_path = root / "dataset.json"
            json_path.write_text(json.dumps({"images": records}), encoding="utf-8")

            dataset = RSICDImageOnlyDataset(
                image_dir, json_path, tensor_transform, "train", max_samples=2
            )
            self.assertEqual(dataset.filenames, ["train_0.jpg", "train_1.jpg"])
            with self.assertRaisesRegex(ValueError, "max_samples"):
                RSICDImageOnlyDataset(
                    image_dir, json_path, tensor_transform, "train", max_samples=0
                )
            with self.assertRaisesRegex(ValueError, "No 'test'-split"):
                RSICDImageOnlyDataset(image_dir, json_path, tensor_transform, "test")

    @unittest.skipUnless(
        (PROJECT_ROOT / "dataset_rsicd.json").is_file()
        and (PROJECT_ROOT / "RSICD_images").is_dir(),
        "Local RSICD resources are unavailable",
    )
    def test_local_rsicd_split_has_8734_unique_non_test_images(self) -> None:
        json_path = PROJECT_ROOT / "dataset_rsicd.json"
        dataset = RSICDImageOnlyDataset(
            PROJECT_ROOT / "RSICD_images", json_path, tensor_transform, "train"
        )
        metadata = json.loads(json_path.read_text(encoding="utf-8"))["images"]
        held_out = {
            record["filename"]
            for record in metadata
            if record.get("split") in {"val", "test"}
        }

        self.assertEqual(len(dataset), 8734)
        self.assertEqual(len(set(dataset.filenames)), 8734)
        self.assertTrue(set(dataset.filenames).isdisjoint(held_out))


if __name__ == "__main__":
    unittest.main()
