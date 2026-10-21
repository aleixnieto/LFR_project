"""Tests for SemiCLIP configuration, record splitting, and concept mining."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_CLIP_DIR = PROJECT_ROOT / "open-clip"
sys.path.insert(0, str(OPEN_CLIP_DIR))

from dataloader import split_rsicd_records  # noqa: E402
from semiclip_concepts import (  # noqa: E402
    hash_filenames,
    mine_semantic_concepts,
    write_concept_artifact,
)
from semiclip_config import load_semiclip_config  # noqa: E402


DEFAULT_CONFIG = OPEN_CLIP_DIR / "configs" / "semiclip" / "semiclip_rsicd.yaml"


def fake_tokenize(text: str) -> list[str]:
    return text.split()


def fake_tag(words) -> list[tuple[str, str]]:
    return [(word, "VB" if word == "crosses" else "NN") for word in words]


def record(filename: str, *captions: str, split: str = "val") -> dict:
    return {
        "filename": filename,
        "split": split,
        "sentences": [{"tokens": caption.split()} for caption in captions],
    }


class SemiCLIPConfigTests(unittest.TestCase):
    def test_default_config_matches_reference_workflow(self) -> None:
        config = load_semiclip_config(DEFAULT_CONFIG)

        self.assertEqual(config.model_name, "ViT-B-32")
        self.assertEqual(config.labeled_split, "val")
        self.assertEqual(config.unlabeled_split, "train")
        self.assertEqual(config.labeled_batch_size, 32)
        self.assertEqual(config.unlabeled_batch_size, 32)
        self.assertEqual((config.stage1_epochs, config.stage2_epochs), (25, 15))
        self.assertEqual((config.concept_loss_top_k, config.surrogate_top_k), (3, 4))
        self.assertEqual(config.selection_ratio, 0.3)
        self.assertEqual(config.weight_decay, 0.2)

    def test_config_rejects_test_leakage_and_empty_selection(self) -> None:
        values = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bad_split = dict(values, unlabeled_split="test")
            split_path = root / "bad_split.yaml"
            split_path.write_text(yaml.safe_dump(bad_split), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "test split"):
                load_semiclip_config(split_path)

            no_selection = dict(values, total_batch_size=2, selection_ratio=0.3)
            selection_path = root / "bad_selection.yaml"
            selection_path.write_text(yaml.safe_dump(no_selection), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least one"):
                load_semiclip_config(selection_path)

    def test_record_split_is_deterministic_and_whole_image_based(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            json_path = Path(temporary_directory) / "dataset.json"
            records = [record(f"image_{index}.jpg", f"caption {index}") for index in range(10)]
            records.append(record("train.jpg", "unused caption", split="train"))
            json_path.write_text(json.dumps({"images": records}), encoding="utf-8")

            first = split_rsicd_records(json_path, "val", 0.8, 42)
            second = split_rsicd_records(json_path, "val", 0.8, 42)

            self.assertEqual(first, second)
            self.assertEqual((len(first[0]), len(first[1])), (8, 2))
            self.assertTrue(
                {item["filename"] for item in first[0]}.isdisjoint(
                    item["filename"] for item in first[1]
                )
            )
            self.assertNotIn("train.jpg", {item["filename"] for part in first for item in part})

    @unittest.skipUnless(
        (PROJECT_ROOT / "dataset_rsicd.json").is_file(),
        "Local RSICD metadata is unavailable",
    )
    def test_local_labeled_partition_is_875_and_219(self) -> None:
        training, validation = split_rsicd_records(
            PROJECT_ROOT / "dataset_rsicd.json", "val", 0.8, 42
        )
        self.assertEqual((len(training), len(validation)), (875, 219))


class SemanticConceptMiningTests(unittest.TestCase):
    def test_mining_deduplicates_filters_and_builds_labels(self) -> None:
        records = [
            record("a.jpg", "airport airport road view", "airport crosses road"),
            record("b.jpg", "road airport car"),
            record("c.jpg", "car field"),
        ]

        result = mine_semantic_concepts(
            records,
            min_frequency=2,
            max_frequency_ratio=1.0,
            tokenize=fake_tokenize,
            tag=fake_tag,
        )

        self.assertEqual(result.concepts, ("airport", "road", "car"))
        self.assertEqual(result.counts, {"airport": 2, "road": 2, "car": 2})
        self.assertEqual(result.labels_by_filename["a.jpg"], (0, 1))
        self.assertEqual(result.labels_by_filename["b.jpg"], (1, 0, 2))
        self.assertEqual(result.labels_by_filename["c.jpg"], (2,))
        self.assertNotIn("view", result.concepts)

    def test_reference_frequency_denominator_and_strict_cutoff(self) -> None:
        records = [
            record("a.jpg", "airport road river field"),
            record("b.jpg", "airport"),
        ]

        result = mine_semantic_concepts(
            records,
            min_frequency=1,
            max_frequency_ratio=0.5,
            tokenize=fake_tokenize,
            tag=fake_tag,
        )

        self.assertNotIn("airport", result.concepts)
        self.assertEqual(result.concepts, ("road", "river", "field"))

    def test_artifact_is_stable_and_json_serializable(self) -> None:
        result = mine_semantic_concepts(
            [record("a.jpg", "airport road"), record("b.jpg", "airport field")],
            min_frequency=1,
            max_frequency_ratio=1.0,
            tokenize=fake_tokenize,
            tag=fake_tag,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "concepts.json"
            write_concept_artifact(path, result)
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved["concepts"], list(result.concepts))
        self.assertEqual(
            saved["source_filename_sha256"], hash_filenames(["a.jpg", "b.jpg"])
        )

    def test_invalid_or_empty_inputs_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            mine_semantic_concepts([], tokenize=fake_tokenize, tag=fake_tag)
        with self.assertRaisesRegex(ValueError, "supplied together"):
            mine_semantic_concepts(
                [record("a.jpg", "airport")], tokenize=fake_tokenize
            )
        with self.assertRaisesRegex(ValueError, "removed every"):
            mine_semantic_concepts(
                [record("a.jpg", "view")],
                min_frequency=1,
                tokenize=fake_tokenize,
                tag=fake_tag,
            )


if __name__ == "__main__":
    unittest.main()
