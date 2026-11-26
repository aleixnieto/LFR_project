"""Tests for the weighted k-NN evaluation of the DINO experiment."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_CLIP_DIR = PROJECT_ROOT / "open-clip"
DINO_DIR = OPEN_CLIP_DIR / "dino"
sys.path.insert(0, str(OPEN_CLIP_DIR))
sys.path.insert(0, str(DINO_DIR))

from knn_utils import (  # noqa: E402
    encode_labels,
    exact_mcnemar,
    knn_accuracy,
    knn_predict,
)


def reference_knn(reference, reference_labels, query, num_classes, k, temperature):
    """Slow, explicit version of DINO's weighted k-NN vote."""
    reference = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    query = query / np.linalg.norm(query, axis=1, keepdims=True)
    predictions = []
    for q in query:
        similarities = [float(q @ r) for r in reference]
        order = sorted(range(len(reference)), key=lambda i: (-similarities[i], i))[:k]
        scores = [0.0] * num_classes
        for i in order:
            scores[int(reference_labels[i])] += float(np.exp(similarities[i] / temperature))
        predictions.append(max(range(num_classes), key=lambda c: (scores[c], -c)))
    return np.array(predictions)


class KnnPredictTests(unittest.TestCase):
    def test_matches_explicit_implementation(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(size=(60, 16))
        labels = rng.integers(0, 5, size=60)
        query = rng.normal(size=(25, 16))
        for k in (1, 3, 7, 20):
            for temperature in (0.07, 0.5):
                with self.subTest(k=k, temperature=temperature):
                    np.testing.assert_array_equal(
                        knn_predict(reference, labels, query, 5, k, temperature),
                        reference_knn(reference, labels, query, 5, k, temperature),
                    )

    def test_chunking_does_not_change_predictions(self):
        rng = np.random.default_rng(1)
        reference = rng.normal(size=(40, 8))
        labels = rng.integers(0, 4, size=40)
        query = rng.normal(size=(33, 8))
        np.testing.assert_array_equal(
            knn_predict(reference, labels, query, 4, k=5, chunk_size=7),
            knn_predict(reference, labels, query, 4, k=5, chunk_size=1000),
        )

    def test_one_neighbor_returns_label_of_closest_reference(self):
        reference = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        labels = np.array([2, 0, 1])
        query = np.array([[0.9, 0.1], [0.1, 0.9], [-0.8, 0.1]])
        np.testing.assert_array_equal(knn_predict(reference, labels, query, 3, k=1), [2, 0, 1])

    def test_similarity_weights_can_overrule_the_majority(self):
        angle = np.deg2rad(60.0)
        reference = np.array(
            [[1.0, 0.0], [np.cos(angle), np.sin(angle)], [np.cos(angle), -np.sin(angle)]]
        )
        labels = np.array([0, 1, 1])
        query = np.array([[1.0, 0.02]])
        # One very close neighbor of class 0 against two distant neighbors of class 1.
        self.assertEqual(knn_predict(reference, labels, query, 2, k=3, temperature=0.07)[0], 0)
        # With a very high temperature all neighbors weigh the same and the majority wins.
        self.assertEqual(knn_predict(reference, labels, query, 2, k=3, temperature=1000.0)[0], 1)

    def test_k_larger_than_the_reference_set_is_reduced(self):
        rng = np.random.default_rng(2)
        reference = rng.normal(size=(6, 4))
        labels = np.array([0, 0, 1, 1, 2, 2])
        query = rng.normal(size=(5, 4))
        np.testing.assert_array_equal(
            knn_predict(reference, labels, query, 3, k=200),
            knn_predict(reference, labels, query, 3, k=6),
        )

    def test_feature_scale_does_not_matter(self):
        rng = np.random.default_rng(3)
        reference = rng.normal(size=(30, 6))
        labels = rng.integers(0, 3, size=30)
        query = rng.normal(size=(10, 6))
        np.testing.assert_array_equal(
            knn_predict(reference, labels, query, 3, k=5),
            knn_predict(7.5 * reference, labels, 0.1 * query, 3, k=5),
        )

    def test_accuracy_on_separated_clusters(self):
        rng = np.random.default_rng(4)
        centers = np.eye(4) * 10.0
        reference = np.concatenate([centers[c] + rng.normal(scale=0.1, size=(12, 4)) for c in range(4)])
        reference_labels = np.repeat(np.arange(4), 12)
        query = np.concatenate([centers[c] + rng.normal(scale=0.1, size=(5, 4)) for c in range(4)])
        query_labels = np.repeat(np.arange(4), 5)
        self.assertEqual(knn_accuracy(reference, reference_labels, query, query_labels, 4, k=5), 1.0)

    def test_invalid_inputs_are_rejected(self):
        reference = np.ones((4, 3))
        labels = np.array([0, 1, 0, 1])
        query = np.ones((2, 3))
        with self.assertRaises(ValueError):
            knn_predict(reference, labels, query, 2, k=0)
        with self.assertRaises(ValueError):
            knn_predict(reference, labels, query, 2, temperature=0.0)
        with self.assertRaises(ValueError):
            knn_predict(reference, labels, np.ones((2, 5)), 2)
        with self.assertRaises(ValueError):
            knn_predict(reference, np.array([0, 1, 2, 1]), query, 2)
        with self.assertRaises(ValueError):
            knn_predict(reference, labels[:3], query, 2)


class LabelAndTestHelpers(unittest.TestCase):
    def test_encode_labels_uses_sorted_names_by_default(self):
        indices, names = encode_labels(["Port", "Airport", "Port", "Beach"])
        self.assertEqual(names, ["Airport", "Beach", "Port"])
        np.testing.assert_array_equal(indices, [2, 0, 2, 1])

    def test_encode_labels_keeps_a_given_class_order(self):
        indices, names = encode_labels(["b", "a"], class_names=["b", "a", "c"])
        self.assertEqual(names, ["b", "a", "c"])
        np.testing.assert_array_equal(indices, [0, 1])
        with self.assertRaises(ValueError):
            encode_labels(["d"], class_names=["a", "b"])

    def test_exact_mcnemar_known_values(self):
        self.assertAlmostEqual(exact_mcnemar(80, 72), 0.570333, places=6)
        self.assertAlmostEqual(exact_mcnemar(33, 78) / 2.32627e-05, 1.0, places=4)
        self.assertAlmostEqual(exact_mcnemar(0, 5), 0.0625, places=12)
        self.assertEqual(exact_mcnemar(5, 5), 1.0)
        self.assertEqual(exact_mcnemar(0, 0), 1.0)
        self.assertEqual(exact_mcnemar(3, 9), exact_mcnemar(9, 3))


try:
    import torch
    from sklearn.model_selection import train_test_split

    from dataloader import split_rsicd_records
    from validate_features import evaluate_knn, evaluate_linear_probe, labeled_partition

    HAVE_TORCH = True
except ImportError:  # pragma: no cover - the training environment always has torch
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch and the project data loader are required")
class PartitionAndMonitorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.class_names = ["Airport", "Beach", "Port"]
        self.filenames = [f"img_{i:03d}.jpg" for i in range(60)]
        self.labels = np.array([self.class_names[i % 3] for i in range(60)])
        centers = {"Airport": [5.0, 0.0, 0.0], "Beach": [0.0, 5.0, 0.0], "Port": [0.0, 0.0, 5.0]}
        self.features = np.array(
            [np.array(centers[label]) + rng.normal(scale=0.05, size=3) for label in self.labels]
        ).astype(np.float32)

        self.tmp = tempfile.TemporaryDirectory()
        self.json_path = Path(self.tmp.name) / "dataset.json"
        records = [{"filename": name, "split": "val"} for name in self.filenames]
        records += [{"filename": f"other_{i}.jpg", "split": "train"} for i in range(5)]
        self.json_path.write_text(json.dumps({"images": records}), encoding="utf-8")

        # A fake encoder that returns the stored feature of each "image" (its index).
        features = torch.tensor(self.features)

        class Encoder(torch.nn.Module):
            def forward(self, indices):
                return features[indices.long().view(-1)]

        class Model:
            student = Encoder()

        self.model = Model()
        self.loader = [
            (
                torch.arange(start, min(start + 16, 60)),
                list(self.labels[start:start + 16]),
                self.filenames[start:start + 16],
            )
            for start in range(0, 60, 16)
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def test_stratified_partition_equals_the_original_probe_split(self):
        reference_idx, heldout_idx = labeled_partition(
            self.filenames, self.labels, split_mode="stratified", seed=42
        )
        expected_reference, expected_heldout = train_test_split(
            np.arange(60), test_size=0.2, random_state=42, stratify=self.labels
        )
        np.testing.assert_array_equal(reference_idx, expected_reference)
        np.testing.assert_array_equal(heldout_idx, expected_heldout)

    def test_project_partition_equals_split_rsicd_records(self):
        reference_idx, heldout_idx = labeled_partition(
            self.filenames, self.labels, split_mode="project", json_path=self.json_path, seed=42
        )
        train_records, val_records = split_rsicd_records(self.json_path, "val", 0.8, 42)
        self.assertEqual(
            {self.filenames[i] for i in reference_idx}, {r["filename"] for r in train_records}
        )
        self.assertEqual(
            {self.filenames[i] for i in heldout_idx}, {r["filename"] for r in val_records}
        )
        self.assertEqual(len(reference_idx), 48)
        self.assertEqual(len(heldout_idx), 12)
        self.assertEqual(set(reference_idx) & set(heldout_idx), set())

    def test_project_partition_rejects_images_from_another_split(self):
        with self.assertRaises(ValueError):
            labeled_partition(
                self.filenames[:-1] + ["other_0.jpg"], self.labels,
                split_mode="project", json_path=self.json_path,
            )
        with self.assertRaises(ValueError):
            labeled_partition(self.filenames, self.labels, split_mode="project")
        with self.assertRaises(ValueError):
            labeled_partition(self.filenames, self.labels, split_mode="unknown")

    def test_knn_monitor_on_separated_classes(self):
        for split_mode in ("project", "stratified"):
            with self.subTest(split_mode=split_mode):
                accuracy = evaluate_knn(
                    self.model, self.loader, torch.device("cpu"), k=5,
                    split_mode=split_mode, json_path=self.json_path,
                )
                self.assertEqual(accuracy, 1.0)

    def test_knn_monitor_detects_a_useless_representation(self):
        constant = torch.ones(60, 3)

        class Encoder(torch.nn.Module):
            def forward(self, indices):
                return constant[indices.long().view(-1)]

        class Model:
            student = Encoder()

        accuracy = evaluate_knn(
            Model(), self.loader, torch.device("cpu"), k=5,
            split_mode="stratified",
        )
        self.assertLess(accuracy, 0.5)

    def test_linear_probe_still_works_and_keeps_its_default_split(self):
        accuracy = evaluate_linear_probe(self.model, self.loader, torch.device("cpu"))
        self.assertEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
