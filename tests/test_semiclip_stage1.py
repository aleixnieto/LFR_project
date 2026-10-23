"""Numerical and artifact tests for the SemiCLIP Stage-1 workflow."""

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_CLIP_DIR = PROJECT_ROOT / "open-clip"
sys.path.insert(0, str(OPEN_CLIP_DIR))

from semiclip.config import load_semiclip_config  # noqa: E402
from semiclip.model import ConceptClassifier  # noqa: E402
from semiclip.stage1 import (  # noqa: E402
    cache_unlabeled_concept_logits,
    create_stage1_optimizer_scheduler,
    fit_stage1,
    multi_caption_clip_loss,
    restore_stage1_final,
    train_stage1_epoch,
    validate_stage1,
)


DEFAULT_CONFIG = OPEN_CLIP_DIR / "semiclip" / "configs" / "semiclip_rsicd.yaml"


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_projection = nn.Linear(4, 3, bias=False)
        self.token_embedding = nn.Embedding(32, 3)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_projection(images.reshape(images.shape[0], -1))

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(tokens).mean(dim=1)


class EpochDataset:
    def __init__(self) -> None:
        self.epoch = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class StaticLoader:
    def __init__(self, batches, dataset=None) -> None:
        self.batches = batches
        self.dataset = dataset

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def training_batch() -> dict[str, object]:
    return {
        "images": torch.tensor(
            [[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5]]
        ),
        "texts": torch.tensor([[1, 2], [3, 4]]),
        "caption_to_image": torch.tensor([0, 1]),
        "concept_targets": torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        "filenames": ["labeled_a.jpg", "labeled_b.jpg"],
    }


def validation_batch() -> dict[str, object]:
    return {
        "images": torch.tensor(
            [[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5]]
        ),
        "texts": torch.tensor([[1, 2], [2, 1], [3, 4]]),
        "caption_to_image": torch.tensor([0, 0, 1]),
        "filenames": ["validation_a.jpg", "validation_b.jpg"],
    }


def unlabeled_batch() -> dict[str, object]:
    return {
        "images": torch.tensor(
            [[0.5, 0.0, 1.0, 0.0], [0.0, 0.5, 0.0, 1.0]]
        ),
        "filenames": ["train_a.jpg", "train_b.jpg"],
    }


class Stage1NumericalTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.config = replace(
            load_semiclip_config(DEFAULT_CONFIG),
            stage1_epochs=2,
            learning_rate=1e-2,
            warmup_steps=1,
        )
        self.model = TinyModel()
        self.classifier = ConceptClassifier(torch.randn(3, 3), torch.tensor(0.0))

    def test_multi_caption_loss_matches_direct_formula(self) -> None:
        images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        texts = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
        mapping = torch.tensor([0, 0, 1])
        scale = torch.tensor(2.0)
        logits = scale * functional.normalize(texts, dim=1) @ images.T
        text_loss = functional.cross_entropy(logits, mapping)
        targets = torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
        image_loss = -(targets * functional.log_softmax(logits.T, dim=1)).sum(1).mean()

        actual = multi_caption_clip_loss(images, texts, mapping, scale)

        torch.testing.assert_close(actual, (text_loss + image_loss) / 2)

    def test_train_and_validation_report_components_and_update_model(self) -> None:
        dataset = EpochDataset()
        train_loader = StaticLoader([training_batch()], dataset)
        validation_loader = StaticLoader([validation_batch()])
        update_config = replace(self.config, warmup_steps=0)
        optimizer, scheduler = create_stage1_optimizer_scheduler(
            self.model, self.classifier, update_config, total_steps=2
        )
        before = self.model.image_projection.weight.detach().clone()

        train_metrics = train_stage1_epoch(
            self.model,
            self.classifier,
            train_loader,
            optimizer,
            scheduler,
            torch.device("cpu"),
            update_config,
            epoch=3,
        )
        validation_metrics = validate_stage1(
            self.model, validation_loader, torch.device("cpu")
        )

        self.assertEqual(dataset.epoch, 3)
        self.assertFalse(torch.equal(before, self.model.image_projection.weight))
        self.assertEqual(train_metrics["batches"], 1)
        self.assertEqual(train_metrics["images"], 2)
        self.assertEqual(train_metrics["captions"], 2)
        self.assertIn("clip_loss", train_metrics)
        self.assertIn("concept_loss", train_metrics)
        self.assertIn("image_order_sha256", train_metrics)
        self.assertEqual(validation_metrics["images"], 2)

    def test_optimizer_uses_configured_adamw_groups_and_cosine_schedule(self) -> None:
        optimizer, scheduler = create_stage1_optimizer_scheduler(
            self.model, self.classifier, self.config, total_steps=4
        )

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.98))
        self.assertEqual(optimizer.defaults["eps"], 1e-6)
        self.assertEqual(
            {group["weight_decay"] for group in optimizer.param_groups},
            {0.0, 0.2},
        )
        self.assertEqual(scheduler.state_dict()["last_epoch"], 0)


class Stage1ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.config = replace(
            load_semiclip_config(DEFAULT_CONFIG),
            stage1_epochs=2,
            learning_rate=1e-2,
            warmup_steps=1,
        )

    def test_fit_writes_final_best_metrics_and_filename_keyed_cache(self) -> None:
        model = TinyModel()
        classifier = ConceptClassifier(torch.randn(3, 3), torch.tensor(0.0))
        train_loader = StaticLoader([training_batch()], EpochDataset())
        validation_loader = StaticLoader([validation_batch()])
        unlabeled_loader = StaticLoader([unlabeled_batch()])

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            summary = fit_stage1(
                model,
                classifier,
                train_loader,
                validation_loader,
                unlabeled_loader,
                torch.device("cpu"),
                self.config,
                ["airport", "road", "river"],
                output_dir,
            )
            final = torch.load(
                output_dir / "stage1_final.pt", map_location="cpu", weights_only=False
            )
            best = torch.load(
                output_dir / "stage1_best.pt", map_location="cpu", weights_only=False
            )
            cache = torch.load(
                output_dir / "stage1_unlabeled_logits.pt",
                map_location="cpu",
                weights_only=False,
            )
            metrics = json.loads(
                (output_dir / "stage1_metrics.json").read_text(encoding="utf-8")
            )

            self.assertEqual(final["checkpoint_kind"], "stage1_final")
            self.assertEqual(final["epoch"], 1)
            self.assertEqual(best["checkpoint_kind"], "stage1_best")
            self.assertIn("optimizer_state_dict", final)
            self.assertIn("scheduler_state_dict", final)
            self.assertEqual(cache["stage1_checkpoint"], "stage1_final.pt")
            self.assertEqual(
                set(cache["logits_by_filename"]), {"train_a.jpg", "train_b.jpg"}
            )
            self.assertEqual(tuple(cache["logits_by_filename"]["train_a.jpg"].shape), (3,))
            self.assertEqual(summary["unlabeled_images"], 2)
            self.assertEqual(metrics["unlabeled_images"], 2)
            self.assertTrue(torch.isfinite(torch.tensor(summary["best_validation_loss"])))

            restored_model = TinyModel()
            restored_classifier = ConceptClassifier(
                torch.randn(3, 3), torch.tensor(0.0)
            )
            restored = restore_stage1_final(
                output_dir / "stage1_final.pt", restored_model, restored_classifier
            )
            self.assertEqual(restored["concepts"], ["airport", "road", "river"])
            for expected, actual in zip(model.parameters(), restored_model.parameters()):
                torch.testing.assert_close(expected, actual)
            with self.assertRaisesRegex(ValueError, "final"):
                restore_stage1_final(
                    output_dir / "stage1_best.pt", restored_model, restored_classifier
                )

    def test_cache_rejects_duplicate_filenames(self) -> None:
        model = TinyModel()
        classifier = ConceptClassifier(torch.randn(3, 3), torch.tensor(0.0))
        loader = StaticLoader([unlabeled_batch(), unlabeled_batch()])

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            cache_unlabeled_concept_logits(
                model, classifier, loader, torch.device("cpu")
            )


if __name__ == "__main__":
    unittest.main()
