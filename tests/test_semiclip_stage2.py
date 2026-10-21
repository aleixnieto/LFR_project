"""CPU tests for SemiCLIP Stage-2 preparation, training, and resume state."""

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_CLIP_DIR = PROJECT_ROOT / "open-clip"
sys.path.insert(0, str(OPEN_CLIP_DIR))

from semiclip.concepts import hash_filenames  # noqa: E402
from semiclip.config import load_semiclip_config  # noqa: E402
from semiclip.model import (  # noqa: E402
    ConceptClassifier,
    SurrogateCaptionEncoder,
    encode_text_embeddings,
)
from semiclip.stage2 import (  # noqa: E402
    create_stage2_optimizer_scheduler,
    fit_stage2,
    load_stage1_logit_cache,
    prepare_stage2,
    restore_stage2_checkpoint,
    stage2_checkpoint_state,
    train_stage2_epoch,
)


DEFAULT_CONFIG = OPEN_CLIP_DIR / "semiclip" / "configs" / "semiclip_rsicd.yaml"
CONCEPTS = ["airport", "road", "river", "field"]


class TinyTokenizer:
    context_length = 18

    def __init__(self) -> None:
        self.concept_ids = {name: 20 + index for index, name in enumerate(CONCEPTS)}

    def __call__(self, texts: list[str]) -> torch.Tensor:
        rows = []
        for text in texts:
            if text == "":
                content = []
            elif text == "a photo includes":
                content = [1, 2, 3]
            elif text in self.concept_ids:
                content = [self.concept_ids[text]]
            else:
                content = [30]
            row = [10, *content, 99]
            row.extend([0] * (self.context_length - len(row)))
            rows.append(row)
        return torch.tensor(rows, dtype=torch.long)


class MixingTransformer(nn.Module):
    batch_first = True

    def get_cast_dtype(self):
        return torch.float32

    def forward(self, values, attn_mask=None):
        return values + values.mean(dim=1, keepdim=True)


class TinyOpenCLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_projection = nn.Linear(4, 3, bias=False)
        self.token_embedding = nn.Embedding(128, 5)
        self.positional_embedding = nn.Parameter(torch.zeros(18, 5))
        self.transformer = MixingTransformer()
        self.attn_mask = None
        self.text_pool_type = "argmax"
        self.text_eos_id = None
        self.ln_final = nn.Identity()
        self.text_projection = nn.Parameter(torch.randn(5, 3))
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_projection(images.reshape(images.shape[0], -1))

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return encode_text_embeddings(self, tokens, self.token_embedding(tokens))


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


def labeled_batch(tokenizer: TinyTokenizer) -> dict[str, object]:
    return {
        "images": torch.tensor(
            [[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5]]
        ),
        "texts": tokenizer(["labeled one", "labeled two"]),
        "concept_targets": torch.zeros(2, len(CONCEPTS)),
        "filenames": ["labeled_a.jpg", "labeled_b.jpg"],
    }


def unlabeled_batch() -> dict[str, object]:
    weak = torch.tensor(
        [
            [0.9, 0.1, 0.5, 0.0],
            [0.1, 0.9, 0.0, 0.5],
            [0.7, 0.3, 0.4, 0.1],
            [0.3, 0.7, 0.1, 0.4],
        ]
    )
    return {
        "weak_images": weak,
        "strong_images": weak.flip(1),
        "filenames": [f"train_{index}.jpg" for index in range(4)],
    }


def validation_batch(tokenizer: TinyTokenizer) -> dict[str, object]:
    return {
        "images": torch.tensor(
            [[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5]]
        ),
        "texts": tokenizer(["validation a", "validation again", "validation b"]),
        "caption_to_image": torch.tensor([0, 0, 1]),
        "filenames": ["validation_a.jpg", "validation_b.jpg"],
    }


def cache_logits() -> dict[str, torch.Tensor]:
    return {
        f"train_{index}.jpg": torch.tensor(
            [4.0 - index * 0.1, 3.0 + index * 0.1, 2.0, 1.0]
        )
        for index in range(4)
    }


def write_stage1_artifacts(
    root: Path,
    model: TinyOpenCLIP,
    classifier: ConceptClassifier,
) -> tuple[Path, Path]:
    checkpoint_path = root / "stage1_final.pt"
    torch.save(
        {
            "checkpoint_kind": "stage1_final",
            "epoch": 24,
            "model_state_dict": model.state_dict(),
            "concept_classifier_state_dict": classifier.state_dict(),
            "concepts": CONCEPTS,
        },
        checkpoint_path,
    )
    cached = cache_logits()
    filenames = list(cached)
    cache_path = root / "stage1_unlabeled_logits.pt"
    torch.save(
        {
            "checkpoint_kind": "stage1_unlabeled_logits",
            "stage1_checkpoint": checkpoint_path.name,
            "concepts": CONCEPTS,
            "filenames": filenames,
            "filename_sha256": hash_filenames(filenames),
            "logits_by_filename": cached,
        },
        cache_path,
    )
    return checkpoint_path, cache_path


class Stage2Tests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        self.tokenizer = TinyTokenizer()
        self.config = replace(
            load_semiclip_config(DEFAULT_CONFIG),
            stage2_epochs=2,
            learning_rate=1e-2,
            warmup_steps=0,
        )

    def test_prepare_requires_final_checkpoint_and_matching_cache(self) -> None:
        source_model = TinyOpenCLIP()
        source_classifier = ConceptClassifier(
            torch.randn(len(CONCEPTS), 3), torch.tensor(0.0)
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_path, cache_path = write_stage1_artifacts(
                root, source_model, source_classifier
            )
            model = TinyOpenCLIP()
            classifier = ConceptClassifier(
                torch.randn(len(CONCEPTS), 3), torch.tensor(0.0)
            )

            prepared = prepare_stage2(
                model,
                classifier,
                self.tokenizer,
                self.config,
                checkpoint_path,
                cache_path,
                torch.device("cpu"),
            )

            self.assertEqual(tuple(prepared.prompt.context.shape), (4, 3, 5))
            self.assertEqual(set(prepared.cached_logits), set(cache_logits()))
            self.assertEqual(prepared.stage1_provenance["checkpoint_epoch"], 24)
            for expected, actual in zip(source_model.parameters(), model.parameters()):
                torch.testing.assert_close(expected, actual)

            bad_cache = torch.load(cache_path, weights_only=False)
            bad_cache["concepts"] = list(reversed(CONCEPTS))
            bad_path = root / "bad_cache.pt"
            torch.save(bad_cache, bad_path)
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                load_stage1_logit_cache(bad_path, CONCEPTS)

    def test_training_uses_distinct_topk_and_updates_prompt(self) -> None:
        model = TinyOpenCLIP()
        classifier = ConceptClassifier(
            torch.randn(len(CONCEPTS), 3), torch.tensor(0.0)
        )
        prompt = SurrogateCaptionEncoder(
            model, self.tokenizer, CONCEPTS, self.config.surrogate_top_k
        )
        labeled_dataset = EpochDataset()
        labeled_loader = StaticLoader(
            [labeled_batch(self.tokenizer), labeled_batch(self.tokenizer)],
            labeled_dataset,
        )
        unlabeled_loader = StaticLoader([unlabeled_batch()])
        optimizer, scheduler = create_stage2_optimizer_scheduler(
            model, classifier, prompt, self.config, total_steps=2
        )
        before = prompt.context.detach().clone()

        metrics = train_stage2_epoch(
            model,
            classifier,
            prompt,
            labeled_loader,
            unlabeled_loader,
            cache_logits(),
            optimizer,
            scheduler,
            torch.device("cpu"),
            self.config,
            epoch=5,
        )

        self.assertEqual(labeled_dataset.epoch, 5)
        self.assertEqual(metrics["batches"], 2)
        self.assertEqual(metrics["selected_pairs_total"], 2)
        self.assertEqual(metrics["selected_pairs"], 1.0)
        self.assertEqual(metrics["unlabeled_images"], 8)
        self.assertIn("concept_loss", metrics)
        self.assertIn("intramodal_loss", metrics)
        self.assertIn("crossmodal_loss", metrics)
        self.assertFalse(torch.equal(before, prompt.context))

    def test_fresh_optimizer_fit_artifacts_and_evaluator_model_state(self) -> None:
        source_model = TinyOpenCLIP()
        source_classifier = ConceptClassifier(
            torch.randn(len(CONCEPTS), 3), torch.tensor(0.0)
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_path, cache_path = write_stage1_artifacts(
                root, source_model, source_classifier
            )
            model = TinyOpenCLIP()
            classifier = ConceptClassifier(
                torch.randn(len(CONCEPTS), 3), torch.tensor(0.0)
            )
            prepared = prepare_stage2(
                model,
                classifier,
                self.tokenizer,
                self.config,
                checkpoint_path,
                cache_path,
                torch.device("cpu"),
            )
            optimizer, _scheduler = create_stage2_optimizer_scheduler(
                model, classifier, prepared.prompt, self.config, total_steps=2
            )
            self.assertEqual(optimizer.state, {})

            output_dir = root / "stage2"
            summary = fit_stage2(
                model,
                classifier,
                prepared.prompt,
                StaticLoader([labeled_batch(self.tokenizer)], EpochDataset()),
                StaticLoader([unlabeled_batch()]),
                StaticLoader([validation_batch(self.tokenizer)]),
                prepared.cached_logits,
                torch.device("cpu"),
                self.config,
                CONCEPTS,
                prepared.stage1_provenance,
                output_dir,
            )
            final = torch.load(
                output_dir / "stage2_final.pt", map_location="cpu", weights_only=False
            )
            last = torch.load(
                output_dir / "stage2_last.pt", map_location="cpu", weights_only=False
            )
            metrics = json.loads(
                (output_dir / "stage2_metrics.json").read_text(encoding="utf-8")
            )

            self.assertEqual(final["checkpoint_kind"], "stage2")
            self.assertEqual(final["artifact_role"], "final")
            self.assertEqual(final["epoch"], 1)
            self.assertEqual(last["artifact_role"], "last")
            self.assertIn("concept_classifier_state_dict", final)
            self.assertIn("prompt_state_dict", final)
            self.assertIn("optimizer_state_dict", final)
            self.assertEqual(summary["final_checkpoint"], "stage2_final.pt")
            self.assertEqual(metrics["final_checkpoint"], "stage2_final.pt")
            evaluator_model = TinyOpenCLIP()
            evaluator_model.load_state_dict(final["model_state_dict"])

    def test_checkpoint_restores_optimizer_scheduler_and_next_epoch(self) -> None:
        model = TinyOpenCLIP()
        classifier = ConceptClassifier(
            torch.randn(len(CONCEPTS), 3), torch.tensor(0.0)
        )
        prompt = SurrogateCaptionEncoder(model, self.tokenizer, CONCEPTS, 4)
        optimizer, scheduler = create_stage2_optimizer_scheduler(
            model, classifier, prompt, self.config, total_steps=2
        )
        train_stage2_epoch(
            model,
            classifier,
            prompt,
            StaticLoader([labeled_batch(self.tokenizer)], EpochDataset()),
            StaticLoader([unlabeled_batch()]),
            cache_logits(),
            optimizer,
            scheduler,
            torch.device("cpu"),
            self.config,
            epoch=0,
        )
        state = stage2_checkpoint_state(
            model,
            classifier,
            prompt,
            optimizer,
            scheduler,
            0,
            "last",
            CONCEPTS,
            self.config,
            [{"epoch": 0}],
            {"checkpoint_sha256": "abc"},
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "resume.pt"
            torch.save(state, path)
            fresh_optimizer, fresh_scheduler = create_stage2_optimizer_scheduler(
                model, classifier, prompt, self.config, total_steps=2
            )
            self.assertEqual(fresh_optimizer.state, {})

            restored = restore_stage2_checkpoint(
                path,
                model,
                classifier,
                prompt,
                fresh_optimizer,
                fresh_scheduler,
                restore_rng=False,
            )

            self.assertEqual(int(restored["epoch"]) + 1, 1)
            self.assertTrue(fresh_optimizer.state)
            self.assertEqual(
                fresh_scheduler.state_dict()["last_epoch"],
                scheduler.state_dict()["last_epoch"],
            )

    def test_training_rejects_filename_missing_from_fixed_cache(self) -> None:
        model = TinyOpenCLIP()
        classifier = ConceptClassifier(
            torch.randn(len(CONCEPTS), 3), torch.tensor(0.0)
        )
        prompt = SurrogateCaptionEncoder(model, self.tokenizer, CONCEPTS, 4)
        optimizer, scheduler = create_stage2_optimizer_scheduler(
            model, classifier, prompt, self.config, total_steps=1
        )
        incomplete = cache_logits()
        incomplete.pop("train_3.jpg")

        with self.assertRaisesRegex(ValueError, "missing"):
            train_stage2_epoch(
                model,
                classifier,
                prompt,
                StaticLoader([labeled_batch(self.tokenizer)], EpochDataset()),
                StaticLoader([unlabeled_batch()]),
                incomplete,
                optimizer,
                scheduler,
                torch.device("cpu"),
                self.config,
                epoch=0,
            )


if __name__ == "__main__":
    unittest.main()
