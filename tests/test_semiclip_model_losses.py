"""Numerical tests for SemiCLIP model additions and both-stage losses."""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_CLIP_DIR = PROJECT_ROOT / "open-clip"
sys.path.insert(0, str(OPEN_CLIP_DIR))

from semiclip.losses import (  # noqa: E402
    clip_contrastive_loss,
    normalized_multihot,
    pseudo_concept_targets,
    select_surrogate_pairs,
    soft_cross_entropy,
    stage1_loss_components,
    stage2_loss_components,
    trapezoidal_consistency_losses,
)
from semiclip.model import (  # noqa: E402
    ConceptClassifier,
    SurrogateCaptionEncoder,
    encode_text_embeddings,
)


class FakeTokenizer:
    context_length = 18

    def __init__(self) -> None:
        self.concept_ids = {"airport": 20, "road": 21, "river": 22, "field": 23}

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
                content = [self.concept_ids.get(text.split()[-1].rstrip("."), 30)]
            row = [10, *content, 99]
            row.extend([0] * (self.context_length - len(row)))
            rows.append(row)
        return torch.tensor(rows, dtype=torch.long)


class MixingTransformer(nn.Module):
    batch_first = True

    def get_cast_dtype(self):
        return torch.float32

    def forward(self, x, attn_mask=None):
        return x + x.mean(dim=1, keepdim=True)


class FakeTextModel(nn.Module):
    def __init__(self, width: int = 5, output_dim: int = 3) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(128, width)
        self.positional_embedding = nn.Parameter(torch.zeros(18, width))
        self.transformer = MixingTransformer()
        self.attn_mask = None
        self.text_pool_type = "argmax"
        self.text_eos_id = None
        self.ln_final = nn.Identity()
        self.text_projection = nn.Parameter(torch.randn(width, output_dim))
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def encode_text(self, token_ids):
        return encode_text_embeddings(self, token_ids, self.token_embedding(token_ids))


class SemiCLIPModelTests(unittest.TestCase):
    def test_concept_classifier_initializes_normalized_trainable_weights(self) -> None:
        features = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
        classifier = ConceptClassifier(features, torch.tensor(0.0))
        logits = classifier(torch.tensor([[3.0, 4.0]]))

        torch.testing.assert_close(classifier.weight.norm(dim=1), torch.ones(2))
        torch.testing.assert_close(logits, torch.tensor([[1.0, 0.8]]))
        logits.sum().backward()
        self.assertIsNotNone(classifier.weight.grad)
        self.assertIsNotNone(classifier.logit_scale.grad)

    def test_surrogate_caption_layout_is_tokenizer_and_width_aware(self) -> None:
        torch.manual_seed(3)
        model = FakeTextModel(width=5, output_dim=3)
        encoder = SurrogateCaptionEncoder(
            model, FakeTokenizer(), ["airport", "road", "river", "field"], top_k=4
        )
        indices = torch.tensor([[3, 1, 0, 2]])

        token_ids, context_positions = encoder.build_token_ids(indices)
        features = encoder(model, indices)

        self.assertEqual(tuple(encoder.context.shape), (4, 3, 5))
        self.assertEqual(context_positions, [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15])
        torch.testing.assert_close(token_ids[0, [4, 8, 12, 16]], torch.tensor([23, 21, 20, 22]))
        self.assertEqual(int(token_ids[0, 17]), 99)
        self.assertEqual(tuple(features.shape), (1, 3))
        features.square().sum().backward()
        self.assertIsNotNone(encoder.context.grad)
        self.assertGreater(float(encoder.context.grad.abs().sum()), 0.0)

    def test_surrogate_indices_are_validated(self) -> None:
        model = FakeTextModel()
        encoder = SurrogateCaptionEncoder(
            model, FakeTokenizer(), ["airport", "road", "river", "field"], top_k=2
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            encoder.build_token_ids(torch.tensor([[0, 1, 2]]))
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            encoder.build_token_ids(torch.tensor([[0, 4]]))


class SemiCLIPLossTests(unittest.TestCase):
    def test_clip_and_soft_losses_match_direct_formulas(self) -> None:
        images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        texts = torch.tensor([[0.8, 0.6], [0.0, 1.0]])
        scale = torch.tensor(2.0)
        logits = scale * images @ functional.normalize(texts, dim=-1).T
        labels = torch.arange(2)
        expected_clip = (
            functional.cross_entropy(logits, labels)
            + functional.cross_entropy(logits.T, labels)
        ) / 2
        targets = torch.tensor([[0.5, 0.5], [0.0, 1.0]])
        expected_soft = -(targets * functional.log_softmax(logits, dim=1)).sum(1).mean()

        torch.testing.assert_close(
            clip_contrastive_loss(images, texts, scale), expected_clip
        )
        torch.testing.assert_close(soft_cross_entropy(logits, targets), expected_soft)

    def test_topk_pseudo_targets_are_normalized_multihot(self) -> None:
        cached = torch.tensor([[4.0, 3.0, 2.0, 1.0], [0.0, 2.0, 3.0, 1.0]])
        targets, indices = pseudo_concept_targets(cached, top_k=3)

        torch.testing.assert_close(indices, torch.tensor([[0, 1, 2], [2, 1, 3]]))
        torch.testing.assert_close(targets.sum(dim=1), torch.ones(2))
        torch.testing.assert_close(targets, normalized_multihot(indices, 4))

    def test_selection_uses_batch_fraction_and_stable_ties(self) -> None:
        images = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        surrogates = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

        selected, similarities = select_surrogate_pairs(images, surrogates, 0.67)

        torch.testing.assert_close(selected, torch.tensor([0, 1]))
        torch.testing.assert_close(similarities, torch.tensor([1.0, 1.0, 0.0]))

    def test_trapezoidal_losses_are_zero_for_identical_modalities(self) -> None:
        labeled = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        selected = torch.tensor([[1.0, 1.0]])

        intramodal, crossmodal = trapezoidal_consistency_losses(
            labeled, labeled.clone(), selected, selected.clone(), torch.tensor(3.0)
        )

        torch.testing.assert_close(intramodal, torch.tensor(0.0))
        torch.testing.assert_close(crossmodal, torch.tensor(0.0))

    def test_stage_loss_components_are_separate_and_differentiable(self) -> None:
        labeled_images = functional.normalize(torch.randn(2, 4), dim=-1).requires_grad_()
        labeled_texts = functional.normalize(torch.randn(2, 4), dim=-1).requires_grad_()
        unlabeled = functional.normalize(torch.randn(4, 4), dim=-1).requires_grad_()
        surrogates = functional.normalize(torch.randn(4, 4), dim=-1).requires_grad_()
        strong_logits = torch.randn(4, 5, requires_grad=True)
        cached_logits = torch.randn(4, 5)
        scale = torch.tensor(2.0, requires_grad=True)

        stage1 = stage1_loss_components(
            labeled_images,
            labeled_texts,
            strong_logits[:2],
            torch.tensor([[0.5, 0.5, 0, 0, 0], [0, 0, 0.5, 0.5, 0]]),
            scale,
        )
        self.assertEqual(set(stage1), {"clip_loss", "concept_loss"})

        stage2, details = stage2_loss_components(
            labeled_images,
            labeled_texts,
            strong_logits,
            cached_logits,
            unlabeled,
            surrogates,
            scale,
            concept_top_k=3,
            selection_ratio=0.5,
        )
        self.assertEqual(
            set(stage2),
            {"clip_loss", "concept_loss", "intramodal_loss", "crossmodal_loss"},
        )
        self.assertEqual(details["selected_indices"].numel(), 2)
        sum(stage2.values()).backward()
        self.assertIsNotNone(strong_logits.grad)
        self.assertIsNotNone(unlabeled.grad)
        self.assertIsNotNone(surrogates.grad)


if __name__ == "__main__":
    unittest.main()
