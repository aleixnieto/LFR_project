"""Numerically explicit losses for both SemiCLIP training stages."""

import math

import torch
import torch.nn.functional as functional


def clip_contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric one-to-one CLIP loss used by the released SemiCLIP code."""
    if image_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError("CLIP features must be two-dimensional")
    if image_features.shape != text_features.shape or image_features.shape[0] == 0:
        raise ValueError("SemiCLIP requires equal nonempty image and text feature batches")
    image_features = functional.normalize(image_features, dim=-1)
    text_features = functional.normalize(text_features, dim=-1)
    logits = logit_scale * image_features @ text_features.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (
        functional.cross_entropy(logits, labels)
        + functional.cross_entropy(logits.T, labels)
    ) / 2


def soft_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("logits and targets must have the same [batch, classes] shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return -(targets * functional.log_softmax(logits / temperature, dim=1)).sum(dim=1).mean()


def normalized_multihot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    if indices.ndim != 2 or indices.shape[1] == 0:
        raise ValueError("indices must be a nonempty [batch, top_k] tensor")
    if num_classes <= 0 or indices.min().item() < 0 or indices.max().item() >= num_classes:
        raise ValueError("indices must refer to valid classes")
    targets = functional.one_hot(indices, num_classes=num_classes).sum(dim=1).float()
    return targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-8)


def pseudo_concept_targets(
    cached_logits: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cached_logits.ndim != 2 or cached_logits.shape[0] == 0:
        raise ValueError("cached_logits must be a nonempty [batch, concepts] tensor")
    if top_k <= 0 or top_k > cached_logits.shape[1]:
        raise ValueError("top_k must fit the concept dimension")
    indices = torch.topk(functional.softmax(cached_logits, dim=1), top_k, dim=1).indices
    return normalized_multihot(indices, cached_logits.shape[1]), indices


def select_surrogate_pairs(
    image_features: torch.Tensor,
    surrogate_features: torch.Tensor,
    selection_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the batch-local top fraction with stable input-order tie breaking."""
    if image_features.shape != surrogate_features.shape or image_features.ndim != 2:
        raise ValueError("Image and surrogate features must have the same [batch, dim] shape")
    if not 0 < selection_ratio <= 1:
        raise ValueError("selection_ratio must be positive and at most one")
    count = math.floor(image_features.shape[0] * selection_ratio)
    if count < 1:
        raise ValueError("selection_ratio retains no unlabeled samples")
    similarities = (
        functional.normalize(image_features, dim=-1)
        * functional.normalize(surrogate_features, dim=-1)
    ).sum(dim=1)
    selected = torch.argsort(similarities, descending=True, stable=True)[:count]
    return selected, similarities


def trapezoidal_consistency_losses(
    labeled_image_features: torch.Tensor,
    labeled_text_features: torch.Tensor,
    selected_unlabeled_features: torch.Tensor,
    selected_surrogate_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if labeled_image_features.shape != labeled_text_features.shape:
        raise ValueError("Labeled image and text features must have matching shapes")
    if selected_unlabeled_features.shape != selected_surrogate_features.shape:
        raise ValueError("Selected image and surrogate features must have matching shapes")
    if labeled_image_features.ndim != 2 or selected_unlabeled_features.ndim != 2:
        raise ValueError("All trapezoidal features must be two-dimensional")
    image_mix = functional.normalize(
        torch.cat((labeled_image_features, selected_unlabeled_features)), dim=-1
    )
    text_mix = functional.normalize(
        torch.cat((labeled_text_features, selected_surrogate_features)), dim=-1
    )
    image_similarity = logit_scale * image_mix @ image_mix.T
    text_similarity = logit_scale * text_mix @ text_mix.T
    cross_similarity = logit_scale * image_mix @ text_mix.T
    scale_squared = logit_scale.square().clamp_min(1e-8)
    reference_scale = labeled_image_features.shape[0]
    intramodal = (
        (image_similarity - text_similarity).square().mean()
        / scale_squared
        * reference_scale
    )
    crossmodal = (
        (cross_similarity - cross_similarity.T).square().mean()
        / scale_squared
        * reference_scale
    )
    return intramodal, crossmodal


def stage1_loss_components(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    concept_logits: torch.Tensor,
    concept_targets: torch.Tensor,
    logit_scale: torch.Tensor,
    clip_weight: float = 1.0,
    concept_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    return {
        "clip_loss": clip_contrastive_loss(
            image_features, text_features, logit_scale
        )
        * clip_weight,
        "concept_loss": soft_cross_entropy(concept_logits, concept_targets)
        * concept_weight,
    }


def stage2_loss_components(
    labeled_image_features: torch.Tensor,
    labeled_text_features: torch.Tensor,
    strong_concept_logits: torch.Tensor,
    cached_concept_logits: torch.Tensor,
    unlabeled_image_features: torch.Tensor,
    surrogate_features: torch.Tensor,
    logit_scale: torch.Tensor,
    concept_top_k: int,
    selection_ratio: float,
    clip_weight: float = 1.0,
    concept_weight: float = 1.0,
    intramodal_weight: float = 1.0,
    crossmodal_weight: float = 1.0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    pseudo_targets, pseudo_indices = pseudo_concept_targets(
        cached_concept_logits, concept_top_k
    )
    selected, similarities = select_surrogate_pairs(
        unlabeled_image_features, surrogate_features, selection_ratio
    )
    intramodal, crossmodal = trapezoidal_consistency_losses(
        labeled_image_features,
        labeled_text_features,
        unlabeled_image_features[selected],
        surrogate_features[selected],
        logit_scale,
    )
    losses = {
        "clip_loss": clip_contrastive_loss(
            labeled_image_features, labeled_text_features, logit_scale
        )
        * clip_weight,
        "concept_loss": soft_cross_entropy(
            strong_concept_logits, pseudo_targets
        )
        * concept_weight,
        "intramodal_loss": intramodal * intramodal_weight,
        "crossmodal_loss": crossmodal * crossmodal_weight,
    }
    details = {
        "pseudo_indices": pseudo_indices,
        "selected_indices": selected,
        "surrogate_similarities": similarities,
    }
    return losses, details
