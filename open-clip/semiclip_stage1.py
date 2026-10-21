"""Stage-1 training and pseudo-concept caching for SemiCLIP."""

import json
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from transformers import get_scheduler

from semiclip_concepts import hash_filenames
from semiclip_config import SemiCLIPConfig
from semiclip_losses import stage1_loss_components
from semiclip_model import ConceptClassifier


def _progress(message: str) -> None:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{timestamp}] {message}", flush=True)


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("No batches were processed")
    return {
        name: sum(row[name] for row in rows) / len(rows)
        for name in rows[0]
    }


def _normalized_targets(targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim != 2 or torch.any(targets < 0):
        raise ValueError("Concept targets must be a nonnegative [batch, concepts] tensor")
    return targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _bounded_batches(loader: Iterable, maximum: int | None):
    if maximum is not None and maximum <= 0:
        raise ValueError("A batch limit must be positive")
    for index, batch in enumerate(loader):
        if maximum is not None and index >= maximum:
            break
        yield index, batch


def multi_caption_clip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    caption_to_image: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric validation loss with every source caption treated as positive."""
    if image_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError("Image and text features must be two-dimensional")
    if image_features.shape[1] != text_features.shape[1]:
        raise ValueError("Image and text features must share an embedding dimension")
    if caption_to_image.shape != (text_features.shape[0],):
        raise ValueError("caption_to_image must map every text feature")
    if caption_to_image.numel() == 0 or caption_to_image.min().item() < 0:
        raise ValueError("caption_to_image must contain valid image indices")
    if caption_to_image.max().item() >= image_features.shape[0]:
        raise ValueError("caption_to_image contains an out-of-range image index")

    images = functional.normalize(image_features, dim=-1)
    texts = functional.normalize(text_features, dim=-1)
    text_to_image = logit_scale * texts @ images.T
    text_loss = functional.cross_entropy(text_to_image, caption_to_image)
    image_to_text = text_to_image.T
    targets = functional.one_hot(
        caption_to_image, num_classes=image_features.shape[0]
    ).T.to(image_to_text.dtype)
    positives = targets.sum(dim=1, keepdim=True)
    if torch.any(positives == 0):
        raise ValueError("Every validation image must have at least one caption")
    targets = targets / positives
    image_loss = -(targets * functional.log_softmax(image_to_text, dim=1)).sum(1).mean()
    return (text_loss + image_loss) / 2


def create_stage1_optimizer_scheduler(
    model: nn.Module,
    classifier: ConceptClassifier,
    config: SemiCLIPConfig,
    total_steps: int,
) -> tuple[torch.optim.Optimizer, object]:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    named_parameters = [
        (f"model.{name}", parameter) for name, parameter in model.named_parameters()
    ]
    named_parameters.extend(
        (f"classifier.{name}", parameter)
        for name, parameter in classifier.named_parameters()
    )
    no_decay = []
    decay = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if (
            parameter.ndim < 2
            or "bn" in name.lower()
            or "ln" in name.lower()
            or "bias" in name.lower()
            or "logit_scale" in name.lower()
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    optimizer = torch.optim.AdamW(
        (
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
    )
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def train_stage1_epoch(
    model: nn.Module,
    classifier: ConceptClassifier,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    config: SemiCLIPConfig,
    epoch: int,
    max_batches: int | None = None,
) -> dict[str, object]:
    if hasattr(loader, "dataset") and hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)
    model.train()
    classifier.train()
    rows: list[dict[str, float]] = []
    filenames: list[str] = []
    total_batches = min(len(loader), max_batches or len(loader))
    for batch_index, batch in _bounded_batches(loader, max_batches):
        images = batch["images"].to(device)
        texts = batch["texts"].to(device)
        targets = _normalized_targets(batch["concept_targets"].to(device))
        filenames.extend(batch["filenames"])
        optimizer.zero_grad(set_to_none=True)
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        concept_logits = classifier(image_features)
        losses = stage1_loss_components(
            image_features,
            text_features,
            concept_logits,
            targets,
            model.logit_scale.exp(),
            clip_weight=config.clip_loss_weight,
            concept_weight=config.concept_loss_weight,
        )
        total = sum(losses.values())
        if not torch.isfinite(total):
            raise FloatingPointError(
                f"Non-finite Stage-1 loss at batch {batch_index}: {total.item()}"
            )
        total.backward()
        optimizer.step()
        with torch.no_grad():
            model.logit_scale.clamp_(0, 4.6052)
            classifier.logit_scale.clamp_(0, 4.6052)
        scheduler.step()
        rows.append(
            {
                **{name: float(value.detach()) for name, value in losses.items()},
                "total_loss": float(total.detach()),
            }
        )
        if (batch_index + 1) % 5 == 0 or batch_index + 1 == total_batches:
            _progress(
                f"Stage 1 epoch {epoch + 1}: batch {batch_index + 1}/{total_batches}, "
                f"loss={float(total.detach()):.4f}, "
                f"lr={optimizer.param_groups[0]['lr']:.3e}"
            )
    metrics: dict[str, object] = _mean_metrics(rows)
    metrics.update(
        {
            "epoch": epoch,
            "batches": len(rows),
            "images": len(filenames),
            "image_order_sha256": hash_filenames(filenames),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "logit_scale": float(model.logit_scale.detach().exp()),
            "concept_logit_scale": float(classifier.logit_scale.detach().exp()),
        }
    )
    return metrics


@torch.no_grad()
def validate_stage1(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, object]:
    model.eval()
    rows: list[dict[str, float]] = []
    filenames: list[str] = []
    for batch_index, batch in _bounded_batches(loader, max_batches):
        images = batch["images"].to(device)
        texts = batch["texts"].to(device)
        caption_to_image = batch["caption_to_image"].to(device)
        loss = multi_caption_clip_loss(
            model.encode_image(images),
            model.encode_text(texts),
            caption_to_image,
            model.logit_scale.exp(),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite Stage-1 validation loss at batch {batch_index}: {loss.item()}"
            )
        filenames.extend(batch["filenames"])
        rows.append({"clip_loss": float(loss)})
    metrics: dict[str, object] = _mean_metrics(rows)
    metrics.update(
        {
            "batches": len(rows),
            "images": len(filenames),
            "image_order_sha256": hash_filenames(filenames),
        }
    )
    return metrics


@torch.no_grad()
def cache_unlabeled_concept_logits(
    model: nn.Module,
    classifier: ConceptClassifier,
    loader: Iterable,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the fixed weak-view pseudo-concept source consumed by Stage 2."""
    model.eval()
    classifier.eval()
    cached: dict[str, torch.Tensor] = {}
    total_batches = min(len(loader), max_batches or len(loader))
    for batch_index, batch in _bounded_batches(loader, max_batches):
        logits = classifier(model.encode_image(batch["images"].to(device))).cpu()
        filenames = batch["filenames"]
        if logits.shape[0] != len(filenames):
            raise ValueError("The unlabeled batch has mismatched images and filenames")
        for filename, row in zip(filenames, logits):
            if filename in cached:
                raise ValueError(f"Duplicate unlabeled filename while caching: {filename}")
            cached[filename] = row.clone()
        if (batch_index + 1) % 25 == 0 or batch_index + 1 == total_batches:
            _progress(
                f"Stage 1 cache: batch {batch_index + 1}/{total_batches}, "
                f"images={len(cached)}"
            )
    if not cached:
        raise ValueError("No unlabeled images were processed for the Stage-1 cache")
    return cached


def _serialized_config(config: SemiCLIPConfig) -> dict[str, object]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in asdict(config).items()
    }


def stage1_checkpoint_state(
    model: nn.Module,
    classifier: ConceptClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    kind: str,
    concepts: Sequence[str],
    config: SemiCLIPConfig,
    history: list[dict[str, object]],
) -> dict[str, object]:
    if kind not in {"stage1_best", "stage1_final"}:
        raise ValueError("Unsupported Stage-1 checkpoint kind")
    state: dict[str, object] = {
        "checkpoint_kind": kind,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "concept_classifier_state_dict": classifier.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "concepts": list(concepts),
        "config": _serialized_config(config),
        "history": history,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_stage1_final(
    path: str | Path,
    model: nn.Module,
    classifier: ConceptClassifier,
) -> dict[str, object]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("checkpoint_kind") != "stage1_final":
        raise ValueError("Stage 2 requires the final Stage-1 checkpoint")
    for name in ("model_state_dict", "concept_classifier_state_dict", "concepts"):
        if name not in state:
            raise ValueError(f"Stage-1 checkpoint is missing {name}")
    model.load_state_dict(state["model_state_dict"])
    classifier.load_state_dict(state["concept_classifier_state_dict"])
    return state


def fit_stage1(
    model: nn.Module,
    classifier: ConceptClassifier,
    train_loader,
    validation_loader,
    unlabeled_loader,
    device: torch.device,
    config: SemiCLIPConfig,
    concepts: Sequence[str],
    output_dir: str | Path,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
    max_unlabeled_batches: int | None = None,
) -> dict[str, object]:
    """Run Stage 1 and materialize every artifact required by Stage 2."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batches_per_epoch = min(len(train_loader), max_train_batches or len(train_loader))
    if batches_per_epoch <= 0:
        raise ValueError("No Stage-1 training batches were selected")
    optimizer, scheduler = create_stage1_optimizer_scheduler(
        model, classifier, config, config.stage1_epochs * batches_per_epoch
    )
    _progress(
        f"Stage 1 starting: epochs={config.stage1_epochs}, "
        f"batches_per_epoch={batches_per_epoch}, concepts={len(concepts)}"
    )
    history: list[dict[str, object]] = []
    best_loss = float("inf")
    for epoch in range(config.stage1_epochs):
        epoch_started = time.perf_counter()
        _progress(f"Stage 1 epoch {epoch + 1}/{config.stage1_epochs} starting")
        train_metrics = train_stage1_epoch(
            model,
            classifier,
            train_loader,
            optimizer,
            scheduler,
            device,
            config,
            epoch,
            max_train_batches,
        )
        validation_metrics = validate_stage1(
            model, validation_loader, device, max_validation_batches
        )
        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_metrics)
        validation_loss = float(validation_metrics["clip_loss"])
        _progress(
            f"Stage 1 epoch {epoch + 1}/{config.stage1_epochs} complete in "
            f"{time.perf_counter() - epoch_started:.1f}s: "
            f"train={train_metrics['total_loss']:.4f}, "
            f"clip={train_metrics['clip_loss']:.4f}, "
            f"concept={train_metrics['concept_loss']:.4f}, "
            f"validation={validation_loss:.4f}"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            _progress(
                f"Stage 1 new best validation loss {best_loss:.4f}; "
                "saving stage1_best.pt"
            )
            torch.save(
                stage1_checkpoint_state(
                    model,
                    classifier,
                    optimizer,
                    scheduler,
                    epoch,
                    "stage1_best",
                    concepts,
                    config,
                    history,
                ),
                output_dir / "stage1_best.pt",
            )

    final_state = stage1_checkpoint_state(
        model,
        classifier,
        optimizer,
        scheduler,
        config.stage1_epochs - 1,
        "stage1_final",
        concepts,
        config,
        history,
    )
    final_path = output_dir / "stage1_final.pt"
    _progress("Stage 1 training complete; saving stage1_final.pt")
    torch.save(final_state, final_path)
    _progress("Stage 1 caching fixed logits for unlabeled images")
    cached = cache_unlabeled_concept_logits(
        model, classifier, unlabeled_loader, device, max_unlabeled_batches
    )
    cache_state = {
        "checkpoint_kind": "stage1_unlabeled_logits",
        "stage1_checkpoint": final_path.name,
        "concepts": list(concepts),
        "filenames": list(cached),
        "filename_sha256": hash_filenames(list(cached)),
        "logits_by_filename": cached,
        "config": _serialized_config(config),
    }
    torch.save(cache_state, output_dir / "stage1_unlabeled_logits.pt")
    summary = {
        "best_validation_loss": best_loss,
        "epochs": history,
        "unlabeled_images": len(cached),
        "unlabeled_filename_sha256": cache_state["filename_sha256"],
    }
    with (output_dir / "stage1_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    _progress(
        f"Stage 1 complete: best_validation={best_loss:.4f}, "
        f"cached_images={len(cached)}"
    )
    return summary
