"""Stage-2 semantic and trapezoidal consistency training for SemiCLIP."""

import hashlib
import json
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from transformers import get_scheduler

from semiclip_concepts import hash_filenames
from semiclip_config import SemiCLIPConfig
from semiclip_losses import stage2_loss_components
from semiclip_model import ConceptClassifier, SurrogateCaptionEncoder
from semiclip_stage1 import _progress, restore_stage1_final, validate_stage1


@dataclass(frozen=True)
class PreparedStage2:
    prompt: SurrogateCaptionEncoder
    cached_logits: dict[str, torch.Tensor]
    stage1_provenance: dict[str, object]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serialized_config(config: SemiCLIPConfig) -> dict[str, object]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in asdict(config).items()
    }


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("No batches were processed")
    return {
        name: sum(row[name] for row in rows) / len(rows)
        for name in rows[0]
    }


def _bounded_batches(loader: Iterable, maximum: int | None):
    if maximum is not None and maximum <= 0:
        raise ValueError("A batch limit must be positive")
    for index, batch in enumerate(loader):
        if maximum is not None and index >= maximum:
            break
        yield index, batch


def load_stage1_logit_cache(
    path: str | Path,
    expected_concepts: Sequence[str],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("checkpoint_kind") != "stage1_unlabeled_logits":
        raise ValueError("Stage 2 requires a Stage-1 unlabeled-logit cache")
    if state.get("concepts") != list(expected_concepts):
        raise ValueError("Stage-1 cache vocabulary does not match the checkpoint")
    raw_cache = state.get("logits_by_filename")
    if not isinstance(raw_cache, dict) or not raw_cache:
        raise ValueError("Stage-1 cache has no filename-keyed logits")
    cached: dict[str, torch.Tensor] = {}
    for filename, logits in raw_cache.items():
        if not isinstance(filename, str) or not filename:
            raise ValueError("Stage-1 cache contains an invalid filename")
        if not isinstance(logits, torch.Tensor) or logits.shape != (
            len(expected_concepts),
        ):
            raise ValueError(f"Invalid cached concept logits for {filename!r}")
        cached[filename] = logits.detach().cpu().clone()
    declared_filenames = state.get("filenames")
    if declared_filenames != list(cached):
        raise ValueError("Stage-1 cache filename order is inconsistent")
    if state.get("filename_sha256") != hash_filenames(list(cached)):
        raise ValueError("Stage-1 cache filename hash is inconsistent")
    return cached, state


def prepare_stage2(
    model: nn.Module,
    classifier: ConceptClassifier,
    tokenizer,
    config: SemiCLIPConfig,
    stage1_checkpoint_path: str | Path,
    stage1_cache_path: str | Path,
    device: torch.device,
) -> PreparedStage2:
    """Restore final Stage 1 before initializing the learnable prompt."""
    checkpoint_path = Path(stage1_checkpoint_path)
    cache_path = Path(stage1_cache_path)
    _progress(f"Stage 2 loading final Stage-1 checkpoint: {checkpoint_path}")
    stage1_state = restore_stage1_final(checkpoint_path, model, classifier)
    concepts = stage1_state["concepts"]
    if config.concept_loss_top_k > len(concepts):
        raise ValueError("concept_loss_top_k exceeds the Stage-1 vocabulary")
    if config.surrogate_top_k > len(concepts):
        raise ValueError("surrogate_top_k exceeds the Stage-1 vocabulary")
    _progress(f"Stage 2 loading fixed unlabeled cache: {cache_path}")
    cached, cache_state = load_stage1_logit_cache(cache_path, concepts)
    if cache_state.get("stage1_checkpoint") != checkpoint_path.name:
        raise ValueError("Stage-1 cache does not identify the supplied final checkpoint")
    model.to(device)
    classifier.to(device)
    prompt = SurrogateCaptionEncoder(
        model, tokenizer, concepts, config.surrogate_top_k
    ).to(device)
    _progress(
        f"Stage 2 inputs ready: concepts={len(concepts)}, "
        f"cached_images={len(cached)}"
    )
    provenance = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_epoch": stage1_state["epoch"],
        "cache_path": str(cache_path.resolve()),
        "cache_sha256": _sha256_file(cache_path),
        "cache_filename_sha256": cache_state["filename_sha256"],
    }
    return PreparedStage2(prompt, cached, provenance)


def create_stage2_optimizer_scheduler(
    model: nn.Module,
    classifier: ConceptClassifier,
    prompt: SurrogateCaptionEncoder,
    config: SemiCLIPConfig,
    total_steps: int,
) -> tuple[torch.optim.Optimizer, object]:
    """Create a fresh Stage-2 optimizer; Stage-1 optimizer state is never reused."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    named_parameters = [
        (f"model.{name}", parameter) for name, parameter in model.named_parameters()
    ]
    named_parameters.extend(
        (f"classifier.{name}", parameter)
        for name, parameter in classifier.named_parameters()
    )
    named_parameters.extend(
        (f"prompt.{name}", parameter) for name, parameter in prompt.named_parameters()
    )
    decay = []
    no_decay = []
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


def _cached_batch(
    filenames: Sequence[str],
    cached_logits: Mapping[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    missing = [filename for filename in filenames if filename not in cached_logits]
    if missing:
        raise ValueError(f"Stage-1 cache is missing unlabeled images: {missing[:3]}")
    return torch.stack([cached_logits[name] for name in filenames]).to(device)


def train_stage2_epoch(
    model: nn.Module,
    classifier: ConceptClassifier,
    prompt: SurrogateCaptionEncoder,
    labeled_loader: Iterable,
    unlabeled_loader: Iterable,
    cached_logits: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    config: SemiCLIPConfig,
    epoch: int,
    max_batches: int | None = None,
) -> dict[str, object]:
    """Pair one labeled and unlabeled half-batch per labeled-loader update."""
    if hasattr(labeled_loader, "dataset") and hasattr(
        labeled_loader.dataset, "set_epoch"
    ):
        labeled_loader.dataset.set_epoch(epoch)
    model.train()
    classifier.train()
    prompt.train()
    rows: list[dict[str, float]] = []
    labeled_filenames: list[str] = []
    unlabeled_filenames: list[str] = []
    total_batches = min(len(labeled_loader), max_batches or len(labeled_loader))
    unlabeled_iterator = iter(unlabeled_loader)
    for batch_index, labeled in _bounded_batches(labeled_loader, max_batches):
        try:
            unlabeled = next(unlabeled_iterator)
        except StopIteration:
            unlabeled_iterator = iter(unlabeled_loader)
            try:
                unlabeled = next(unlabeled_iterator)
            except StopIteration as error:
                raise ValueError("The Stage-2 unlabeled loader is empty") from error

        filenames = unlabeled["filenames"]
        fixed_logits = _cached_batch(filenames, cached_logits, device)
        if config.surrogate_top_k > fixed_logits.shape[1]:
            raise ValueError("surrogate_top_k exceeds cached concept logits")
        surrogate_indices = torch.topk(
            fixed_logits, config.surrogate_top_k, dim=1
        ).indices

        images = labeled["images"].to(device)
        texts = labeled["texts"].to(device)
        weak_images = unlabeled["weak_images"].to(device)
        strong_images = unlabeled["strong_images"].to(device)
        labeled_filenames.extend(labeled["filenames"])
        unlabeled_filenames.extend(filenames)

        optimizer.zero_grad(set_to_none=True)
        labeled_image_features = model.encode_image(images)
        labeled_text_features = model.encode_text(texts)
        weak_features = model.encode_image(weak_images)
        strong_features = model.encode_image(strong_images)
        strong_concept_logits = classifier(strong_features)
        surrogate_features = prompt(model, surrogate_indices)
        losses, details = stage2_loss_components(
            labeled_image_features,
            labeled_text_features,
            strong_concept_logits,
            fixed_logits,
            weak_features,
            surrogate_features,
            model.logit_scale.exp(),
            concept_top_k=config.concept_loss_top_k,
            selection_ratio=config.selection_ratio,
            clip_weight=config.clip_loss_weight,
            concept_weight=config.concept_loss_weight,
            intramodal_weight=config.intramodal_loss_weight,
            crossmodal_weight=config.crossmodal_loss_weight,
        )
        total = sum(losses.values())
        if not torch.isfinite(total):
            raise FloatingPointError(
                f"Non-finite Stage-2 loss at batch {batch_index}: {total.item()}"
            )
        total.backward()
        optimizer.step()
        with torch.no_grad():
            model.logit_scale.clamp_(0, 4.6052)
            classifier.logit_scale.clamp_(0, 4.6052)
        scheduler.step()
        top_probabilities = functional.softmax(fixed_logits, dim=1).gather(
            1, details["pseudo_indices"]
        )
        rows.append(
            {
                **{name: float(value.detach()) for name, value in losses.items()},
                "total_loss": float(total.detach()),
                "selected_pairs": float(details["selected_indices"].numel()),
                "surrogate_similarity": float(
                    details["surrogate_similarities"].mean().detach()
                ),
                "pseudo_concept_probability": float(top_probabilities.mean().detach()),
            }
        )
        if (batch_index + 1) % 5 == 0 or batch_index + 1 == total_batches:
            _progress(
                f"Stage 2 epoch {epoch + 1}: batch {batch_index + 1}/{total_batches}, "
                f"loss={float(total.detach()):.4f}, "
                f"selected={details['selected_indices'].numel()}, "
                f"lr={optimizer.param_groups[0]['lr']:.3e}"
            )
    metrics: dict[str, object] = _mean_metrics(rows)
    metrics.update(
        {
            "epoch": epoch,
            "batches": len(rows),
            "labeled_images": len(labeled_filenames),
            "unlabeled_images": len(unlabeled_filenames),
            "selected_pairs_total": int(
                sum(row["selected_pairs"] for row in rows)
            ),
            "labeled_image_order_sha256": hash_filenames(labeled_filenames),
            "unlabeled_image_order_sha256": hash_filenames(unlabeled_filenames),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "logit_scale": float(model.logit_scale.detach().exp()),
            "concept_logit_scale": float(classifier.logit_scale.detach().exp()),
        }
    )
    return metrics


def stage2_checkpoint_state(
    model: nn.Module,
    classifier: ConceptClassifier,
    prompt: SurrogateCaptionEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    artifact_role: str,
    concepts: Sequence[str],
    config: SemiCLIPConfig,
    history: list[dict[str, object]],
    stage1_provenance: Mapping[str, object],
) -> dict[str, object]:
    if artifact_role not in {"last", "final"}:
        raise ValueError("Stage-2 artifact role must be last or final")
    state: dict[str, object] = {
        "checkpoint_kind": "stage2",
        "artifact_role": artifact_role,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "concept_classifier_state_dict": classifier.state_dict(),
        "prompt_state_dict": prompt.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "concepts": list(concepts),
        "config": _serialized_config(config),
        "history": history,
        "stage1_provenance": dict(stage1_provenance),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_stage2_checkpoint(
    path: str | Path,
    model: nn.Module,
    classifier: ConceptClassifier,
    prompt: SurrogateCaptionEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler,
    restore_rng: bool = True,
    expected_concepts: Sequence[str] | None = None,
    expected_stage1_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = (
        "model_state_dict",
        "concept_classifier_state_dict",
        "prompt_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "history",
        "stage1_provenance",
    )
    if state.get("checkpoint_kind") != "stage2":
        raise ValueError("Resume checkpoint is not a Stage-2 checkpoint")
    missing = [name for name in required if name not in state]
    if restore_rng:
        missing.extend(
            name
            for name in ("python_rng_state", "numpy_rng_state", "torch_rng_state")
            if name not in state
        )
    if missing:
        raise ValueError("Stage-2 checkpoint is missing: " + ", ".join(missing))
    if expected_concepts is not None and state.get("concepts") != list(
        expected_concepts
    ):
        raise ValueError("Resume checkpoint concept vocabulary does not match")
    if (
        expected_stage1_provenance is not None
        and state["stage1_provenance"] != dict(expected_stage1_provenance)
    ):
        raise ValueError("Resume checkpoint has different Stage-1 provenance")
    model.load_state_dict(state["model_state_dict"])
    classifier.load_state_dict(state["concept_classifier_state_dict"])
    prompt.load_state_dict(state["prompt_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    if restore_rng:
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state_all" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    return state


def fit_stage2(
    model: nn.Module,
    classifier: ConceptClassifier,
    prompt: SurrogateCaptionEncoder,
    labeled_loader,
    unlabeled_loader,
    validation_loader,
    cached_logits: Mapping[str, torch.Tensor],
    device: torch.device,
    config: SemiCLIPConfig,
    concepts: Sequence[str],
    stage1_provenance: Mapping[str, object],
    output_dir: str | Path,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
    resume_path: str | Path | None = None,
) -> dict[str, object]:
    """Run or resume Stage 2 with a fresh optimizer before optional restoration."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batches_per_epoch = min(len(labeled_loader), max_train_batches or len(labeled_loader))
    if batches_per_epoch <= 0:
        raise ValueError("No Stage-2 labeled batches were selected")
    total_steps = config.stage2_epochs * batches_per_epoch
    optimizer, scheduler = create_stage2_optimizer_scheduler(
        model, classifier, prompt, config, total_steps
    )
    _progress(
        f"Stage 2 starting: epochs={config.stage2_epochs}, "
        f"batches_per_epoch={batches_per_epoch}, total_steps={total_steps}"
    )
    start_epoch = 0
    history: list[dict[str, object]] = []
    if resume_path is not None:
        resumed = restore_stage2_checkpoint(
            resume_path,
            model,
            classifier,
            prompt,
            optimizer,
            scheduler,
            expected_concepts=concepts,
            expected_stage1_provenance=stage1_provenance,
        )
        start_epoch = int(resumed["epoch"]) + 1
        history = list(resumed["history"])
        _progress(
            f"Stage 2 resumed from epoch {start_epoch}; "
            f"remaining={config.stage2_epochs - start_epoch}"
        )
    if start_epoch >= config.stage2_epochs:
        raise ValueError("Stage-2 checkpoint has no remaining configured epochs")

    for epoch in range(start_epoch, config.stage2_epochs):
        epoch_started = time.perf_counter()
        _progress(f"Stage 2 epoch {epoch + 1}/{config.stage2_epochs} starting")
        train_metrics = train_stage2_epoch(
            model,
            classifier,
            prompt,
            labeled_loader,
            unlabeled_loader,
            cached_logits,
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
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        _progress(
            f"Stage 2 epoch {epoch + 1}/{config.stage2_epochs} complete in "
            f"{time.perf_counter() - epoch_started:.1f}s: "
            f"total={train_metrics['total_loss']:.4f}, "
            f"clip={train_metrics['clip_loss']:.4f}, "
            f"semantic={train_metrics['concept_loss']:.4f}, "
            f"intra={train_metrics['intramodal_loss']:.4f}, "
            f"cross={train_metrics['crossmodal_loss']:.4f}, "
            f"validation={validation_metrics['clip_loss']:.4f}, "
            f"selected={train_metrics['selected_pairs_total']}"
        )
        _progress("Stage 2 saving stage2_last.pt")
        torch.save(
            stage2_checkpoint_state(
                model,
                classifier,
                prompt,
                optimizer,
                scheduler,
                epoch,
                "last",
                concepts,
                config,
                history,
                stage1_provenance,
            ),
            output_dir / "stage2_last.pt",
        )

    final_state = stage2_checkpoint_state(
        model,
        classifier,
        prompt,
        optimizer,
        scheduler,
        config.stage2_epochs - 1,
        "final",
        concepts,
        config,
        history,
        stage1_provenance,
    )
    _progress("Stage 2 training complete; saving stage2_final.pt")
    torch.save(final_state, output_dir / "stage2_final.pt")
    summary = {
        "epochs": history,
        "stage1_provenance": dict(stage1_provenance),
        "final_checkpoint": "stage2_final.pt",
    }
    with (output_dir / "stage2_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    _progress(
        f"Stage 2 complete: epochs={len(history)}, "
        f"final_validation={history[-1]['validation']['clip_loss']:.4f}"
    )
    return summary
