"""Supervised OpenCLIP image-caption fine-tuning on a configured RSICD split."""

import argparse
import contextlib
import json
import logging
import random
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as functional
import yaml
from torch.utils.data import DataLoader
from transformers import Adafactor, get_scheduler

from dataloader import build_supervised_train_val_datasets, collate_supervised_captions


DEFAULT_CONFIG_PATH = Path(__file__).with_name("configs") / "finetune_config.yaml"


@dataclass(frozen=True)
class FineTuneConfig:
    model_name: str
    pretrained_tag: str
    checkpoint_path: Path
    image_dir: Path
    json_path: Path
    dataset_split: str
    seed: int
    supervised_train_fraction: float
    batch_size: int
    num_workers: int
    epochs: int
    learning_rate: float
    warmup_fraction: float
    output_dir: Path
    optimizer: str = "adafactor"
    weight_decay: float = 0.0


def resolve_config_path(config_path: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_config(config_path: Path) -> FineTuneConfig:
    """Load the local-only model and supervised-data settings."""
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Fine-tuning config does not exist: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        values = yaml.safe_load(config_file)
    if not isinstance(values, dict):
        raise ValueError(f"Fine-tuning config must contain a mapping: {config_path}")

    required_strings = (
        "model_name",
        "pretrained_tag",
        "checkpoint_path",
        "image_dir",
        "json_path",
        "dataset_split",
        "output_dir",
    )
    missing = [key for key in required_strings if not isinstance(values.get(key), str)]
    if missing:
        raise ValueError(f"Fine-tuning config is missing string values for: {', '.join(missing)}")

    try:
        seed = int(values["seed"])
        train_fraction = float(values["supervised_train_fraction"])
        batch_size = int(values["batch_size"])
        num_workers = int(values["num_workers"])
        epochs = int(values["epochs"])
        learning_rate = float(values["learning_rate"])
        optimizer = str(values.get("optimizer", "adafactor")).lower()
        weight_decay = float(values.get("weight_decay", 0.0))
        warmup_fraction = float(values["warmup_fraction"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Fine-tuning config has invalid numeric settings") from error
    if not 0 < train_fraction < 1:
        raise ValueError("supervised_train_fraction must be strictly between zero and one")
    if batch_size <= 0 or num_workers < 0 or epochs <= 0 or learning_rate <= 0:
        raise ValueError("batch_size, epochs, and learning_rate must be positive; num_workers cannot be negative")
    if optimizer not in {"adafactor", "adamw"}:
        raise ValueError("optimizer must be either 'adafactor' or 'adamw'")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    if not 0 <= warmup_fraction < 1:
        raise ValueError("warmup_fraction must be at least zero and less than one")

    return FineTuneConfig(
        model_name=values["model_name"],
        pretrained_tag=values["pretrained_tag"],
        checkpoint_path=resolve_config_path(config_path, values["checkpoint_path"]),
        image_dir=resolve_config_path(config_path, values["image_dir"]),
        json_path=resolve_config_path(config_path, values["json_path"]),
        dataset_split=values["dataset_split"],
        seed=seed,
        supervised_train_fraction=train_fraction,
        batch_size=batch_size,
        num_workers=num_workers,
        epochs=epochs,
        learning_rate=learning_rate,
        optimizer=optimizer,
        weight_decay=weight_decay,
        warmup_fraction=warmup_fraction,
        output_dir=resolve_config_path(config_path, values["output_dir"]),
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device_name}")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device_name}. Use auto, cpu, or cuda.")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_local_openclip_model(
    config: FineTuneConfig, device: torch.device
) -> tuple[torch.nn.Module, object, object, object]:
    """Load a configured checkpoint without permitting an automatic download."""
    if not config.checkpoint_path.is_file():
        raise FileNotFoundError(
            f"OpenCLIP checkpoint is missing: {config.checkpoint_path}. "
            "Download it manually and update checkpoint_path in finetune_config.yaml."
        )

    pretrained_config = open_clip.get_pretrained_cfg(
        config.model_name, config.pretrained_tag
    )
    if pretrained_config is None:
        raise RuntimeError(
            "Installed OpenCLIP has no preprocessing configuration for "
            f"{config.model_name} / {config.pretrained_tag}."
        )

    model, train_transform, validation_transform = open_clip.create_model_and_transforms(
        config.model_name,
        pretrained=str(config.checkpoint_path),
        device=device,
        image_mean=pretrained_config["mean"],
        image_std=pretrained_config["std"],
        image_interpolation=pretrained_config["interpolation"],
        image_resize_mode=pretrained_config["resize_mode"],
    )
    return model, open_clip.get_tokenizer(config.model_name), train_transform, validation_transform


def supervised_contrastive_loss(
    model: torch.nn.Module,
    images: torch.Tensor,
    texts: torch.Tensor,
    caption_to_image: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Symmetric CLIP loss where every caption is positive for its source image."""
    if images.ndim != 4 or texts.ndim != 2:
        raise ValueError("Expected images [batch, channels, height, width] and texts [captions, tokens]")
    if caption_to_image.ndim != 1 or caption_to_image.numel() != texts.shape[0]:
        raise ValueError("caption_to_image must map every caption token row to an image")
    if caption_to_image.numel() == 0:
        raise ValueError("A supervised batch must contain at least one caption")
    if caption_to_image.min().item() < 0 or caption_to_image.max().item() >= images.shape[0]:
        raise ValueError("caption_to_image contains an out-of-range image index")

    image_features = functional.normalize(model.encode_image(images), dim=-1)
    text_features = functional.normalize(model.encode_text(texts), dim=-1)
    logit_scale = model.logit_scale.exp()
    text_to_image_logits = logit_scale * text_features @ image_features.T
    image_to_text_logits = text_to_image_logits.T

    loss_text_to_image = functional.cross_entropy(text_to_image_logits, caption_to_image)
    image_to_text_targets = functional.one_hot(
        caption_to_image, num_classes=images.shape[0]
    ).T.to(image_to_text_logits.dtype)
    image_to_text_targets /= image_to_text_targets.sum(dim=1, keepdim=True)
    loss_image_to_text = -(
        image_to_text_targets * functional.log_softmax(image_to_text_logits, dim=1)
    ).sum(dim=1).mean()
    loss = (loss_text_to_image + loss_image_to_text) / 2

    positive_text_to_image = text_to_image_logits[
        torch.arange(texts.shape[0], device=images.device), caption_to_image
    ].mean()
    positive_image_to_text = (image_to_text_logits * image_to_text_targets).sum(dim=1).mean()
    return loss, {
        "positive_text_to_image_logit": float(positive_text_to_image.detach().item()),
        "positive_image_to_text_logit": float(positive_image_to_text.detach().item()),
        "logit_scale": float(logit_scale.detach().item()),
    }


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["images"].to(device),
        batch["texts"].to(device),
        batch["caption_to_image"].to(device),
    )


def mean_epoch_statistics(statistics: list[dict[str, float]]) -> dict[str, float]:
    if not statistics:
        raise ValueError("No batches were processed")
    keys = statistics[0].keys()
    return {key: sum(item[key] for item in statistics) / len(statistics) for key in keys}


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, float]:
    """Fine-tune every OpenCLIP parameter for one supervised epoch."""
    model.train()
    batch_statistics: list[dict[str, float]] = []
    for batch_index, batch in enumerate(data_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        loss, statistics = supervised_contrastive_loss(
            model, *move_batch_to_device(batch, device)
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at batch {batch_index}: {loss.item()}")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.logit_scale.clamp_(0, 4.6052)
        scheduler.step()
        batch_statistics.append({"loss": float(loss.detach().item()), **statistics})

    metrics = mean_epoch_statistics(batch_statistics)
    metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
    metrics["batches"] = float(len(batch_statistics))
    return metrics


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, float]:
    """Evaluate the held-out supervised subset without updating parameters."""
    model.eval()
    batch_statistics: list[dict[str, float]] = []
    for batch_index, batch in enumerate(data_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        loss, statistics = supervised_contrastive_loss(
            model, *move_batch_to_device(batch, device)
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite validation loss at batch {batch_index}: {loss.item()}")
        batch_statistics.append({"loss": float(loss.item()), **statistics})

    metrics = mean_epoch_statistics(batch_statistics)
    metrics["batches"] = float(len(batch_statistics))
    return metrics


def create_optimizer_and_scheduler(
    model: torch.nn.Module, config: FineTuneConfig, total_steps: int
) -> tuple[torch.optim.Optimizer, object]:
    if config.optimizer == "adafactor":
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=True,
            relative_step=False,
            warmup_init=False,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        decay_parameters = []
        no_decay_parameters = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if (
                parameter.ndim < 2
                or "bn" in name.lower()
                or "ln" in name.lower()
                or "bias" in name.lower()
                or "logit_scale" in name.lower()
            ):
                no_decay_parameters.append(parameter)
            else:
                decay_parameters.append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_parameters},
                {"params": no_decay_parameters, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=config.weight_decay,
        )
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(config.warmup_fraction * total_steps),
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def prepare_output_dir(output_dir: Path, config_path: Path) -> None:
    """Create the run directory and retain the exact configuration used."""
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path exists but is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")


def write_metrics(output_dir: Path, metrics: dict[str, object]) -> None:
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write("\n")


def checkpoint_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_validation_loss: float,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
) -> dict[str, object]:
    """Capture all state required for a future reproducible continuation."""
    state: dict[str, object] = {
        "epoch": epoch,
        "best_validation_loss": best_validation_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def verify_last_checkpoint(
    checkpoint_path: Path,
    config: FineTuneConfig,
    total_steps: int,
) -> int:
    """Restore the final checkpoint into a fresh, matching local OpenCLIP model."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required_keys = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "train_metrics",
        "validation_metrics",
    }
    missing = sorted(required_keys.difference(state))
    if missing:
        raise ValueError(f"Checkpoint is missing required state: {', '.join(missing)}")

    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        restored_model = open_clip.create_model(
            config.model_name, pretrained=None, device="cpu"
        )
    finally:
        logging.disable(previous_disable_level)
    restored_optimizer, restored_scheduler = create_optimizer_and_scheduler(
        restored_model, config, total_steps
    )
    restored_model.load_state_dict(state["model_state_dict"])
    restored_optimizer.load_state_dict(state["optimizer_state_dict"])
    restored_scheduler.load_state_dict(state["scheduler_state_dict"])
    return int(state["epoch"])


class TeeStream:
    """Mirror text to the terminal and a line-flushed run log."""

    def __init__(self, terminal, log_file) -> None:
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log_file.write(text)
        self.log_file.flush()
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def __getattr__(self, name: str):
        return getattr(self.terminal, name)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, help="Override config batch_size.")
    parser.add_argument("--epochs", type=int, help="Override config epochs.")
    parser.add_argument(
        "--max-train-batches",
        type=int,
        help="Limit batches per training epoch; omit for the full training subset.",
    )
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        help="Limit validation batches per epoch; omit for the full validation subset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override config output_dir for run artifacts and checkpoints.",
    )
    args = parser.parse_args()
    for name in ("batch_size", "epochs", "max_train_batches", "max_validation_batches"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def run_finetune(args: argparse.Namespace, config: FineTuneConfig) -> None:
    device = resolve_device(args.device)
    set_seed(config.seed)
    model, tokenizer, train_transform, validation_transform = load_local_openclip_model(
        config, device
    )
    train_dataset, validation_dataset = build_supervised_train_val_datasets(
        image_dir=config.image_dir,
        json_path=config.json_path,
        train_transform=train_transform,
        validation_transform=validation_transform,
        tokenizer=tokenizer,
        train_fraction=config.supervised_train_fraction,
        seed=config.seed,
        source_split=config.dataset_split,
    )
    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_supervised_captions,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    train_batches = min(len(train_loader), args.max_train_batches or len(train_loader))
    if train_batches == 0:
        raise ValueError("No supervised training batches selected")
    total_steps = config.epochs * train_batches
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, total_steps)

    metrics: dict[str, object] = {
        "model_name": config.model_name,
        "pretrained_tag": config.pretrained_tag,
        "checkpoint_path": str(config.checkpoint_path),
        "image_dir": str(config.image_dir),
        "json_path": str(config.json_path),
        "device": str(device),
        "seed": config.seed,
        "train_images": len(train_dataset),
        "validation_images": len(validation_dataset),
        "configured_epochs": config.epochs,
        "optimizer": config.optimizer,
        "weight_decay": config.weight_decay,
        "max_train_batches": args.max_train_batches,
        "max_validation_batches": args.max_validation_batches,
        "epochs": [],
    }

    print(f"Model: {config.model_name} / {config.pretrained_tag}")
    print(f"Checkpoint: {config.checkpoint_path}")
    print(f"Device: {device}")
    print(f"Optimizer: {config.optimizer}; weight decay: {config.weight_decay:g}")
    print(f"Train/validation images: {len(train_dataset)}/{len(validation_dataset)}")
    best_validation_loss = float("inf")
    for epoch in range(config.epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, args.max_train_batches
        )
        validation_metrics = validate_one_epoch(
            model, validation_loader, device, args.max_validation_batches
        )
        validation_loss = validation_metrics["loss"]
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            is_best = True
        else:
            is_best = False

        state = checkpoint_state(
            model,
            optimizer,
            scheduler,
            epoch,
            best_validation_loss,
            train_metrics,
            validation_metrics,
        )
        torch.save(state, config.output_dir / "ckp_last.pth")
        if is_best:
            torch.save(state, config.output_dir / "ckp_best.pth")

        epoch_record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": validation_metrics,
            "is_best": is_best,
        }
        metrics["epochs"].append(epoch_record)
        metrics["best_validation_loss"] = best_validation_loss
        write_metrics(config.output_dir, metrics)
        print(
            f"Epoch {epoch + 1}/{config.epochs}: "
            f"train_loss={train_metrics['loss']:.6f} "
            f"validation_loss={validation_loss:.6f} "
            f"lr={train_metrics['learning_rate']:.8f}"
        )

    restored_epoch = verify_last_checkpoint(
        config.output_dir / "ckp_last.pth", config, total_steps
    )
    metrics["checkpoint_load_epoch"] = restored_epoch + 1
    write_metrics(config.output_dir, metrics)
    print(f"Artifacts: {config.output_dir}")
    print(f"Checkpoint restore verified for epoch {restored_epoch + 1}")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir.resolve())
    prepare_output_dir(config.output_dir, config_path)

    log_path = config.output_dir / "train.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        stdout = TeeStream(sys.stdout, log_file)
        stderr = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print(f"Writing terminal output to: {log_path}")
            run_finetune(args, config)

# Deferred: mixed supervised/SimCLR training is intentionally introduced later.
if __name__ == "__main__":
    main()
