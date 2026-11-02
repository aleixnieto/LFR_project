import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import copy
import json
import yaml
import argparse
import open_clip
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.init import trunc_normal_
from torchvision import transforms
from torch.utils.data import DataLoader

from plot_utils import plot_training_history
from validate_features import evaluate_linear_probe
from main_zeroshot import load_model_config, load_dataset_config
from dataloader import RSICDImageOnlyDataset, RSICDZeroShotDataset
from augment_utils import GaussianBlur, Solarization

DEFAULT_CONFIG_PATH = (Path(__file__).with_name("configs") / "dino_config.yaml")
DEFAULT_OUTPUT_DIR = Path("results/dino")

@dataclass(frozen=True)
class DinoConfig:
    in_dim: int
    out_dim: int
    hidden_dim: int
    bottleneck_dim: int
    epochs: int
    teacher_momentum: float
    center_momentum: float
    student_temperature: float
    teacher_temperature: float
    encoder_lr: float
    head_lr: float
    weight_decay: float
    image_size: int

    # Multi-crop settings
    global_crops_scale: tuple[float, float]
    local_crops_scale: tuple[float, float]
    local_crops_number: int

    # Dataloader
    batch_size: int
    num_workers: int

class DinoHead(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim,
                 bottleneck_dim, out_dim):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, bottleneck_dim),
        )

        self.mlp.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))

        # original0 = weight_g (magnitude)
        # original1 = weight_v (direction)
        self.last_layer.parametrizations.weight.original0.data.fill_(1.0) # initialize magnitude to 1, keep it frozen

        # Freeze the magnitude, as in original DINO
        self.last_layer.parametrizations.weight.original0.requires_grad_(False)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2) # L2 normalize the output embeddings, similar to DINO paper https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
        return self.last_layer(x)

class DinoModel(torch.nn.Module):
    def __init__(self, clip_visual_encoder, config: DinoConfig): # in_dim the embed. dim. that CLIP vision encoder gives out
        super().__init__()

        self.config = config

        self.student = clip_visual_encoder
        self.teacher = copy.deepcopy(clip_visual_encoder)

        # Dino head 
        self.student_head = DinoHead(in_dim=config.in_dim, hidden_dim=config.hidden_dim, bottleneck_dim=config.bottleneck_dim, out_dim=config.out_dim)
        self.teacher_head = copy.deepcopy(self.student_head)

        self.teacher.requires_grad_(False)
        self.teacher_head.requires_grad_(False)

        self.register_buffer("center", torch.zeros(1, config.out_dim)) 

    def forward(self, crops) -> torch.Tensor:
        """
        crops is a list of augmented views from x.
        The first two elements are global crops.
        Students processes all crops.
        Teacher processes only the global crops.
        """

        # Student: 2 global + 8 local
        student_out = [
            self.student_head(self.student(crop))
            for crop in crops
        ]

        # Teacher: 2 global only
        with torch.no_grad():
            teacher_out = [
                self.teacher_head(self.teacher(crop))
                for crop in crops[:2]  # only the first two global crops
            ]

        return student_out, teacher_out

    @torch.no_grad()
    def update_center(self, teacher_logits):
        "The center should be updated with momentum at every batch."
        batch_center = teacher_logits.float().mean(dim=0, keepdim=True)
        m = self.config.center_momentum
        self.center.mul_(m).add_(
            batch_center,
            alpha=1 - m
        )

    @torch.no_grad()
    def update_teacher(self, momentum=None):
        "EMA update of teacher parameters using student's parameters"

        if momentum is None:
            momentum = self.config.teacher_momentum

        for student_module, teacher_module in [
            (self.student, self.teacher), 
            (self.student_head, self.teacher_head)
            ]:
            for ps, pt in zip(student_module.parameters(), teacher_module.parameters()):
                # pt.data = momentum * pt.data + (1 - momentum) * ps.data 
                pt.lerp_(ps, 1 - momentum)

class DINOAugmentation:
    """
    DINO image augmentation for 2 global + 8 local crops. 
    Copied from original DINO implementation at:
    https://github.com/facebookresearch/dino/blob/main/main_dino.py
    """
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, image_size=224, mean=None, std=None):

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2)
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # First global crop
        self.global_transfo1 = transforms.Compose([
            transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                GaussianBlur(1.0),
                normalize
            ])
        ])

        # Second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                GaussianBlur(0.1),
                Solarization(0.2),
                normalize
            ])
        ])

        # Transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            # keep image size to 224x224, so as not to mess up with the positional embeddings...
            # in DINO's implementation, the image size of local crops is instead 96x96.
            transforms.RandomResizedCrop(image_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.5),
            normalize
        ])

    def __call__(self, image):
        "Produces a list of 10 tensors with shape [3, image_size, image_size]"
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))

        return crops


def dino_loss(student_out, teacher_out, center, temp_student, temp_teacher):
    """
    Compute DINO loss in one batch.

    Teacher: 2 global crops
    Student: all crops (2 global + local)

    Excludes same-view global comparisons.
    """

    student_log_probs = [
        F.log_softmax(s.float() / temp_student, dim=-1) # do float() for something to do with num. stability of floating points
        for s in student_out
    ]

    with torch.no_grad():
        teacher_probs = [
            F.softmax((t.float() - center) / temp_teacher, dim=-1)
            for t in teacher_out
        ]

    total_loss = 0.0
    n_loss_terms = 0

    for i, teacher_prob in enumerate(teacher_probs):
        for j, student_log_prob in enumerate(student_log_probs):
            if i == j:
                continue
            loss = -(teacher_prob * student_log_prob).sum(dim=-1).mean() # mean() averages over batch, but what does dim=-1 do? TODO draw dummy example to understand
            total_loss += loss
            n_loss_terms += 1

    return total_loss / n_loss_terms

def train_one_epoch(model, dataloader, optimizer, device):
    "Compute the loss for one epoch over all batches in the dataloader."
    model.train()
    model.teacher.eval()
    model.teacher_head.eval()

    total_loss = 0.0

    for crops, _ in tqdm(dataloader, desc="DINO training", unit="batch"):

        # Move all ten crop batches to the GPU
        crops = [
            crop.to(device, non_blocking=True)
            for crop in crops
        ]

        student_out, teacher_out = model(crops)

        loss = dino_loss(
            student_out, 
            teacher_out, 
            model.center,
            model.config.student_temperature,
            model.config.teacher_temperature
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.update_teacher()
        teacher_out = torch.cat(teacher_out, dim=0)
        model.update_center(teacher_out)

        total_loss += loss.item()

    # return loss for this epoch
    return total_loss / len(dataloader) # len(dataloader) gives the number of batches


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device_name}")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device_name}. Use auto, cpu, or cuda.")
    return device

def load_dino_config(config_path: Path) -> DinoConfig:
    with config_path.open(encoding="utf-8") as f:
        values = yaml.safe_load(f)

    return DinoConfig(
        **{
            key: values[key]
            for key in DinoConfig.__dataclass_fields__
            if key in values
        }
    )

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML model/checkpoint configuration (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Evaluate only the first N test images; omit to evaluate the full test split.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Evaluation device (default: auto).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    dataset_config = load_dataset_config(args.config)
    model_config = load_model_config(args.config)
    dino_config = load_dino_config(args.config)
    pretrained_config = open_clip.get_pretrained_cfg(
        model_config.model_name, model_config.pretrained_tag
    )
    device = resolve_device(args.device)

    augmentation = DINOAugmentation(
        global_crops_scale=dino_config.global_crops_scale,
        local_crops_scale=dino_config.local_crops_scale,
        local_crops_number=dino_config.local_crops_number,
        image_size=dino_config.image_size,
        mean=pretrained_config["mean"],
        std=pretrained_config["std"]
    )

    # 1. Load pre-trained CLIP
    clip_model, train_transform, val_transform = open_clip.create_model_and_transforms(
        model_config.model_name,
        pretrained=str(model_config.checkpoint_path),
        device=device,
        image_mean=pretrained_config["mean"],
        image_std=pretrained_config["std"],
        image_interpolation=pretrained_config["interpolation"],
        image_resize_mode=pretrained_config["resize_mode"],
    )

    train_dataset = RSICDImageOnlyDataset(
        image_dir=dataset_config.image_dir,
        json_path=dataset_config.json_path,
        transform=augmentation,
        split=dataset_config.dataset_split
    )

    train_dt = DataLoader(
        train_dataset,
        batch_size=dino_config.batch_size,
        shuffle=True,
        num_workers=dino_config.num_workers,
        pin_memory=device.type == "cuda",
    )

    probe_dataset = RSICDZeroShotDataset(
        image_dir=dataset_config.image_dir,
        json_path=dataset_config.json_path,
        class_dir=dataset_config.class_dir,
        transform=val_transform,
        split="val"
    )

    probe_loader = DataLoader(
        probe_dataset,
        batch_size=dino_config.batch_size,
        shuffle=False,
        num_workers=dino_config.num_workers,
        pin_memory=device.type == "cuda",
    )

    # 2. Init DINO student and teacher
    model = DinoModel(
        clip_visual_encoder=clip_model.visual,
        config=dino_config
    ).to(device)


    # crops, _ = next(iter(train_dt))
    # print("Number of crops:", len(crops))

    # for i, crop in enumerate(crops):
    #     print(f"Crop {i}: {crop.shape}")

    # crops = [crop.to(device) for crop in crops]
    # with torch.no_grad():
    #     student_out, teacher_out = model(crops)
    # print("Student outputs:", len(student_out))
    # print("Teacher outputs:", len(teacher_out))
    # print("Student output shape:", student_out[0].shape)
    # print("Teacher output shape:", teacher_out[0].shape)

    # Output:
    # Number of crops: 10
    # Crop 0: torch.Size([32, 3, 224, 224])
    # ...
    # Crop 9: torch.Size([32, 3, 224, 224])
    # Student outputs: 10
    # Teacher outputs: 2
    # Student output shape: torch.Size([32, 65536])
    # Teacher output shape: torch.Size([32, 65536])

    # 3. Optimizer
    optimizer = torch.optim.AdamW(
        [{"params": model.student.parameters(), "lr": dino_config.encoder_lr},
         {"params": model.student_head.parameters(), "lr": dino_config.head_lr}
        ],
        weight_decay=dino_config.weight_decay
    )

    # 4. Output
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = DEFAULT_OUTPUT_DIR / "best_model.pth"
    last_checkpoint_path = DEFAULT_OUTPUT_DIR / "last_model.pth"

    # 5. Training
    history = []
    probe_history = []
    best_probe_accuracy = float("-inf")

    # Original CLIP baseline (epoch 0)
    baseline_accuracy = evaluate_linear_probe(
        model, probe_loader, device
    )

    probe_frequency = 5
    checkpoint_frequency = 5

    probe_history.append({
        "epoch": 0,
        "accuracy": baseline_accuracy
    })

    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")

    epochs = args.epochs if args.epochs is not None else dino_config.epochs

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_dt, optimizer, device)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Train Loss: {train_loss:.4f}"
        )
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
        })
        # Evaluate linear probe periodically
        if (epoch + 1) % probe_frequency == 0 or epoch + 1 == epochs:
            accuracy = evaluate_linear_probe(
                model, probe_loader, device
            )
            probe_history.append({
                "epoch": epoch + 1,
                "accuracy": accuracy
            })
            print(
                f"Epoch {epoch + 1} | "
                f"Linear probe accuracy: {accuracy:.4f}"
            )

            # Save best checkpoint based on probe accuracy
            if accuracy > best_probe_accuracy:
                best_probe_accuracy = accuracy

                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "probe_accuracy": accuracy,
                }, best_checkpoint_path)

                print(f"New best checkpoint: {best_checkpoint_path}")

        # Always save last checkpoint
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
        }, last_checkpoint_path)

        # Save periodic checkpoints for comparing epochs
        if (epoch + 1) % checkpoint_frequency == 0:
            path = DEFAULT_OUTPUT_DIR / f"checkpoint_epoch_{epoch + 1}.pth"

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
            }, path)

            print(f"Saved periodic checkpoint: {path}")

        # Save metrics
        with (DEFAULT_OUTPUT_DIR / "history.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump({
                "training": history,
                "linear_probe": probe_history
            }, f, indent=2)

    plot_training_history(
        history,
        probe_history,
        DEFAULT_OUTPUT_DIR
    )
    print("DINO training completed.")

if __name__ == "__main__":
    main()