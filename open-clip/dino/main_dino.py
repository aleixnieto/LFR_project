import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

import open_clip

from dataloader import RSICDImageOnlyDataset
import copy

import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from main_zeroshot import load_model_config, load_dataset_config


DEFAULT_CONFIG_PATH = (Path(__file__).with_name("configs") / "dino_config.yaml")
DEFAULT_OUTPUT_DIR = Path("results/dino")

@dataclass(frozen=True)
class DinoConfig:
    in_dim: int = 512
    out_dim: int = 8192
    epochs: int = 30
    teacher_momentum: float = 0.996
    center_momentum: float = 0.9
    student_temperature: float = 0.1
    teacher_temperature: float = 0.04
    encoder_lr: float = 1e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.04

class DinoHead(torch.nn.Module):
    def __init__(self, in_dim=512, hidden_dim=2048,
                 bottleneck_dim=256, out_dim=8192):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, bottleneck_dim),
        )

        self.last_layer = torch.nn.utils.parametrizations.weight_norm(
            torch.nn.Linear(bottleneck_dim, out_dim, bias=False)
        )

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)

class DinoModel(torch.nn.Module):
    def __init__(self, clip_visual_encoder, config: DinoConfig): # in_dim the embed. dim. that CLIP vision encoder gives out
        super().__init__()

        self.config = config

        self.student = clip_visual_encoder
        self.teacher = copy.deepcopy(clip_visual_encoder)

        # Dino head 
        self.student_head = DinoHead(in_dim=config.in_dim, out_dim=config.out_dim)
        self.teacher_head = copy.deepcopy(self.student_head)

        self.teacher.requires_grad_(False)
        self.teacher_head.requires_grad_(False)

        self.register_buffer("center", torch.zeros(1, config.out_dim)) 


    def forward(self, x1, x2) -> torch.Tensor:
        """
        x1, x2 are two GLOBAL augmented views from x.
        By `global` we mean that more than 50% of the image is visible.
        """

        # Student processes both views
        s1 = self.student_head(self.student(x1))
        s2 = self.student_head(self.student(x2))

        # Teacher processes both views
        with torch.no_grad():
            t1 = self.teacher_head(self.teacher(x1))
            t2 = self.teacher_head(self.teacher(x2))

        return (s1, s2), (t1, t2)

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
    def __init__(self, image_size=224, mean=None, std=None):

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.5, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05
                )
            ], p=0.8),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, image):
        x1 = self.transform(image)
        x2 = self.transform(image)
        return x1, x2


def dino_loss(student_out, teacher_out, center, temp_student, temp_teacher):
    "Compute the DINO loss within one batch."

    s1, s2 = student_out
    t1, t2 = teacher_out

    log_p_s1 = F.log_softmax(s1.float() / temp_student, dim=-1)
    log_p_s2 = F.log_softmax(s2.float() / temp_student, dim=-1) # TODO need to do s1.float()? No, but something to do with num. stability if float32 ot float16

    with torch.no_grad():
        p_t1 = F.softmax((t1.float() - center) / temp_teacher, dim=-1) # TODO: need to do t2.float() and t1.float()?
        p_t2 = F.softmax((t2.float() - center) / temp_teacher, dim=-1)

    # Compute loss on different views:
    loss_1 = -(p_t1 * log_p_s2).sum(dim=-1).mean() 
    loss_2 = -(p_t2 * log_p_s1).sum(dim=-1).mean()

    return (loss_1 + loss_2) / 2

def train_one_epoch(model, dataloader, optimizer, device):
    "Compute the loss for one epoch over all batches in the dataloader."
    model.train()
    model.teacher.eval()
    model.teacher_head.eval()

    total_loss = 0.0

    for (x1, x2), _ in tqdm(dataloader, desc="DINO training", unit="batch"):
        x1 = x1.to(device) # TODO: add "non_blocking=True"?
        x2 = x2.to(device)

        student_out, teacher_out = model(x1, x2)

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
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
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
        mean=pretrained_config["mean"],
        std=pretrained_config["std"]
    )

    train_dataset = RSICDImageOnlyDataset(
        image_dir=dataset_config.image_dir,
        json_path=dataset_config.json_path,
        transform=augmentation,
        split=dataset_config.dataset_split
    )

    train_dt = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )


    # 1. Load pre-trained CLIP
    clip_model, train_transform, validation_transform = open_clip.create_model_and_transforms(
        model_config.model_name,
        pretrained=str(model_config.checkpoint_path),
        device=device,
        image_mean=pretrained_config["mean"],
        image_std=pretrained_config["std"],
        image_interpolation=pretrained_config["interpolation"],
        image_resize_mode=pretrained_config["resize_mode"],
    )

    # 2. Init DINO student and teacher
    model = DinoModel(
        clip_visual_encoder=clip_model.visual,
        config=dino_config
    ).to(device)

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
    best_val_loss = float("inf")


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

if __name__ == "__main__":
    main()