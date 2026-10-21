"""Dataset assembly and weak/strong views for the RSICD SemiCLIP workflow."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dataloader import (
    RSICDImageOnlyDataset,
    RSICDSupervisedDataset,
    split_rsicd_records,
)
from semiclip_augment import RandAugmentMC
from semiclip_config import SemiCLIPConfig


OPENCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENCLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class WeakStrongTransform:
    """Return the normal OpenCLIP view and the released strong view."""

    def __init__(
        self,
        weak_transform: Callable,
        image_size: int = 224,
        resize_size: int = 256,
        mean: Sequence[float] = OPENCLIP_MEAN,
        std: Sequence[float] = OPENCLIP_STD,
    ) -> None:
        if image_size <= 0 or resize_size < image_size:
            raise ValueError("resize_size must be at least the positive image_size")
        self.weak_transform = weak_transform
        self.strong_transform = transforms.Compose(
            (
                transforms.Resize(resize_size),
                transforms.RandomCrop(
                    image_size,
                    padding=int(image_size * 0.125),
                    padding_mode="reflect",
                ),
                transforms.RandomHorizontalFlip(),
                RandAugmentMC(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=tuple(mean), std=tuple(std)),
            )
        )

    def __call__(self, image):
        return self.weak_transform(image), self.strong_transform(image)


class ConceptLabeledDataset(Dataset):
    """Add fixed per-image concept labels to the supervised RSICD dataset."""

    def __init__(
        self,
        dataset: RSICDSupervisedDataset,
        labels_by_filename: Mapping[str, Sequence[int]],
        num_concepts: int,
    ) -> None:
        if num_concepts <= 0:
            raise ValueError("num_concepts must be positive")
        missing = [name for name in dataset.filenames if name not in labels_by_filename]
        if missing:
            raise ValueError(f"Concept labels are missing for: {missing[:3]}")
        self.dataset = dataset
        self.num_concepts = num_concepts
        self.targets: list[torch.Tensor] = []
        for filename in dataset.filenames:
            indices = tuple(labels_by_filename[filename])
            if any(index < 0 or index >= num_concepts for index in indices):
                raise ValueError(f"Concept label is out of range for {filename!r}")
            target = torch.zeros(num_concepts, dtype=torch.float32)
            if indices:
                target[list(dict.fromkeys(indices))] = 1.0
            self.targets.append(target)

    @property
    def filenames(self) -> list[str]:
        return self.dataset.filenames

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, tokens, filename = self.dataset[index]
        return image, tokens, self.targets[index], filename


@dataclass(frozen=True)
class SemiCLIPDatasets:
    labeled_train: ConceptLabeledDataset
    labeled_validation: RSICDSupervisedDataset
    unlabeled_stage1: RSICDImageOnlyDataset
    unlabeled_stage2: RSICDImageOnlyDataset


def build_semiclip_datasets(
    config: SemiCLIPConfig,
    weak_transform: Callable,
    validation_transform: Callable,
    tokenizer: Callable,
    labels_by_filename: Mapping[str, Sequence[int]],
    num_concepts: int,
    max_unlabeled_samples: int | None = None,
) -> SemiCLIPDatasets:
    """Build isolated labeled and image-only datasets for both stages."""
    train_records, validation_records = split_rsicd_records(
        config.json_path,
        config.labeled_split,
        config.supervised_train_fraction,
        config.seed,
    )
    labeled_train = RSICDSupervisedDataset(
        config.image_dir,
        train_records,
        weak_transform,
        tokenizer,
        caption_policy="random_one",
        sampling_seed=config.seed,
    )
    labeled_validation = RSICDSupervisedDataset(
        config.image_dir,
        validation_records,
        validation_transform,
        tokenizer,
        caption_policy="all_unique",
        sampling_seed=config.seed,
    )
    paired_transform = WeakStrongTransform(weak_transform)
    return SemiCLIPDatasets(
        labeled_train=ConceptLabeledDataset(
            labeled_train, labels_by_filename, num_concepts
        ),
        labeled_validation=labeled_validation,
        unlabeled_stage1=RSICDImageOnlyDataset(
            config.image_dir,
            config.json_path,
            weak_transform,
            config.unlabeled_split,
            max_samples=max_unlabeled_samples,
        ),
        unlabeled_stage2=RSICDImageOnlyDataset(
            config.image_dir,
            config.json_path,
            paired_transform,
            config.unlabeled_split,
            max_samples=max_unlabeled_samples,
        ),
    )


def collate_concept_labeled(batch) -> dict[str, object]:
    images, caption_batches, concept_targets, filenames = zip(*batch)
    if any(tokens.shape[0] != 1 for tokens in caption_batches):
        raise ValueError("SemiCLIP training requires exactly one caption per image")
    return {
        "images": torch.stack(images),
        "texts": torch.cat(caption_batches),
        "concept_targets": torch.stack(concept_targets),
        "filenames": list(filenames),
    }


def collate_unlabeled_images(batch) -> dict[str, object]:
    images, filenames = zip(*batch)
    return {"images": torch.stack(images), "filenames": list(filenames)}


def collate_unlabeled_views(batch) -> dict[str, object]:
    view_pairs, filenames = zip(*batch)
    weak_views, strong_views = zip(*view_pairs)
    return {
        "weak_images": torch.stack(weak_views),
        "strong_images": torch.stack(strong_views),
        "filenames": list(filenames),
    }
