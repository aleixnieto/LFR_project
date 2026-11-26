import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from knn_utils import DEFAULT_K, DEFAULT_TEMPERATURE, encode_labels, knn_predict

# Allow imports from the parent open-clip directory (dataloader.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SPLIT_MODES = ("project", "stratified")


@torch.no_grad()
def extract_features(encoder, dataloader, device, return_filenames=False):
    encoder.eval()

    features = []
    labels = []
    filenames = []

    for images, targets, names in dataloader:
        images = images.to(device, non_blocking=True)

        embeddings = encoder(images)
        embeddings = F.normalize(embeddings.float(), dim=-1)

        features.append(embeddings.cpu().numpy())
        labels.extend(targets)
        filenames.extend(names)

    if return_filenames:
        return np.concatenate(features), np.array(labels), list(filenames)
    return np.concatenate(features), np.array(labels)


def labeled_partition(
    filenames,
    labels,
    split_mode="project",
    json_path=None,
    source_split="val",
    train_fraction=0.8,
    seed=42,
):
    """Split the labeled images into a reference part and a held-out part.

    Returns two index arrays (reference, held-out) into ``filenames``.

    - ``project``: the partition used by the other experiments of this repository
      (``split_rsicd_records`` with the given seed, 875/219 images for the RSICD
      ``val`` split and seed 42). The held-out part is then the same validation
      subset on which the other models are selected.
    - ``stratified``: a class-stratified split, as in the original linear probe.
    """
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"split_mode must be one of {SPLIT_MODES}, got {split_mode!r}")

    if split_mode == "stratified":
        indices = np.arange(len(labels))
        reference_idx, heldout_idx = train_test_split(
            indices,
            test_size=round(1 - train_fraction, 10),
            random_state=seed,
            stratify=labels,
        )
        return reference_idx, heldout_idx

    if json_path is None:
        raise ValueError("split_mode='project' needs json_path (the RSICD metadata file)")
    from dataloader import split_rsicd_records

    reference_records, heldout_records = split_rsicd_records(
        json_path, source_split, train_fraction, seed
    )
    reference_names = {record["filename"] for record in reference_records}
    heldout_names = {record["filename"] for record in heldout_records}
    is_reference = np.array([name in reference_names for name in filenames])
    is_heldout = np.array([name in heldout_names for name in filenames])
    if not np.all(is_reference ^ is_heldout):
        raise ValueError(
            "The evaluated images do not match the labeled split in the metadata file "
            f"({source_split!r}); cannot rebuild the project partition."
        )
    reference_idx, heldout_idx = np.flatnonzero(is_reference), np.flatnonzero(is_heldout)
    if len(reference_idx) == 0 or len(heldout_idx) == 0:
        raise ValueError("The project partition has an empty reference or held-out part")
    return reference_idx, heldout_idx


def evaluate_linear_probe(
    model,
    dataloader,
    device,
    seed=42,
    split_mode="stratified",
    json_path=None,
):
    # Extract frozen CLIP representations
    X, y, filenames = extract_features(
        model.student, # only use the student encoder for feature extraction, ignore the head and the teacher encoder
        dataloader,
        device,
        return_filenames=True,
    )

    # Same reproducible split at every evaluation
    train_idx, val_idx = labeled_partition(
        filenames, y, split_mode=split_mode, json_path=json_path, seed=seed
    )

    # Fit linear classifier on frozen representations
    classifier = LogisticRegression(
        C=1.0,
        max_iter=1000
    )

    classifier.fit(X[train_idx], y[train_idx])

    # Evaluate on held-out labeled examples
    predictions = classifier.predict(X[val_idx])

    return accuracy_score(y[val_idx], predictions)


def evaluate_knn(
    model,
    dataloader,
    device,
    k=DEFAULT_K,
    temperature=DEFAULT_TEMPERATURE,
    split_mode="project",
    json_path=None,
    seed=42,
):
    """Weighted k-NN accuracy on the held-out part of the labeled images.

    The reference part supplies the labeled neighbors and the held-out part is
    classified, as in DINO's k-NN evaluation. Nothing is trained, so this replaces
    the supervised linear probe for monitoring the representation during training.
    """
    X, y, filenames = extract_features(
        model.student, # only the student encoder, as for the linear probe
        dataloader,
        device,
        return_filenames=True,
    )

    reference_idx, heldout_idx = labeled_partition(
        filenames, y, split_mode=split_mode, json_path=json_path, seed=seed
    )

    targets, class_names = encode_labels(y)
    predictions = knn_predict(
        X[reference_idx],
        targets[reference_idx],
        X[heldout_idx],
        num_classes=len(class_names),
        k=k,
        temperature=temperature,
    )

    return float(np.mean(predictions == targets[heldout_idx]))
