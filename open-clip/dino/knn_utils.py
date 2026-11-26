"""Weighted k-NN classification on frozen features, as in DINO's ``eval_knn.py``.

The functions here only need NumPy, so they can be tested without a GPU or a model.

Reference: https://github.com/facebookresearch/dino/blob/main/eval_knn.py
"""

import math
from collections.abc import Sequence

import numpy as np

# Defaults of the DINO paper: 20 neighbors and a temperature of 0.07.
DEFAULT_K = 20
DEFAULT_TEMPERATURE = 0.07


def encode_labels(labels: Sequence, class_names: Sequence | None = None):
    """Map labels (for example class-name strings) to integer indices.

    Returns ``(indices, class_names)``. When ``class_names`` is not given, the sorted
    distinct labels are used.
    """
    labels = list(labels)
    if class_names is None:
        class_names = sorted(set(labels))
    class_names = list(class_names)
    index = {name: i for i, name in enumerate(class_names)}
    unknown = sorted({str(label) for label in labels if label not in index})
    if unknown:
        raise ValueError(f"Labels missing from class_names: {unknown[:5]}")
    return np.array([index[label] for label in labels], dtype=np.int64), class_names


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("features must be a [num_images, dim] array")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, 1e-12, None)


def knn_predict(
    reference_features: np.ndarray,
    reference_labels: np.ndarray,
    query_features: np.ndarray,
    num_classes: int,
    k: int = DEFAULT_K,
    temperature: float = DEFAULT_TEMPERATURE,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Predict the class index of every query image with a weighted k-NN vote.

    Each query looks at its ``k`` most similar reference images (cosine similarity).
    A neighbor with similarity ``s`` adds ``exp(s / temperature)`` to the score of its
    class, and the class with the highest score is predicted. This is the classifier
    of DINO's ``eval_knn.py``. No parameter is trained.

    ``k`` is reduced to the number of reference images when it is larger. Ties are
    broken deterministically: by the lower reference index among neighbors and by the
    lower class index among classes.
    """
    reference = _l2_normalize(reference_features)
    query = _l2_normalize(query_features)
    reference_labels = np.asarray(reference_labels, dtype=np.int64)
    if reference.shape[0] == 0 or query.shape[0] == 0:
        raise ValueError("reference and query sets must not be empty")
    if reference.shape[1] != query.shape[1]:
        raise ValueError("reference and query features must share their dimension")
    if reference_labels.shape != (reference.shape[0],):
        raise ValueError("reference_labels must hold one label per reference image")
    if num_classes <= 0 or reference_labels.min() < 0 or reference_labels.max() >= num_classes:
        raise ValueError("reference_labels must be class indices in [0, num_classes)")
    if k < 1:
        raise ValueError("k must be at least 1")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    k = min(int(k), reference.shape[0])

    predictions = np.empty(query.shape[0], dtype=np.int64)
    for start in range(0, query.shape[0], chunk_size):
        similarity = query[start:start + chunk_size] @ reference.T
        neighbors = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
        weights = np.exp(np.take_along_axis(similarity, neighbors, axis=1) / temperature)
        scores = np.zeros((similarity.shape[0], num_classes), dtype=np.float64)
        rows = np.repeat(np.arange(similarity.shape[0]), k)
        np.add.at(scores, (rows, reference_labels[neighbors].ravel()), weights.ravel())
        predictions[start:start + chunk_size] = scores.argmax(axis=1)
    return predictions


def knn_accuracy(
    reference_features: np.ndarray,
    reference_labels: np.ndarray,
    query_features: np.ndarray,
    query_labels: np.ndarray,
    num_classes: int,
    k: int = DEFAULT_K,
    temperature: float = DEFAULT_TEMPERATURE,
) -> float:
    """Top-1 accuracy (between 0 and 1) of the weighted k-NN classifier."""
    predictions = knn_predict(
        reference_features, reference_labels, query_features, num_classes, k, temperature
    )
    return float(np.mean(predictions == np.asarray(query_labels, dtype=np.int64)))


def exact_mcnemar(first_only_correct: int, second_only_correct: int) -> float:
    """Exact two-sided McNemar p-value from the two discordant counts.

    Under the null hypothesis the images on which exactly one of two models is
    correct are split evenly, so the smaller count follows Binomial(n, 0.5).
    """
    if first_only_correct < 0 or second_only_correct < 0:
        raise ValueError("discordant counts must be non-negative")
    n = first_only_correct + second_only_correct
    if n == 0:
        return 1.0
    smaller = min(first_only_correct, second_only_correct)
    tail = sum(math.comb(n, i) for i in range(smaller + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)
