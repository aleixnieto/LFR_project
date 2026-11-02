import numpy as np
import torch
import torch.nn.functional as F

@torch.no_grad()
def extract_features(encoder, dataloader, device):
    encoder.eval()

    features = []
    labels = []

    for images, targets, _ in dataloader:
        images = images.to(device, non_blocking=True)

        embeddings = encoder(images)
        embeddings = F.normalize(embeddings.float(), dim=-1)

        features.append(embeddings.cpu().numpy())
        labels.extend(targets)

    return np.concatenate(features), np.array(labels)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def evaluate_linear_probe(
    model,
    dataloader,
    device,
    seed=42,
):
    # Extract frozen CLIP representations
    X, y = extract_features(
        model.student, # only use the student encoder for feature extraction, ignore the head and the teacher encoder
        dataloader,
        device
    )

    # Same reproducible split at every evaluation
    indices = np.arange(len(y))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=y
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
