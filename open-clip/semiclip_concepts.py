"""Semantic-concept mining used by SemiCLIP supervised pre-training."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


NOUN_TAGS = frozenset({"NN", "NNS", "NNP", "NNPS"})
NLTK_RESOURCE_COMMAND = (
    "python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng"
)
# Reference-code filter for generic/noisy remote-sensing caption nouns.
OOD_NOUNS = frozenset(
    {
        "lot", "zone", "top", "left", "inspection", "way", "brand", "hand",
        "length", "b", "ease", "communications", "economy", "number", "lies",
        "towards", "type", "places", "shapes", "Lots", "structure", "sizes",
        "pieces", "stands", "end", "built", "thirds", "zones", "bottom",
        "south", "prints", "air", "take-off", "types", "tens", "fact",
        "kind", "style", "side", "question", "situation", "y", "tp",
        "others", "styles", "amount", "drop", "holiday", "differnet", "size",
        "u", "pictures", "use", "l", "picture", "looks", "rate", "design",
        "s", "uses", "arrange", "z", "piece", "palyground", "lots", "view",
        "kinds", "j", "work", "beautiful", "R", "none", "lack", "]",
        "colourful",
    }
)


@dataclass(frozen=True)
class ConceptMiningResult:
    concepts: tuple[str, ...]
    counts: dict[str, int]
    labels_by_filename: dict[str, tuple[int, ...]]
    source_filenames: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "concepts": list(self.concepts),
            "counts": self.counts,
            "labels_by_filename": {
                filename: list(indices)
                for filename, indices in self.labels_by_filename.items()
            },
            "source_filenames": list(self.source_filenames),
            "source_filename_sha256": hash_filenames(self.source_filenames),
        }


def hash_filenames(filenames: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for filename in filenames:
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def require_nltk_resources() -> None:
    """Fail with an actionable message instead of downloading resources."""
    try:
        import nltk

        nltk.data.find("tokenizers/punkt_tab")
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except (ImportError, LookupError) as error:
        raise RuntimeError(
            "SemiCLIP concept mining requires NLTK and its English tokenizer/tagger. "
            f"Install them with: {NLTK_RESOURCE_COMMAND}"
        ) from error


def _default_nltk_functions() -> tuple[Callable, Callable]:
    require_nltk_resources()
    from nltk import pos_tag, word_tokenize

    return word_tokenize, pos_tag


def _unique_in_order(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(items))


def mine_semantic_concepts(
    records: Sequence[dict],
    min_frequency: int = 5,
    max_frequency_ratio: float = 0.3,
    tokenize: Callable[[str], Sequence[str]] | None = None,
    tag: Callable[[Sequence[str]], Sequence[tuple[str, str]]] | None = None,
) -> ConceptMiningResult:
    """Reproduce the released SemiCLIP noun-mining and filtering behavior."""
    if not records:
        raise ValueError("Concept mining requires at least one labeled record")
    if min_frequency <= 0:
        raise ValueError("min_frequency must be positive")
    if not 0 < max_frequency_ratio <= 1:
        raise ValueError("max_frequency_ratio must be positive and at most one")
    if (tokenize is None) != (tag is None):
        raise ValueError("tokenize and tag must be supplied together")
    if tokenize is None or tag is None:
        tokenize, tag = _default_nltk_functions()

    filenames: list[str] = []
    nouns_by_image: list[list[str]] = []
    for record in records:
        filename = record.get("filename")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"Labeled record has no valid filename: {record!r}")
        filenames.append(filename)
        image_nouns: list[str] = []
        for sentence in record.get("sentences", []):
            tokens = sentence.get("tokens")
            if tokens:
                caption = " ".join(str(token) for token in tokens)
                image_nouns.extend(
                    word for word, part in tag(tokenize(caption)) if part in NOUN_TAGS
                )
        nouns_by_image.append(_unique_in_order(image_nouns))

    candidates = _unique_in_order(
        [word for image_nouns in nouns_by_image for word in image_nouns]
    )
    candidates = [
        word
        for word in candidates
        if tag([word])[0][1] in NOUN_TAGS and word not in OOD_NOUNS
    ]
    counts = {
        word: sum(word in image_nouns for image_nouns in nouns_by_image)
        for word in candidates
    }
    # The released code divides by candidate-vocabulary size, not image count.
    denominator = len(candidates)
    concepts = [
        word
        for word in candidates
        if (
            len(word) > 2
            and counts[word] >= min_frequency
            and denominator > 0
            and counts[word] / denominator < max_frequency_ratio
        )
    ]
    if not concepts:
        raise ValueError("Concept filtering removed every mined noun")

    concept_to_index = {word: index for index, word in enumerate(concepts)}
    labels = {
        filename: tuple(
            concept_to_index[word]
            for word in image_nouns
            if word in concept_to_index
        )
        for filename, image_nouns in zip(filenames, nouns_by_image)
    }
    return ConceptMiningResult(
        concepts=tuple(concepts),
        counts={word: counts[word] for word in concepts},
        labels_by_filename=labels,
        source_filenames=tuple(filenames),
    )


def write_concept_artifact(path: str | Path, result: ConceptMiningResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as artifact_file:
        json.dump(result.to_json_dict(), artifact_file, indent=2)
        artifact_file.write("\n")
