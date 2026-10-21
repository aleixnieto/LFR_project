"""Trainable SemiCLIP concept and surrogate-caption modules."""

from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch import nn

from open_clip.transformer import text_global_pool


class ConceptClassifier(nn.Module):
    """Text-initialized trainable classifier used for semantic concepts."""

    def __init__(self, concept_features: torch.Tensor, logit_scale: torch.Tensor):
        super().__init__()
        if concept_features.ndim != 2 or concept_features.shape[0] == 0:
            raise ValueError("concept_features must be a nonempty [concepts, dim] tensor")
        self.weight = nn.Parameter(functional.normalize(concept_features.detach(), dim=-1))
        self.logit_scale = nn.Parameter(logit_scale.detach().reshape(()).clone())

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        if image_features.ndim != 2 or image_features.shape[1] != self.weight.shape[1]:
            raise ValueError("image_features and concept weights must share an embedding dimension")
        return self.logit_scale.exp() * functional.normalize(image_features, dim=-1) @ self.weight.T


@torch.no_grad()
def initialize_concept_classifier(
    model: nn.Module,
    tokenizer,
    concepts: Sequence[str],
    device: torch.device,
) -> ConceptClassifier:
    if not concepts:
        raise ValueError("At least one concept is required")
    prompts = [f"a photo includes {concept.replace('_', ' ')}." for concept in concepts]
    tokens = tokenizer(prompts).to(device)
    features = functional.normalize(model.encode_text(tokens), dim=-1)
    return ConceptClassifier(features, model.logit_scale).to(device)


def encode_text_embeddings(
    model: nn.Module,
    token_ids: torch.Tensor,
    token_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Run an OpenCLIP text tower from caller-supplied token embeddings."""
    if token_ids.ndim != 2 or token_embeddings.ndim != 3:
        raise ValueError("Expected token ids [batch, length] and embeddings [batch, length, width]")
    if token_ids.shape[:2] != token_embeddings.shape[:2]:
        raise ValueError("Token ids and embeddings must have matching batch and sequence dimensions")
    cast_dtype = (
        model.transformer.get_cast_dtype()
        if hasattr(model.transformer, "get_cast_dtype")
        else token_embeddings.dtype
    )
    sequence_length = token_ids.shape[1]
    x = token_embeddings.to(cast_dtype)
    x = x + model.positional_embedding[:sequence_length].to(cast_dtype)
    batch_first = bool(getattr(model.transformer, "batch_first", False))
    if batch_first:
        x = model.transformer(x, attn_mask=model.attn_mask)
    else:
        x = x.permute(1, 0, 2)
        x = model.transformer(x, attn_mask=model.attn_mask)
        x = x.permute(1, 0, 2)
    x = model.ln_final(x)
    pooled = text_global_pool(
        x,
        token_ids,
        model.text_pool_type,
        eos_token_id=getattr(model, "text_eos_id", None),
    )
    projection = model.text_projection
    pooled = projection(pooled) if isinstance(projection, nn.Linear) else pooled @ projection
    return functional.normalize(pooled, dim=-1)


class SurrogateCaptionEncoder(nn.Module):
    """Build top-k concept captions with three learnable vectors per concept."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        concepts: Sequence[str],
        top_k: int,
    ) -> None:
        super().__init__()
        if not concepts or top_k <= 0:
            raise ValueError("Concepts and a positive top_k are required")
        empty = tokenizer([""])
        phrase = tokenizer(["a photo includes"])
        concept_rows = tokenizer(list(concepts))
        if any(tensor.ndim != 2 for tensor in (empty, phrase, concept_rows)):
            raise TypeError("Tokenizer must return two-dimensional token tensors")
        start_token = empty[0, 0]
        end_token = empty[0, 1]
        phrase_ids = phrase[0]
        phrase_content = phrase_ids[1:(phrase_ids == end_token).nonzero()[0, 0]]
        if phrase_content.numel() < 3:
            raise ValueError("The prompt initializer must contain at least three tokens")
        context_length = empty.shape[1]
        required_length = 1 + top_k * 4 + 1
        if required_length > context_length:
            raise ValueError("Surrogate prompt does not fit the tokenizer context length")

        with torch.no_grad():
            initial_context = model.token_embedding(phrase_content[:3].to(model.logit_scale.device))
        context = initial_context.repeat(top_k, 1).reshape(top_k, 3, -1)
        self.context = nn.Parameter(context.detach().clone())
        self.top_k = top_k
        self.context_length = context_length
        self.register_buffer("start_token", start_token.detach().clone())
        self.register_buffer("end_token", end_token.detach().clone())
        self.register_buffer("concept_token_ids", concept_rows[:, 1].detach().clone())

    def build_token_ids(self, concept_indices: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        if concept_indices.ndim != 2 or concept_indices.shape[1] != self.top_k:
            raise ValueError(f"concept_indices must have shape [batch, {self.top_k}]")
        if concept_indices.numel() and (
            concept_indices.min().item() < 0
            or concept_indices.max().item() >= self.concept_token_ids.numel()
        ):
            raise ValueError("concept_indices contains an out-of-range concept")
        batch_size = concept_indices.shape[0]
        token_ids = torch.zeros(
            batch_size,
            self.context_length,
            dtype=self.concept_token_ids.dtype,
            device=concept_indices.device,
        )
        token_ids[:, 0] = self.start_token
        context_positions: list[int] = []
        cursor = 1
        for rank in range(self.top_k):
            context_positions.extend((cursor, cursor + 1, cursor + 2))
            token_ids[:, cursor + 3] = self.concept_token_ids[concept_indices[:, rank]]
            cursor += 4
        token_ids[:, cursor] = self.end_token
        return token_ids, context_positions

    def forward(self, model: nn.Module, concept_indices: torch.Tensor) -> torch.Tensor:
        token_ids, context_positions = self.build_token_ids(concept_indices)
        embeddings = model.token_embedding(token_ids)
        context = self.context.reshape(self.top_k * 3, -1)
        embeddings[:, context_positions] = context.unsqueeze(0)
        return encode_text_embeddings(model, token_ids, embeddings)
