"""Next-character cross-entropy shared by the bigram and later GPT."""

from torch.nn import functional as F


def next_character_loss(logits, targets, reduction="mean"):
    if logits.ndim != 3 or logits.shape[:2] != targets.shape:
        raise ValueError("Expected (B,T,V) logits and (B,T) targets.")
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           reduction=reduction)
