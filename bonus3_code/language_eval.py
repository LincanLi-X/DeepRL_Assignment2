"""Full-validation diagnostics and handwritten autoregressive character sampling."""

from contextlib import nullcontext
import math

import torch

from losses import next_character_loss
from paths import CONTEXT_LENGTH


def autocast_context(device, dtype):
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype is not None else nullcontext()


@torch.inference_mode()
def full_validation(model, sequence, batch_size=32, autocast_dtype=None):
    """Score every target after the first once, resetting context in each chunk."""
    model.eval()
    device = next(model.parameters()).device
    count = sequence.numel() - 1
    starts = torch.arange(0, count, CONTEXT_LENGTH, dtype=torch.long)
    lengths = torch.minimum(torch.full_like(starts, CONTEXT_LENGTH), count - starts)
    coverage = torch.zeros(count, dtype=torch.long)
    for start, length in zip(starts.tolist(), lengths.tolist()):
        coverage[start:start + length] += 1
    if not bool((coverage == 1).all()):
        raise ValueError("The diagnostic must score every target exactly once.")
    nll_terms, evaluated_targets = [], 0
    # Batch equal-length chunks, including any one final shorter chunk.
    for length in lengths.unique(sorted=True).tolist():
        matching_starts = starts[lengths == length]
        for offset in range(0, matching_starts.numel(), batch_size):
            selected = matching_starts[offset:offset + batch_size]
            positions = selected[:, None] + torch.arange(length)
            inputs = sequence[positions].to(device)
            targets = sequence[positions + 1].to(device)
            with autocast_context(device, autocast_dtype):
                logits = model(inputs)
                losses = next_character_loss(logits, targets, reduction="none")
            nll_terms.append(losses.to(torch.float64).sum().item())
            evaluated_targets += targets.numel()
    if evaluated_targets != count:
        raise RuntimeError("Incorrect full-validation target count.")
    total_nll = math.fsum(nll_terms)
    mean = total_nll / count
    return {"cross_entropy": mean, "perplexity": math.exp(mean),
            "total_negative_log_likelihood": total_nll, "targets": count,
            "full_chunks": (lengths == CONTEXT_LENGTH).sum().item(),
            "last_chunk_length": lengths[-1].item(),
            "chunk_count": starts.numel(), "all_targets_scored_once": True}, starts, lengths


@torch.inference_mode()
def generate(model, prompt_ids, new_characters, temperature, generator,
             max_context=CONTEXT_LENGTH, autocast_dtype=None):
    """Sample with a dedicated generator, seeded once by the caller."""
    if prompt_ids.ndim != 1 or prompt_ids.numel() == 0 or temperature <= 0:
        raise ValueError("Use a nonempty prompt and a positive temperature.")
    model.eval()
    device = next(model.parameters()).device
    prefix = prompt_ids.to(device).clone()
    for _ in range(new_characters):
        inputs = prefix[-max_context:].unsqueeze(0)
        with autocast_context(device, autocast_dtype):
            last_logits = model(inputs)[:, -1, :] / temperature
            probabilities = torch.softmax(last_logits, dim=-1)
        # Generator and probabilities must share a device. The sampled token
        # is moved back to the model device before it is appended.
        sample = torch.multinomial(probabilities.to(generator.device), 1,
                                   generator=generator).reshape(1).to(device)
        prefix = torch.cat((prefix, sample))
    return prefix[prompt_ids.numel():].cpu()
