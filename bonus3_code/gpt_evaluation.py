"""The fixed fresh-window evaluation used for GPT checkpoint selection."""

import math

import torch

from losses import next_character_loss
from text_data import sample_batch


@torch.inference_mode()
def sampled_evaluation(model, data, evaluation_generator, training_generator, dtype):
    training_state = training_generator.get_state().clone()
    model.eval()
    metrics, all_starts = {}, []
    try:
        # Order is prescribed: 200 training batches, then 200 validation batches.
        for split in ("train", "validation"):
            negative_log_likelihoods, starts_for_split = [], []
            count = 0
            for _ in range(200):
                inputs, targets, starts = sample_batch(data[split], evaluation_generator, 64)
                inputs, targets = inputs.to("cuda:0"), targets.to("cuda:0")
                with torch.autocast("cuda", dtype=dtype):
                    logits = model(inputs)
                    losses = next_character_loss(logits, targets, reduction="none")
                negative_log_likelihoods.append(losses.to(torch.float64).sum().item())
                count += targets.numel()
                starts_for_split.append(starts)
            if count != 3276800:
                raise RuntimeError("Each sampled evaluation must score 3,276,800 targets per split.")
            mean = math.fsum(negative_log_likelihoods) / count
            if not math.isfinite(mean):
                raise RuntimeError("Non-finite sampled evaluation loss.")
            metrics[split + "_cross_entropy"] = mean
            metrics[split + "_targets"] = count
            all_starts.append(torch.stack(starts_for_split))
    finally:
        model.train()
    if not torch.equal(training_generator.get_state(), training_state):
        raise RuntimeError("Evaluation advanced the training sampler.")
    return metrics, torch.stack(all_starts)
