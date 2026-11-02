"""Prescribed unique-parameter AdamW groups and the exact learning-rate formula."""

import math

import torch


def learning_rate(iteration):
    if not 0 <= iteration < 5000:
        raise ValueError("Training iteration must be in 0 through 4999.")
    if iteration < 100:
        return 1e-3 * (iteration + 1) / 101
    return 1e-4 + (9e-4 / 2) * (1 + math.cos(math.pi * (iteration - 100) / 4900))


def build_optimizer(model, fused=None):
    unique = list(model.named_parameters())
    decay = [(name, parameter) for name, parameter in unique if parameter.ndim >= 2]
    no_decay = [(name, parameter) for name, parameter in unique if parameter.ndim < 2]
    groups = [{"params": [p for _, p in decay], "weight_decay": 0.1},
              {"params": [p for _, p in no_decay], "weight_decay": 0.0}]
    all_parameters = [p for group in groups for p in group["params"]]
    if len({id(p) for p in all_parameters}) != len(list(model.parameters())):
        raise ValueError("Optimizer parameters must be unique and complete.")
    if sum(p is model.embeddings.token.weight for p in all_parameters) != 1:
        raise ValueError("Tied embedding/output weights must be optimized exactly once.")
    if fused is None:
        fused = next(model.parameters()).device.type == "cuda"
    optimizer = torch.optim.AdamW(groups, lr=learning_rate(0), betas=(0.9, 0.99),
                                  eps=1e-8, fused=fused)
    metadata = {"fused": fused, "decay_parameter_tensors": len(decay),
                "no_decay_parameter_tensors": len(no_decay),
                "decay_parameter_count": sum(p.numel() for _, p in decay),
                "no_decay_parameter_count": sum(p.numel() for _, p in no_decay),
                "decay_names": [name for name, _ in decay],
                "no_decay_names": [name for name, _ in no_decay],
                "tied_weight_appears_once": True}
    return optimizer, metadata
