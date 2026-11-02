"""Snapshots and RNG state for exact continuation on the same CUDA runtime."""

import random

import torch


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return value


def capture_rng(training_generator):
    return {"python": random.getstate(), "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
            "training_window_generator": training_generator.get_state()}


def restore_rng(states, training_generator):
    random.setstate(states["python"])
    torch.random.set_rng_state(states["torch_cpu"])
    torch.cuda.set_rng_state_all(states["torch_cuda"])
    training_generator.set_state(states["training_window_generator"])


def atomic_save(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def configure_float32_gpu():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES.")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
