"""Run Question 2(b) once on one visible GPU, with epoch-end evaluation.

Prepare orders first with prepare_training.py. Set CUDA_VISIBLE_DEVICES to a
single GPU UUID and CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching Python.
Use --resume only to continue the saved run after an interruption.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.datasets import FashionMNIST

from data import centered_canvases
from models import CNN, MLP
from prepare_data import BASE_DIR, DATA_ROOT, SEED, SPLIT_DIR
from prepare_training import (
    BATCH_SIZE, EPOCHS, ORDER_PATH, PLAN_PATH, PREDICTION_PATH,
    TRAIN_SIZE, sha256_file, validate_orders,
)


RESULTS_DIR = BASE_DIR / "results"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"


@torch.no_grad()
def evaluate(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Sample-weighted mean CE and accuracy over every example exactly once."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    weighted_loss = torch.zeros((), dtype=torch.float64, device=images.device)
    correct = torch.zeros((), dtype=torch.long, device=images.device)
    for start in range(0, labels.numel(), batch_size):
        batch_images = images[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]
        logits = model(batch_images)
        mean_loss = criterion(logits, batch_labels)
        weighted_loss += mean_loss.to(torch.float64) * batch_labels.numel()
        correct += (logits.argmax(dim=1) == batch_labels).sum()
    count = labels.numel()
    result = {"loss": weighted_loss.item() / count,
              "accuracy": correct.item() / count, "examples": count}
    if not math.isfinite(result["loss"]):
        raise RuntimeError("Non-finite evaluation loss.")
    return result


def cpu_tree(value):
    """Snapshot state tensors instead of retaining live state_dict references."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def random_states() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        # Store NumPy's state as primitives/tensors for weights_only=True loads.
        "numpy": {"algorithm": numpy_state[0],
                  "keys": torch.from_numpy(numpy_state[1].copy()),
                  "position": numpy_state[2], "has_gauss": numpy_state[3],
                  "cached_gaussian": numpy_state[4]},
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_random_states(states: dict) -> None:
    random.setstate(states["python"])
    numpy_state = states["numpy"]
    np.random.set_state((
        numpy_state["algorithm"], numpy_state["keys"].numpy(),
        numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"],
    ))
    torch.random.set_rng_state(states["torch_cpu"])
    torch.cuda.set_rng_state_all(states["torch_cuda"])


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_history(path: Path, history: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train_model(
    name: str, model_class: type[nn.Module], inputs: dict,
    order_record: dict, plan: dict, environment: dict, resume: bool, emit,
) -> dict:
    random.seed(SEED)
    np.random.seed(SEED)
    # This reset is immediately before construction, as required by the prompt.
    torch.manual_seed(SEED)
    model = model_class().to("cuda:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    latest_path = CHECKPOINT_DIR / f"{name.lower()}_latest.pt"
    best_path = CHECKPOINT_DIR / f"{name.lower()}_best.pt"
    history = []
    completed_epoch = 0
    best_loss, best_epoch = float("inf"), None
    previous_elapsed = 0.0
    if resume and latest_path.exists():
        saved = torch.load(latest_path, map_location="cpu", weights_only=True)
        if saved["training_plan"] != plan or saved["model_name"] != name:
            raise ValueError("Checkpoint configuration does not match the prepared run.")
        if saved["source_sha256"] != environment["source_sha256"]:
            raise ValueError("Training source changed; exact continuation is not possible.")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        restore_random_states(saved["random_states"])
        history = saved["history"]
        completed_epoch = saved["completed_epoch"]
        best_loss, best_epoch = saved["best_validation_loss"], saved["best_epoch"]
        previous_elapsed = saved["elapsed_seconds"]
        emit(f"{name}: resuming after epoch {completed_epoch}")

    # Move the same pre-generated orders to the same GPU before starting timing.
    permutations = order_record["permutations"].to("cuda:0")
    if completed_epoch < EPOCHS:
        torch.cuda.synchronize()
        started = time.perf_counter()
        for epoch_index in range(completed_epoch, EPOCHS):
            model.train()
            order = permutations[epoch_index]
            seen = updates = 0
            for start in range(0, TRAIN_SIZE, BATCH_SIZE):
                batch_positions = order[start:start + BATCH_SIZE]
                batch_images = inputs["train_images"][batch_positions]
                batch_labels = inputs["train_labels"][batch_positions]
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_images)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                seen += batch_labels.numel()
                updates += 1
            if seen != TRAIN_SIZE or updates != plan["batches_per_epoch"]:
                raise RuntimeError("Incorrect example/update count.")

            train_metrics = evaluate(model, inputs["train_images"], inputs["train_labels"])
            validation_metrics = evaluate(model, inputs["validation_images"], inputs["validation_labels"])
            torch.cuda.synchronize()
            elapsed = previous_elapsed + time.perf_counter() - started
            # At epoch five this is the stopping time, before final checkpoint I/O.
            epoch = epoch_index + 1
            row = {"epoch": epoch, "train_loss": train_metrics["loss"],
                   "train_accuracy": train_metrics["accuracy"],
                   "validation_loss": validation_metrics["loss"],
                   "validation_accuracy": validation_metrics["accuracy"],
                   "training_examples": seen, "updates": updates,
                   "elapsed_seconds": elapsed}
            history.append(row)
            improved = validation_metrics["loss"] < best_loss
            if improved:
                best_loss, best_epoch = validation_metrics["loss"], epoch

            checkpoint = {
                "model_name": name, "model_state": cpu_tree(model.state_dict()),
                "optimizer_state": cpu_tree(optimizer.state_dict()),
                "scaler_state": None, "completed_epoch": epoch,
                "completed_updates": epoch * plan["batches_per_epoch"],
                "best_epoch": best_epoch, "best_validation_loss": best_loss,
                "best_validation_accuracy": history[best_epoch - 1]["validation_accuracy"],
                "history": list(history), "elapsed_seconds": elapsed,
                "random_states": random_states(),
                "epoch_permutations": order_record["permutations"].clone(),
                "order_generator_state": order_record["generator_state_after_generation"].clone(),
                "training_plan": plan, "environment": environment,
                "source_sha256": environment["source_sha256"],
            }
            if improved:
                save_checkpoint(best_path, checkpoint)
            save_checkpoint(latest_path, checkpoint)
            write_history(RESULTS_DIR / f"{name.lower()}_history.csv", history)
            emit(f"{name} epoch {epoch}/{EPOCHS}: "
                 f"train CE={train_metrics['loss']:.6f}, acc={train_metrics['accuracy']:.4%}; "
                 f"val CE={validation_metrics['loss']:.6f}, acc={validation_metrics['accuracy']:.4%}; "
                 f"elapsed={elapsed:.3f}s" + (" [best]" if improved else ""))

    selected = history[best_epoch - 1]
    return {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "selected_epoch": best_epoch,
        "selected_validation_loss": selected["validation_loss"],
        "selected_validation_accuracy": selected["validation_accuracy"],
        "elapsed_seconds": history[-1]["elapsed_seconds"],
        "completed_epochs": len(history),
        "total_updates": sum(row["updates"] for row in history),
        "best_checkpoint": str(best_path.relative_to(BASE_DIR)),
        "latest_checkpoint": str(latest_path.relative_to(BASE_DIR)),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU using CUDA_VISIBLE_DEVICES.")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Launch with CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic CUDA.")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    plan = json.loads(PLAN_PATH.read_text())
    for path, key in (
        (SPLIT_DIR / "fashion_mnist_indices.pt", "split_file_sha256"),
        (ORDER_PATH, "epoch_order_file_sha256"),
        (PREDICTION_PATH, "pretraining_prediction_sha256"),
    ):
        if sha256_file(path) != plan[key]:
            raise ValueError(f"Prepared input changed: {path}")
    order_record = torch.load(ORDER_PATH, map_location="cpu", weights_only=True)
    validate_orders(order_record["permutations"])
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    if not args.resume and any(CHECKPOINT_DIR.glob("*.pt")):
        raise FileExistsError("Checkpoints exist; use --resume to continue this run.")

    environment = {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "visible_device_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "process_id": os.getpid(),
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "precision": "float32", "autocast": False, "tf32": False,
        "deterministic_algorithms": True, "cpu_threads": torch.get_num_threads(),
        "allocation": "one server GPU directly selected by UUID; no job scheduler available",
        "deviation": "server GPU used instead of the prescribed free Google Colab GPU",
        "timing": "synchronized training start through final epoch-end evaluation; "
                  "includes intermediate checkpoint writes; excludes data preparation, "
                  "model/optimizer construction, plotting, and final checkpoint writes",
        "source_sha256": {
            filename: sha256_file(BASE_DIR / filename)
            for filename in ("train.py", "models.py", "data.py", "prepare_training.py")
        },
    }
    (RESULTS_DIR / "training_environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    # Deliberately load only the official training source. No test images or
    # predictions enter training, validation, or checkpoint selection.
    saved_splits = torch.load(
        SPLIT_DIR / "fashion_mnist_indices.pt", map_location="cpu", weights_only=True
    )
    source = FashionMNIST(root=str(DATA_ROOT), train=True, download=False)
    inputs = {}
    for name in ("train", "validation"):
        indices = saved_splits[name]
        inputs[f"{name}_images"] = centered_canvases(source.data[indices]).to("cuda:0")
        inputs[f"{name}_labels"] = source.targets[indices].to("cuda:0")

    log_path = RESULTS_DIR / "training.log"
    with log_path.open("a" if args.resume else "x", encoding="utf-8") as log:
        def emit(message):
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()

        emit(f"Single visible GPU: {environment['gpu_name']}; PyTorch {environment['torch_version']}")
        results = {"training_plan": plan, "environment": environment, "models": {}}
        for name, model_class in (("MLP", MLP), ("CNN", CNN)):
            results["models"][name] = train_model(
                name, model_class, inputs, order_record, plan, environment, args.resume, emit
            )
            (RESULTS_DIR / "training_results.json").write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8"
            )
        emit("Completed both five-epoch runs. Test split was not loaded or evaluated.")


if __name__ == "__main__":
    main()
