"""Step 4: train the specified bigram once for exactly 500 Adam updates."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time

import torch

from bigram import build_bigram
from losses import next_character_loss
from paths import (BASE_DIR, CHECKPOINT_DIR, CONTEXT_LENGTH, DATA_DIR, RESULTS_DIR,
                   SEED, TEXT_PATH, sha256_file)
from runtime import atomic_save, capture_rng, configure_float32_gpu, cpu_snapshot, restore_rng
from text_data import load_data, sample_batch


UPDATES = 500
BATCH_SIZE = 32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    setup = json.loads((RESULTS_DIR / "bigram_setup_checks.json").read_text())
    if not setup["final_targets_checked"] or not setup["boundary_windows_checked_for_both_splits"]:
        raise RuntimeError("Complete the setup checks before training.")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Use the provided single-GPU launcher.")
    configure_float32_gpu()
    data, vocabulary = load_data()
    latest_path = CHECKPOINT_DIR / "bigram_latest.pt"
    final_path = CHECKPOINT_DIR / "bigram_final.pt"
    if latest_path.exists() and not args.resume:
        raise FileExistsError("A run exists; use --resume to continue it, not retrain.")

    config = {"seed": SEED, "updates": UPDATES, "batch_size": BATCH_SIZE,
              "context_length": CONTEXT_LENGTH, "vocabulary_size": 65,
              "optimizer": "Adam", "learning_rate": 0.01,
              "betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": 0,
              "initial_weight_std": 0.02, "precision": "float32",
              "selection": "final update; no validation-based checkpoint selection"}
    source_hashes = {filename: sha256_file(BASE_DIR / filename) for filename in (
        "train_bigram.py", "bigram.py", "text_data.py", "losses.py", "runtime.py", "paths.py"
    )}
    input_hashes = {str(path.relative_to(BASE_DIR)): sha256_file(path) for path in (
        TEXT_PATH, DATA_DIR / "text_splits.pt", DATA_DIR / "vocabulary.json"
    )}
    environment = {
        "gpu_name": torch.cuda.get_device_name(0), "visible_device_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "precision": "float32", "autocast": False, "tf32": False,
        "deterministic_algorithms": True,
        "allocation": "one idle server GPU selected by UUID; no job scheduler discovered",
        "deviation": "server GPU instead of the prescribed free Google Colab GPU",
    }
    random.seed(SEED)
    model = build_bigram(vocabulary["size"])
    initial_hash = hashlib.sha256(model.table.weight.detach().numpy().tobytes()).hexdigest()
    if initial_hash != setup["initial_weight_sha256"]:
        raise ValueError("Initial weights differ from the verified initialization.")
    model = model.to("cuda:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    training_generator = torch.Generator().manual_seed(SEED)
    completed, elapsed_before = 0, 0.0
    history, window_starts = [], []
    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=True)
        if checkpoint["config"] != config or checkpoint["source_sha256"] != source_hashes:
            raise ValueError("Configuration/source changed; cannot continue the recorded run exactly.")
        if checkpoint["input_sha256"] != input_hashes:
            raise ValueError("Training data changed.")
        if any(checkpoint["environment"][key] != environment[key] for key in ("gpu_name", "torch_version", "cuda_runtime")):
            raise ValueError("Use the original hardware/runtime for exact continuation.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        restore_rng(checkpoint["random_states"], training_generator)
        completed = checkpoint["completed_updates"]
        elapsed_before = checkpoint["elapsed_seconds"]
        history = checkpoint["history"]
        window_starts = list(checkpoint["training_window_starts"].unbind(0))

    def save(update, elapsed):
        starts = torch.stack(window_starts) if window_starts else torch.empty(0, 32, dtype=torch.long)
        snapshot = {
            "model_state": cpu_snapshot(model.state_dict()),
            "optimizer_state": cpu_snapshot(optimizer.state_dict()), "scaler_state": None,
            "completed_updates": update, "best_validation_result": None,
            "random_states": capture_rng(training_generator),
            "training_window_starts": starts.clone(), "history": list(history),
            "config": config, "environment": environment, "elapsed_seconds": elapsed,
            "input_sha256": input_hashes, "source_sha256": source_hashes,
        }
        atomic_save(snapshot, latest_path)
        if update == UPDATES:
            atomic_save(snapshot, final_path)

    if completed == 0:
        save(0, 0.0)
    torch.cuda.synchronize()
    started = time.perf_counter()
    model.train()
    print(f"Single GPU: {environment['gpu_name']}; float32; {UPDATES} total updates.", flush=True)
    for update in range(completed + 1, UPDATES + 1):
        inputs, targets, starts = sample_batch(data["train"], training_generator, BATCH_SIZE)
        if update == 1 and starts.tolist() != setup["preview_starts"]:
            raise ValueError("Training sampler did not start from the prescribed independent RNG.")
        inputs, targets = inputs.to("cuda:0"), targets.to("cuda:0")
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = next_character_loss(logits, targets)
        value = loss.item()
        if not math.isfinite(value):
            raise RuntimeError("Non-finite training loss.")
        loss.backward()
        optimizer.step()
        history.append({"update": update, "batch_cross_entropy": value, "targets": targets.numel()})
        window_starts.append(starts.clone())
        if update % 100 == 0:
            torch.cuda.synchronize()
            elapsed = elapsed_before + time.perf_counter() - started
            save(update, elapsed)
            print(f"Update {update}/{UPDATES}: batch CE={value:.6f}; elapsed={elapsed:.3f}s", flush=True)
    if len(history) != UPDATES or len(window_starts) != UPDATES:
        raise RuntimeError("Training did not complete all 500 updates.")
    final = torch.load(final_path, map_location="cpu", weights_only=True)
    torch.save({"seed": SEED, "starts": torch.stack(window_starts),
                "generator_state_after_training": training_generator.get_state()},
               RESULTS_DIR / "bigram_training_starts.pt")
    with (RESULTS_DIR / "bigram_training_history.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {"config": config, "environment": environment, "completed_updates": UPDATES,
               "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "training_targets_processed": sum(row["targets"] for row in history),
               "initial_weight_sha256": initial_hash, "input_sha256": input_hashes,
               "source_sha256": source_hashes, "elapsed_seconds": final["elapsed_seconds"],
               "final_checkpoint": "checkpoints/bigram_final.pt",
               "final_checkpoint_sha256": sha256_file(final_path),
               "training_starts_sha256": sha256_file(RESULTS_DIR / "bigram_training_starts.pt")}
    (RESULTS_DIR / "bigram_training.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Completed exactly 500 updates; final/latest checkpoints and all window starts saved.")


if __name__ == "__main__":
    main()
