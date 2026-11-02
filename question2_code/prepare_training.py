"""Save and validate the shared five-epoch training order before any training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from prepare_data import BASE_DIR, SEED, SPLIT_DIR


EPOCHS = 5
BATCH_SIZE = 128
TRAIN_SIZE = 10000
ORDER_PATH = SPLIT_DIR / "epoch_permutations.pt"
PLAN_PATH = BASE_DIR / "results" / "training_plan.json"
PREDICTION_PATH = BASE_DIR / "results" / "pretraining_prediction.md"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_orders(permutations: torch.Tensor) -> None:
    if permutations.dtype != torch.long or permutations.shape != (EPOCHS, TRAIN_SIZE):
        raise ValueError("Expected five int64 permutations of 10,000 positions.")
    expected = torch.arange(TRAIN_SIZE)
    for epoch, order in enumerate(permutations, start=1):
        if not torch.equal(order.sort().values, expected):
            raise ValueError(f"Epoch {epoch} does not visit every position exactly once.")


def main() -> None:
    if not PREDICTION_PATH.is_file():
        raise FileNotFoundError("Record the original pre-training prediction first.")
    saved_splits = torch.load(
        SPLIT_DIR / "fashion_mnist_indices.pt", map_location="cpu", weights_only=True
    )
    if saved_splits["train"].numel() != TRAIN_SIZE:
        raise ValueError("Unexpected saved training split size.")

    # Independent of data-selection RNGs and model initialization RNGs. Seed
    # once before the epoch loop, then advance continuously through five draws.
    generator = torch.Generator().manual_seed(SEED)
    permutations = torch.stack([
        torch.randperm(TRAIN_SIZE, generator=generator) for _ in range(EPOCHS)
    ])
    validate_orders(permutations)
    payload = {
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "train_size": TRAIN_SIZE,
        "permutations": permutations,
        "generator_state_after_generation": generator.get_state(),
    }
    if ORDER_PATH.exists():
        existing = torch.load(ORDER_PATH, map_location="cpu", weights_only=True)
        if not torch.equal(existing["permutations"], permutations):
            raise ValueError("Existing epoch orders differ; refusing to overwrite them.")
        if not torch.equal(existing["generator_state_after_generation"], generator.get_state()):
            raise ValueError("Existing epoch-order RNG state differs.")
    else:
        torch.save(payload, ORDER_PATH)

    plan = {
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "train_examples_per_epoch": TRAIN_SIZE,
        "batches_per_epoch": (TRAIN_SIZE + BATCH_SIZE - 1) // BATCH_SIZE,
        "last_training_batch_size": TRAIN_SIZE % BATCH_SIZE,
        "updates_per_model": EPOCHS * ((TRAIN_SIZE + BATCH_SIZE - 1) // BATCH_SIZE),
        "same_saved_orders_for_both_models": True,
        "optimizer": {"name": "Adam", "lr": 0.001, "betas": [0.9, 0.999],
                      "eps": 1e-8, "weight_decay": 0, "amsgrad": False},
        "loss": "CrossEntropyLoss on raw logits and integer labels",
        "precision": "float32; no autocast; TF32 disabled",
        "checkpoint_selection": "lowest validation loss; retain earlier epoch on ties",
        "split_file_sha256": sha256_file(SPLIT_DIR / "fashion_mnist_indices.pt"),
        "epoch_order_file_sha256": sha256_file(ORDER_PATH),
        "pretraining_prediction_sha256": sha256_file(PREDICTION_PATH),
        "epoch_order_tensor_sha256": [
            hashlib.sha256(order.numpy().astype("<i8").tobytes()).hexdigest()
            for order in permutations
        ],
        "all_orders_are_valid_permutations": True,
    }
    PLAN_PATH.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"Saved shared orders: {ORDER_PATH}")
    print(f"Validated {EPOCHS} permutations, each containing every training position once.")
    print(f"Per epoch: {plan['batches_per_epoch']} batches; last batch: "
          f"{plan['last_training_batch_size']} images.")
    print(f"Training updates per model: {plan['updates_per_model']}")


if __name__ == "__main__":
    main()
