"""Audit the completed run and reevaluate the saved best checkpoints (no training)."""

from __future__ import annotations

import json
import math

import torch
from torchvision.datasets import FashionMNIST

from data import centered_canvases
from models import CNN, MLP
from prepare_data import BASE_DIR, DATA_ROOT, SPLIT_DIR
from prepare_training import ORDER_PATH, PLAN_PATH, PREDICTION_PATH, sha256_file
from train import CHECKPOINT_DIR, evaluate, restore_random_states


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose the same single GPU used for training.")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    results_dir = BASE_DIR / "results"
    results = json.loads((results_dir / "training_results.json").read_text())
    plan = json.loads(PLAN_PATH.read_text())
    assert plan == results["training_plan"]
    assert sha256_file(PREDICTION_PATH) == plan["pretraining_prediction_sha256"]
    assert sha256_file(SPLIT_DIR / "fashion_mnist_indices.pt") == plan["split_file_sha256"]
    assert sha256_file(ORDER_PATH) == plan["epoch_order_file_sha256"]
    orders = torch.load(ORDER_PATH, weights_only=True)
    split_indices = torch.load(SPLIT_DIR / "fashion_mnist_indices.pt", weights_only=True)
    source = FashionMNIST(root=str(DATA_ROOT), train=True, download=False)
    inputs = {
        name: (centered_canvases(source.data[split_indices[name]]).to("cuda:0"),
               source.targets[split_indices[name]].to("cuda:0"))
        for name in ("train", "validation")
    }
    audit = {"all_checks_passed": True, "test_split_evaluated": False, "models": {}}
    for name, model_class in (("MLP", MLP), ("CNN", CNN)):
        entry = results["models"][name]
        history = entry["history"]
        assert [row["epoch"] for row in history] == [1, 2, 3, 4, 5]
        assert all(row["updates"] == 79 and row["training_examples"] == 10000 for row in history)
        assert sum(row["updates"] for row in history) == 395
        assert all(left["elapsed_seconds"] < right["elapsed_seconds"]
                   for left, right in zip(history, history[1:]))
        assert entry["elapsed_seconds"] == history[-1]["elapsed_seconds"] > 0
        selected = min(history, key=lambda row: (row["validation_loss"], row["epoch"]))
        assert entry["selected_epoch"] == selected["epoch"]
        latest = torch.load(CHECKPOINT_DIR / f"{name.lower()}_latest.pt", weights_only=True)
        best = torch.load(CHECKPOINT_DIR / f"{name.lower()}_best.pt", weights_only=True)
        assert latest["completed_epoch"] == 5 and latest["completed_updates"] == 395
        assert best["completed_epoch"] == selected["epoch"]
        assert latest["history"] == history
        assert latest["best_epoch"] == best["completed_epoch"]
        for checkpoint in (latest, best):
            assert checkpoint["training_plan"] == plan
            assert torch.equal(checkpoint["epoch_permutations"], orders["permutations"])
            assert torch.equal(checkpoint["order_generator_state"], orders["generator_state_after_generation"])
            assert checkpoint["scaler_state"] is None
            assert checkpoint["environment"]["visible_device_count"] == 1
            assert checkpoint["environment"]["precision"] == "float32"
            assert checkpoint["random_states"].keys() == {"python", "numpy", "torch_cpu", "torch_cuda"}
            assert len(checkpoint["random_states"]["torch_cuda"]) == 1
            # Verify RNG states can actually be restored after serialization.
            restore_random_states(checkpoint["random_states"])
            assert torch.equal(torch.random.get_rng_state(), checkpoint["random_states"]["torch_cpu"])
            assert torch.equal(torch.cuda.get_rng_state(), checkpoint["random_states"]["torch_cuda"][0])
            group = checkpoint["optimizer_state"]["param_groups"][0]
            assert group["lr"] == 0.001 and group["weight_decay"] == 0
            assert group["betas"] == (0.9, 0.999) and group["eps"] == 1e-8
            assert not group["amsgrad"]
            assert all(state["step"].item() == checkpoint["completed_updates"]
                       for state in checkpoint["optimizer_state"]["state"].values())

        model = model_class().to("cuda:0")
        model.load_state_dict(best["model_state"])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
        optimizer.load_state_dict(best["optimizer_state"])
        assert len(optimizer.state) == len(list(model.parameters()))
        assert sum(p.numel() for p in model.parameters()) == entry["trainable_parameters"]
        reloaded = {split: evaluate(model, images, labels)
                    for split, (images, labels) in inputs.items()}
        for split in ("train", "validation"):
            assert math.isclose(reloaded[split]["loss"], selected[f"{split}_loss"], abs_tol=1e-8)
            assert reloaded[split]["accuracy"] == selected[f"{split}_accuracy"]
        audit["models"][name] = {
            "completed_epochs": 5, "completed_updates": 395,
            "shared_orders_verified": True, "selection_verified": True,
            "optimizer_and_rng_states_loadable": True,
            "reloaded_best_checkpoint_metrics": reloaded,
            "elapsed_seconds": entry["elapsed_seconds"],
        }
        print(f"PASS {name}: 395 updates; best/latest states valid; shared orders verified; "
              "reloaded train/validation metrics match logged values.")
    (results_dir / "training_verification.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print("PASS: original prediction and shared data/order artifacts remain unchanged.")


if __name__ == "__main__":
    main()
