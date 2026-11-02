"""Step 2: paired position evaluation of the frozen best-validation models."""

import csv
import hashlib
import json
import os

import torch

from models import CNN, MLP
from prepare_data import BASE_DIR, SPLIT_DIR
from prepare_training import sha256_file
from spatial_data import CONDITIONS, load_paired_test_split, place_images


RESULTS_DIR = BASE_DIR / "results"


@torch.inference_mode()
def predict_logits(model, images, batch_size=128):
    model.eval()
    logits = torch.cat([
        model(images[start:start + batch_size])
        for start in range(0, images.shape[0], batch_size)
    ]).cpu()
    if logits.shape != (images.shape[0], 10) or logits.dtype != torch.float32:
        raise RuntimeError("Unexpected raw logit format.")
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("Non-finite logits.")
    return logits


def main() -> None:
    placements = json.loads((RESULTS_DIR / "position_placements.json").read_text())
    if not placements["visual_inspection_passed"]:
        raise RuntimeError("Complete and confirm the step-1 visual check first.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one server GPU using CUDA_VISIBLE_DEVICES.")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Use the provided launcher for the cuBLAS setting.")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    training = json.loads((RESULTS_DIR / "training_results.json").read_text())
    protected_paths = [
        *sorted((BASE_DIR / "checkpoints").glob("*.pt")),
        SPLIT_DIR / "fashion_mnist_indices.pt",
        RESULTS_DIR / "pretraining_prediction.md",
        RESULTS_DIR / "training_results.json",
    ]
    protected = {str(path.relative_to(BASE_DIR)): sha256_file(path) for path in protected_paths}
    dataset = load_paired_test_split()
    n = len(dataset)
    index_hash = hashlib.sha256(dataset.original_indices.numpy().astype("<i8").tobytes()).hexdigest()
    if index_hash != placements["test_index_order_sha256"]:
        raise ValueError("Test index order changed after step 1.")
    if sha256_file(SPLIT_DIR / "fashion_mnist_indices.pt") != placements["split_file_sha256"]:
        raise ValueError("Test split file changed after step 1.")
    raw_images = dataset.images.to("cuda:0")
    predictions = {}
    all_logits = {}
    rows = []
    metadata = {}
    for name, model_class in (("MLP", MLP), ("CNN", CNN)):
        checkpoint_path = BASE_DIR / training["models"][name]["best_checkpoint"]
        if sha256_file(checkpoint_path) != placements["checkpoint_sha256"][name]:
            raise ValueError("Best checkpoint changed after the placement check.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if sha256_file(BASE_DIR / "models.py") != checkpoint["source_sha256"]["models.py"]:
            raise ValueError("Model implementation changed after training.")
        if checkpoint["completed_epoch"] != training["models"][name]["selected_epoch"]:
            raise ValueError("Checkpoint was not selected by the recorded validation criterion.")
        model = model_class().to("cuda:0")
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise ValueError("Models must remain float32.")
        predictions[name], all_logits[name] = {}, {}
        for condition, dr, dc in CONDITIONS:
            # Every condition starts from the original uint8 images and a fresh
            # canvas. No crop, roll, resize, interpolation, or label transform.
            images = place_images(raw_images, dr, dc)
            logits = predict_logits(model, images)
            predicted = logits.argmax(dim=1)
            all_logits[name][condition] = logits
            predictions[name][condition] = predicted
            centered = predictions[name]["centered"]
            correct = (predicted == dataset.labels)
            same = (predicted == centered)
            row = {
                "model": name, "condition": condition,
                "delta_row": dr, "delta_column": dc, "examples": n,
                "correct": correct.sum().item(),
                "consistent_with_centered": same.sum().item(),
                "prediction_changes": (~same).sum().item(),
                "consistent_but_incorrect": (same & ~correct).sum().item(),
                "accuracy": correct.sum().item() / n,
                "consistency": same.sum().item() / n,
            }
            if condition == "centered" and row["consistency"] != 1:
                raise RuntimeError("Centered consistency must equal one.")
            rows.append(row)
            print(f"{name:3s} {condition:8s}: accuracy={row['accuracy']:.2%}, "
                  f"consistency={row['consistency']:.2%}, "
                  f"changed predictions={row['prediction_changes']}", flush=True)
        metadata[name] = {
            "selected_epoch": checkpoint["completed_epoch"],
            "selection_validation_loss": checkpoint["best_validation_loss"],
            "checkpoint_path": str(checkpoint_path.relative_to(BASE_DIR)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }
        del model

    for relative_path, expected in protected.items():
        if sha256_file(BASE_DIR / relative_path) != expected:
            raise RuntimeError(f"Protected training artifact changed: {relative_path}")
    if len(rows) != 10:
        raise RuntimeError("Expected ten paired-test rows.")
    torch.save({"official_test_indices": dataset.original_indices,
                "labels": dataset.labels, "class_names": dataset.classes,
                "predictions": predictions, "raw_logits": all_logits},
               RESULTS_DIR / "position_predictions.pt")
    result = {
        "test_examples": n, "test_index_order_sha256": index_hash,
        "environment": {
            "gpu_name": torch.cuda.get_device_name(0),
            "visible_device_count": torch.cuda.device_count(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": str(torch.__version__), "precision": "float32",
            "autocast": False, "tf32": False,
        },
        "models": metadata, "rows": rows,
        "all_training_artifacts_unchanged": True,
        "protected_training_artifact_sha256": protected,
        "source_sha256": {
            filename: sha256_file(BASE_DIR / filename)
            for filename in ("evaluate_position.py", "spatial_data.py", "models.py")
        },
        "predictions_sha256": sha256_file(RESULTS_DIR / "position_predictions.pt"),
    }
    (RESULTS_DIR / "position_results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (RESULTS_DIR / "position_metrics.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved all 2,000 paired predictions/logits per condition and the ten-row table.")
    print("Best/latest training checkpoints and original prediction remain unchanged.")


if __name__ == "__main__":
    main()
