"""Run the Question 2(a) input/model checks, without training either model."""

from __future__ import annotations

import json

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import centered_canvases, load_shared_splits
from models import build_models
from prepare_data import BASE_DIR, SEED


RESULTS_DIR = BASE_DIR / "results"
PREDICTION_PATH = RESULTS_DIR / "pretraining_prediction.md"
PRETRAINING_PREDICTION = (
    "I expect the MLP to be more sensitive to position changes than the CNN "
    "because its first layer assigns separate weights to each pixel coordinate, "
    "so shifting an object changes which weights process it. "
    "The CNN reuses local filters across positions, and pooling plus global "
    "average pooling should make its predictions more stable under small translations. "
    "I do not expect exact translation invariance, since strided pooling and "
    "boundary effects can still change the CNN's output."
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def record_prediction() -> str:
    """Create the original prediction once; preserve it on subsequent runs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with PREDICTION_PATH.open("x", encoding="utf-8") as output:
            output.write(PRETRAINING_PREDICTION + "\n")
    except FileExistsError:
        pass
    return PREDICTION_PATH.read_text(encoding="utf-8").strip()


def main() -> None:
    prediction = record_prediction()
    # These checks use CPU float32. They do not choose a device for later training.
    splits = load_shared_splits()
    results = {
        "torch_version": torch.__version__,
        "check_device": "cpu",
        "initialization_seed_per_model": SEED,
        "pretraining_prediction": prediction,
        "input_checks": {},
        "models": {},
    }

    for name, dataset in splits.items():
        canvases = centered_canvases(dataset.images)
        expected_center = dataset.images.to(torch.float32) / 255.0
        center_error = (canvases[:, 0, 4:32, 4:32] - expected_center).abs().max().item()
        border_max = max(
            canvases[:, :, :4, :].abs().max().item(),
            canvases[:, :, 32:, :].abs().max().item(),
            canvases[:, :, :, :4].abs().max().item(),
            canvases[:, :, :, 32:].abs().max().item(),
        )
        require(canvases.shape == (len(dataset), 1, 36, 36), f"{name}: canvas shape")
        require(canvases.dtype == torch.float32, f"{name}: image dtype")
        require(bool(torch.isfinite(canvases).all()), f"{name}: non-finite pixels")
        minimum, maximum = canvases.min().item(), canvases.max().item()
        require(0 <= minimum <= maximum <= 1, f"{name}: pixel range")
        require(center_error == 0 and border_max == 0, f"{name}: placement/scaling")
        require(dataset.labels.shape == (len(dataset),), f"{name}: label shape")
        require(dataset.labels.dtype == torch.long, f"{name}: label dtype")
        require(bool(((dataset.labels >= 0) & (dataset.labels <= 9)).all()), f"{name}: labels")

        images, labels = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
        require(torch.equal(images, canvases[:4]), f"{name}: dataset batching")
        require(torch.equal(labels, dataset.labels[:4]), f"{name}: batch labels")
        results["input_checks"][name] = {
            "dataset_image_shape": list(canvases.shape),
            "checked_batch_shape": list(images.shape),
            "image_dtype": str(images.dtype),
            "pixel_range": [minimum, maximum],
            "label_shape": list(labels.shape),
            "label_dtype": str(labels.dtype),
            "label_range": [dataset.labels.min().item(), dataset.labels.max().item()],
            "maximum_center_scaling_error": center_error,
            "maximum_absolute_border_pixel": border_max,
        }
        del canvases, expected_center

    image_batch, _ = next(iter(DataLoader(splits["train"], batch_size=4, shuffle=False)))
    dummy = torch.zeros(4, 1, 36, 36, dtype=torch.float32)
    models = build_models()
    for name, model in models.items():
        model.eval()
        require(
            not any(isinstance(layer, (nn.Softmax, nn.LogSoftmax)) for layer in model.modules()),
            f"{name}: outputs must be raw logits",
        )
        with torch.inference_mode():
            dummy_logits = model(dummy)
            logits = model(image_batch)
        for output in (dummy_logits, logits):
            require(output.shape == (4, 10), f"{name}: logit shape")
            require(output.dtype == torch.float32, f"{name}: logit dtype")
            require(bool(torch.isfinite(output).all()), f"{name}: non-finite logits")

        layer_parameters = []
        for layer_name, layer in model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                require(layer.bias is not None, f"{name}: missing bias")
                layer_parameters.append({
                    "layer": layer_name,
                    "type": type(layer).__name__,
                    "weights": layer.weight.numel(),
                    "biases": layer.bias.numel(),
                    "total": layer.weight.numel() + layer.bias.numel(),
                })
        count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        results["models"][name] = {
            "dummy_logit_shape": list(dummy_logits.shape),
            "real_batch_logit_shape": list(logits.shape),
            "logit_dtype": str(logits.dtype),
            "finite_logits": True,
            "output_layer": type(model.layers[-1]).__name__,
            "trainable_parameters": count,
            "layer_parameters": layer_parameters,
            "architecture": str(model),
        }

    # Rebuild with the same per-model seeds to check the initial tensors repeat.
    repeated_models = build_models()
    for name, model in models.items():
        require(
            all(torch.equal(value, repeated_models[name].state_dict()[key])
                for key, value in model.state_dict().items()),
            f"{name}: initialization is not reproducible",
        )
        results["models"][name]["initialization_reproducible"] = True

    output_path = RESULTS_DIR / "setup_checks.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for name, check in results["input_checks"].items():
        print(f"{name}: images {check['checked_batch_shape']} {check['image_dtype']}; "
              f"labels {check['label_shape']} {check['label_dtype']}; "
              f"range {check['pixel_range']}; centered contents and zero border verified")
    for name, check in results["models"].items():
        print(f"{name}: logits {check['real_batch_logit_shape']} {check['logit_dtype']}; "
              f"trainable parameters {check['trainable_parameters']:,}; "
              "initialization reproducible")
    print(f"Original pre-training prediction: {PREDICTION_PATH}")
    print(f"Check results: {output_path}")


if __name__ == "__main__":
    main()
