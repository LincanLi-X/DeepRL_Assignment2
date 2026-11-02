"""Experiment 1: crop inputs, preserve positions/targets, and score last logits only."""

import csv
import json

import torch
from torch.nn import functional as F

from gpt import parameter_fingerprint
from part_c_common import assert_protected_unchanged, load_selected_model
from paths import BASE_DIR, RESULTS_DIR, sha256_file
from text_data import load_data


def main():
    model, dtype, protocol = load_selected_model()
    before = parameter_fingerprint(model)
    data, _ = load_data()
    selected = torch.load(RESULTS_DIR / "context_targets.pt", weights_only=True)
    targets = selected["targets"]
    sequence = data["validation"]
    assert torch.equal(targets, sequence[selected["starts"] + 256])
    saved = {"targets": targets.clone(), "inputs": {}, "positions": {},
             "last_logits": {}, "per_target_losses": {}}
    rows = []
    observed_positions = []
    def position_hook(module, arguments):
        observed_positions.append(arguments[0].detach().cpu().clone())
    handle = model.embeddings.position.register_forward_pre_hook(position_hook)
    try:
        for length in protocol["context_lengths"]:
            positions = torch.arange(256 - length, 256, dtype=torch.long)
            offsets = torch.arange(256 - length, 256, dtype=torch.long)
            inputs = sequence[selected["starts"][:, None] + offsets]
            assert torch.equal(inputs[:, -1], sequence[selected["target_indices"] - 1])
            observed_positions.clear()
            loss_batches, logit_batches = [], []
            # Each cropped batch receives a new complete forward pass. No
            # full-window hidden states or cached keys/values are reused.
            with torch.inference_mode():
                for start in range(0, 512, 32):
                    batch_inputs = inputs[start:start + 32].to("cuda:0")
                    batch_targets = targets[start:start + 32].to("cuda:0")
                    explicit_positions = positions.to("cuda:0")
                    with torch.autocast("cuda", dtype=dtype):
                        logits = model(batch_inputs, position_ids=explicit_positions)
                        last_logits = logits[:, -1, :]
                        # Explicit float32 input avoids low-precision CE
                        # rounding on BF16 logits in this CUDA runtime.
                        losses = F.cross_entropy(last_logits.float(), batch_targets, reduction="none")
                    loss_batches.append(losses.float().cpu())
                    logit_batches.append(last_logits.float().cpu())
            per_target = torch.cat(loss_batches)
            last_logits = torch.cat(logit_batches)
            assert per_target.shape == (512,) and last_logits.shape == (512, 65)
            assert len(observed_positions) == 16
            assert all(torch.equal(actual, positions) for actual in observed_positions)
            assert bool(torch.isfinite(per_target).all())
            mean = per_target.to(torch.float64).mean().item()
            saved["inputs"][str(length)] = inputs
            saved["positions"][str(length)] = positions
            saved["last_logits"][str(length)] = last_logits
            saved["per_target_losses"][str(length)] = per_target
            rows.append({"context_length": length, "first_position_id": 256 - length,
                         "last_position_id": 255, "targets": 512, "cross_entropy": mean,
                         "fresh_forward_batches": 16, "explicit_positions_observed": True,
                         "loss_dtype": str(losses.dtype)})
            print(f"Context {length:3d}: CE={mean:.8f} nats/target; positions {256-length}-255; 512 fixed targets.")
    finally:
        handle.remove()
    assert parameter_fingerprint(model) == before
    assert_protected_unchanged(protocol)
    torch.save(saved, RESULTS_DIR / "context_inference.pt")
    result = {"selected_update": protocol["selected_update"],
              "selected_checkpoint_sha256": protocol["selected_checkpoint_sha256"],
              "seed": 2027, "target_count": 512, "rows": rows,
              "monotonically_nonincreasing": all(left["cross_entropy"] >= right["cross_entropy"]
                                                   for left, right in zip(rows, rows[1:])),
              "autocast_dtype": str(dtype), "gpu_name": torch.cuda.get_device_name(0),
              "loss_precision": "float32 logits for CE; float64 mean over 512 target losses",
              "visible_device_count": torch.cuda.device_count(),
              "inference_only": True, "frozen_artifacts_unchanged": True,
              "inference_file_sha256": sha256_file(RESULTS_DIR / "context_inference.pt"),
              "source_sha256": {name: sha256_file(BASE_DIR / name) for name in (
                  "context_experiment.py", "part_c_common.py", "gpt.py")}}
    (RESULTS_DIR / "context_results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (RESULTS_DIR / "context_losses.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
