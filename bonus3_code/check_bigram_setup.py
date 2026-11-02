"""Step 3: verify windows, target alignment, model shape, and the bigram loss."""

import hashlib
import json

import torch

from bigram import build_bigram
from losses import next_character_loss
from paths import CONTEXT_LENGTH, RESULTS_DIR, SEED
from text_data import decode, load_data, sample_batch, visible_whitespace


def main() -> None:
    torch.set_num_threads(4)
    data, vocabulary = load_data()
    # Preview uses a fresh independent generator; training starts from its own
    # newly seeded generator, rather than inheriting a consumed preview state.
    preview_generator = torch.Generator().manual_seed(SEED)
    x, y, starts = sample_batch(data["train"], preview_generator, batch_size=32)
    if x.shape != (32, 256) or y.shape != (32, 256):
        raise ValueError("Unexpected batch shapes.")
    if x.dtype != torch.long or y.dtype != torch.long:
        raise ValueError("Inputs and targets must be int64.")
    if not torch.equal(x[:, 1:], y[:, :-1]):
        raise ValueError("Input/target alignment failed.")
    if not torch.equal(y[:, -1], data["train"][starts + CONTEXT_LENGTH]):
        raise ValueError("Final targets are missing or misaligned.")
    for split in ("train", "validation"):
        sequence = data[split]
        last_start = sequence.numel() - CONTEXT_LENGTH - 1
        for start in (0, last_start):
            inputs = sequence[start:start + CONTEXT_LENGTH]
            targets = sequence[start + 1:start + CONTEXT_LENGTH + 1]
            if inputs.numel() != 256 or targets.numel() != 256:
                raise ValueError(f"Boundary window failed in {split}.")
            if targets[-1] != sequence[start + CONTEXT_LENGTH]:
                raise ValueError("Boundary target check failed.")

    model = build_bigram(vocabulary["size"])
    model.eval()
    with torch.inference_mode():
        logits = model(x)
        loss = next_character_loss(logits, y)
        selected_log_probabilities = logits.log_softmax(dim=-1).gather(-1, y.unsqueeze(-1))
        if not torch.allclose(loss, -selected_log_probabilities.mean(), atol=1e-6):
            raise ValueError("Flattened loss does not match all-position mean NLL.")
        first = model(torch.tensor([[0, 1]]))[:, -1]
        second = model(torch.tensor([[2, 1]]))[:, -1]
        if not torch.equal(first, second):
            raise ValueError("The bigram must ignore earlier prefix characters.")
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if count != 4225 or logits.shape != (32, 256, 65) or logits.dtype != torch.float32:
        raise ValueError("Unexpected bigram shape, precision, or parameter count.")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Non-finite logits.")
    repeated = build_bigram(vocabulary["size"])
    if not torch.equal(repeated.table.weight, model.table.weight):
        raise ValueError("Initialization did not reproduce the same model.")
    start = starts[0].item()
    sample = {
        "split": "train", "start": start, "length": CONTEXT_LENGTH,
        "input_text": decode(x[0], vocabulary), "target_text": decode(y[0], vocabulary),
        "final_target_split_index": start + CONTEXT_LENGTH,
        "final_target_character": decode(y[0, -1:], vocabulary),
    }
    if sample["input_text"] + sample["final_target_character"] != decode(data["train"][start:start + 257], vocabulary):
        raise ValueError("Decoded example does not reproduce the complete window.")
    (RESULTS_DIR / "sample_pair.json").write_text(json.dumps(sample, ensure_ascii=True, indent=2) + "\n")
    checks = {
        "input_shape": list(x.shape), "target_shape": list(y.shape),
        "input_dtype": str(x.dtype), "target_dtype": str(y.dtype),
        "logit_shape": list(logits.shape), "logit_dtype": str(logits.dtype),
        "parameters": count, "batch_targets": y.numel(),
        "final_targets_checked": True, "boundary_windows_checked_for_both_splits": True,
        "same_current_character_same_distribution": True,
        "loss_matches_all_position_nll": True, "initialization_reproducible": True,
        "initial_weight_sha256": hashlib.sha256(model.table.weight.detach().numpy().tobytes()).hexdigest(),
        "preview_starts": starts.tolist(), "check_device": "cpu",
    }
    (RESULTS_DIR / "bigram_setup_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print("PASS: int64 input/target (32,256), float32 logits (32,256,65), and 4,225 parameters.")
    print("PASS: all 8,192 targets, final target, both split boundaries, and prefix-independence checks.")
    print(f"Decoded sample start: {start}; last target: {visible_whitespace(sample['final_target_character'])!r}")


if __name__ == "__main__":
    main()
