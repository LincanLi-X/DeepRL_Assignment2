"""Step 4: prescribed full-model float32 checks, before any training."""

import json

import torch
from torch import nn

from gpt import build_gpt, parameter_fingerprint
from paths import ASSIGNMENT_DIR, CHECKPOINT_DIR, RESULTS_DIR, SEED, sha256_file
from runtime import configure_float32_gpu


def main():
    if not (RESULTS_DIR / "gpt_assembly_checks.json").is_file():
        raise RuntimeError("Complete architecture assembly checks first.")
    configure_float32_gpu()
    model = build_gpt()
    fingerprint = parameter_fingerprint(model)
    repeated = build_gpt()
    assert parameter_fingerprint(repeated) == fingerprint
    del repeated
    assert all(torch.equal(layer.weight, torch.ones_like(layer.weight))
               for layer in model.modules() if isinstance(layer, nn.LayerNorm))
    model = model.to("cuda:0").eval()
    generator = torch.Generator().manual_seed(SEED)
    long_ids = torch.randint(65, (2, 256), generator=generator).to("cuda:0")
    short_ids = long_ids[:, :17]
    changed_ids = long_ids.clone()
    changed_ids[:, 128:] = (changed_ids[:, 128:] + 1) % 65
    with torch.inference_mode():
        short_logits = model(short_ids)
        long_logits, attention = model(long_ids, return_attention=True)
        changed_logits = model(changed_ids)
        explicit_logits = model(short_ids, torch.arange(17, device="cuda:0"))
        retained_logits = model(long_ids[:, 224:], torch.arange(224, 256, device="cuda:0"))
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    row_error = max((weights.sum(dim=-1) - 1).abs().max().item() for weights in attention)
    future_weight = max(weights.triu(diagonal=1).abs().max().item() for weights in attention)
    delta = (long_logits[:, :128] - changed_logits[:, :128]).abs().max().item()
    assert count == 10745088 and len(attention) == 6
    assert short_logits.shape == (2, 17, 65) and long_logits.shape == (2, 256, 65)
    assert retained_logits.shape == (2, 32, 65)
    assert short_logits.dtype == torch.float32 and long_logits.dtype == torch.float32
    assert torch.equal(short_logits, explicit_logits)
    assert row_error <= 1e-5 and future_weight == 0 and delta <= 1e-5
    assert all(bool(torch.isfinite(output).all()) for output in (short_logits, long_logits, changed_logits))
    assert model.output.weight is model.embeddings.token.weight
    model.train()
    checks = {
        "all_checks_passed": True, "check_device": torch.cuda.get_device_name(0),
        "check_precision": "float32; no autocast; TF32 disabled",
        "short_logit_shape": list(short_logits.shape), "long_logit_shape": list(long_logits.shape),
        "explicit_retained_position_shape": list(retained_logits.shape),
        "parameters": count, "maximum_attention_row_sum_error": row_error,
        "maximum_future_attention_weight": future_weight, "prefix_logit_delta": delta,
        "weight_tensor_shared": True, "initialization_reproducible": True,
        "initial_parameter_sha256": fingerprint, "model_restored_to_training_mode": model.training,
        "earlier_report_sha256": sha256_file(ASSIGNMENT_DIR / "Assignment2_submission.md"),
        "bigram_final_sha256": sha256_file(CHECKPOINT_DIR / "bigram_final.pt"),
    }
    (RESULTS_DIR / "gpt_setup_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(f"PASS: {count:,} parameters; short logits {tuple(short_logits.shape)}; long logits {tuple(long_logits.shape)}.")
    print(f"Row-sum error {row_error:.3g}; future weight {future_weight:.3g}; prefix-logit delta {delta:.3g}.")
    print("PASS: true weight sharing, prescribed initialization, retained position IDs, and training-mode restoration.")


if __name__ == "__main__":
    main()
