"""Independently check fixed targets, actual cropped inputs, and last-position NLL."""

import json

import torch
from torch.nn import functional as F

from part_c_common import assert_protected_unchanged, read_protocol
from paths import BASE_DIR, RESULTS_DIR, sha256_file
from text_data import load_data


def main():
    torch.set_num_threads(4)
    protocol = read_protocol()
    assert_protected_unchanged(protocol)
    data, _ = load_data()
    selected = torch.load(RESULTS_DIR / "context_targets.pt", weights_only=True)
    saved = torch.load(RESULTS_DIR / "context_inference.pt", weights_only=True)
    results = json.loads((RESULTS_DIR / "context_results.json").read_text())
    assert sha256_file(RESULTS_DIR / "context_inference.pt") == results["inference_file_sha256"]
    generator = torch.Generator().manual_seed(2027)
    expected = torch.randint(data["validation"].numel() - 256, (512,), generator=generator)
    assert torch.equal(expected, selected["starts"])
    assert torch.equal(generator.get_state(), selected["generator_state_after_sampling"])
    assert torch.equal(selected["target_indices"], expected + 256)
    assert torch.equal(selected["targets"], data["validation"][expected + 256])
    assert torch.equal(saved["targets"], selected["targets"])
    assert [row["context_length"] for row in results["rows"]] == [32, 128, 256]
    reference_errors = {}
    for row in results["rows"]:
        length = row["context_length"]
        inputs, positions = saved["inputs"][str(length)], saved["positions"][str(length)]
        assert inputs.shape == (512, length)
        assert torch.equal(positions, torch.arange(256 - length, 256))
        for position, start in enumerate(expected.tolist()):
            assert torch.equal(inputs[position], data["validation"][start + 256 - length:start + 256])
        assert saved["last_logits"][str(length)].shape == (512, 65)
        independent_losses = F.cross_entropy(saved["last_logits"][str(length)], selected["targets"], reduction="none")
        reference_mean = independent_losses.to(torch.float64).mean().item()
        error = abs(reference_mean - row["cross_entropy"])
        assert error < 1e-6
        assert saved["per_target_losses"][str(length)].shape == (512,)
        assert saved["per_target_losses"][str(length)].double().mean().item() == row["cross_entropy"]
        assert row["targets"] == 512 and row["fresh_forward_batches"] == 16
        assert row["explicit_positions_observed"]
        reference_errors[str(length)] = error
    for filename, expected_hash in results["source_sha256"].items():
        assert sha256_file(BASE_DIR / filename) == expected_hash
    audit = {"all_checks_passed": True, "seed_and_fixed_targets_replayed": True,
             "all_inputs_match_original_cropped_tokens": True,
             "retained_position_ids_verified": True, "only_last_target_scored": True,
             "independent_cpu_ce_absolute_errors": reference_errors,
             "all_frozen_artifacts_unchanged": True}
    (RESULTS_DIR / "context_verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("PASS: exact seed-2027 targets, all cropped token inputs, preserved position IDs, and last-target-only losses.")
    print("PASS: independent CPU loss calculations and frozen checkpoint/source hashes.")


if __name__ == "__main__":
    main()
