"""Audit the baseline using persisted starts and independent transition-count NLL."""

import hashlib
import json
import math

import torch

from paths import ASSIGNMENT_DIR, BASE_DIR, CHECKPOINT_DIR, CONTEXT_LENGTH, RESULTS_DIR, SEED, sha256_file
from text_data import decode, load_data


def main() -> None:
    torch.set_num_threads(4)
    training = json.loads((RESULTS_DIR / "bigram_training.json").read_text())
    results = json.loads((RESULTS_DIR / "bigram_results.json").read_text())
    setup = json.loads((RESULTS_DIR / "bigram_setup_checks.json").read_text())
    text_checks = json.loads((RESULTS_DIR / "text_checks.json").read_text())
    data, vocabulary = load_data()
    final_path = CHECKPOINT_DIR / "bigram_final.pt"
    final = torch.load(final_path, map_location="cpu", weights_only=True)
    latest = torch.load(CHECKPOINT_DIR / "bigram_latest.pt", map_location="cpu", weights_only=True)
    assert final["completed_updates"] == latest["completed_updates"] == training["completed_updates"] == 500
    assert final["best_validation_result"] is None and final["scaler_state"] is None
    assert torch.equal(final["model_state"]["table.weight"], latest["model_state"]["table.weight"])
    assert final["model_state"]["table.weight"].shape == (65, 65)
    assert final["model_state"]["table.weight"].dtype == torch.float32
    assert training["parameters"] == 4225
    assert training["initial_weight_sha256"] == setup["initial_weight_sha256"]
    assert [row["update"] for row in final["history"]] == list(range(1, 501))
    assert all(row["targets"] == 32 * 256 for row in final["history"])
    assert sum(row["targets"] for row in final["history"]) == 4096000
    group = final["optimizer_state"]["param_groups"][0]
    assert group["lr"] == 0.01 and group["weight_decay"] == 0
    assert group["betas"] == (0.9, 0.999) and group["eps"] == 1e-8
    assert all(state["step"].item() == 500 for state in final["optimizer_state"]["state"].values())
    assert final["random_states"].keys() == {"python", "torch_cpu", "torch_cuda", "training_window_generator"}
    assert len(final["random_states"]["torch_cuda"]) == 1
    for filename, expected in training["source_sha256"].items():
        assert sha256_file(BASE_DIR / filename) == expected, filename
    for filename, expected in training["input_sha256"].items():
        assert sha256_file(BASE_DIR / filename) == expected, filename
    for filename, expected in results["evaluation_source_sha256"].items():
        assert sha256_file(BASE_DIR / filename) == expected, filename
    assert sha256_file(final_path) == training["final_checkpoint_sha256"] == results["final_checkpoint_sha256"]

    saved_starts = torch.load(RESULTS_DIR / "bigram_training_starts.pt", weights_only=True)
    assert saved_starts["starts"].shape == (500, 32)
    replay_generator = torch.Generator().manual_seed(SEED)
    replay = torch.stack([torch.randint(data["train"].numel() - 256, (32,), generator=replay_generator)
                          for _ in range(500)])
    assert torch.equal(replay, saved_starts["starts"])
    assert torch.equal(replay, final["training_window_starts"])
    assert torch.equal(replay_generator.get_state(), final["random_states"]["training_window_generator"])
    assert torch.equal(replay_generator.get_state(), saved_starts["generator_state_after_training"])
    assert replay[0].tolist() == setup["preview_starts"]
    assert bool(((replay >= 0) & (replay + CONTEXT_LENGTH < data["train"].numel())).all())

    validation = data["validation"]
    windows = torch.load(RESULTS_DIR / "full_validation_windows.pt", weights_only=True)
    coverage = torch.zeros(validation.numel() - 1, dtype=torch.long)
    for start, length in zip(windows["starts"].tolist(), windows["lengths"].tolist()):
        coverage[start:start + length] += 1
    assert bool((coverage == 1).all()) and coverage.numel() == 111539
    assert (windows["lengths"] == 256).sum().item() == 435
    assert windows["lengths"][-1].item() == 179
    # Independent baseline-specific calculation: count every adjacent pair and
    # weight its log probability. This does not use the chunk-evaluation code.
    pair_counts = torch.bincount(validation[:-1] * 65 + validation[1:], minlength=65 * 65).reshape(65, 65)
    log_probabilities = final["model_state"]["table.weight"].to(torch.float64).log_softmax(dim=-1)
    reference_ce = -(pair_counts * log_probabilities).sum().item() / pair_counts.sum().item()
    difference = abs(reference_ce - results["validation"]["cross_entropy"])
    assert difference < 1e-6, (reference_ce, results["validation"]["cross_entropy"])
    assert math.isclose(math.exp(results["validation"]["cross_entropy"]), results["validation"]["perplexity"], abs_tol=1e-12)
    generated = torch.load(RESULTS_DIR / "bigram_generation.pt", weights_only=True)
    continuation = (RESULTS_DIR / "bigram_continuation.txt").read_bytes().decode("utf-8")
    assert generated["new_ids"].shape == (128,) and len(continuation) == 128
    assert decode(generated["new_ids"], vocabulary) == continuation == results["generation"]["continuation"]
    assert results["generation"]["prompt"] == "ROMEO:\n"
    assert results["generation"]["seed"] == 1337 and results["generation"]["temperature"] == 1
    sample = json.loads((RESULTS_DIR / "sample_pair.json").read_text())
    start = sample["start"]
    assert sample["input_text"] == decode(data["train"][start:start + 256], vocabulary)
    assert sample["target_text"] == decode(data["train"][start + 1:start + 257], vocabulary)
    assert sample["target_text"][-1] == sample["final_target_character"]
    report = (ASSIGNMENT_DIR / "Assignment2_submission.md").read_text()
    prefix = report.split("#### Bonus 3(a):", 1)[0]
    assert any(hashlib.sha256(text.encode()).hexdigest() == text_checks["earlier_report_sha256"]
               for text in (prefix, prefix.removesuffix("\n")))
    audit = {
        "all_checks_passed": True, "training_updates": 500,
        "training_targets_processed": 4096000, "saved_window_starts_replayed_exactly": True,
        "full_validation_targets": 111539, "every_validation_target_scored_once": True,
        "independent_float64_transition_count_ce": reference_ce,
        "absolute_ce_difference": difference, "ce_tolerance": 1e-6,
        "generation_characters": 128, "generated_text_matches_saved_ids": True,
        "original_report_preserved": True, "checkpoint_and_source_hashes_verified": True,
    }
    (RESULTS_DIR / "bigram_verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("PASS: exactly 500 optimizer steps and all 16,000 saved window starts replayed.")
    print(f"PASS: all 111,539 validation targets covered once; independent CE={reference_ce:.10f}, "
          f"absolute difference={difference:.3g}.")
    print("PASS: 128-character continuation, complete decoded pair, preserved checkpoint/data/report.")


if __name__ == "__main__":
    main()
