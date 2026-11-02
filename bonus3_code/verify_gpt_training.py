"""Audit the completed GPT run, exact sampler streams, schedule, and checkpoints."""

import hashlib
import json
import math

import torch

from gpt import build_gpt, parameter_fingerprint
from gpt_optimization import learning_rate
from paths import ASSIGNMENT_DIR, BASE_DIR, CHECKPOINT_DIR, RESULTS_DIR, SEED, sha256_file
from text_data import load_data


def main():
    torch.set_num_threads(4)
    results = json.loads((RESULTS_DIR / "gpt_training_results.json").read_text())
    setup = json.loads((RESULTS_DIR / "gpt_setup_checks.json").read_text())
    latest = torch.load(CHECKPOINT_DIR / "gpt_latest.pt", map_location="cpu", weights_only=True)
    best = torch.load(CHECKPOINT_DIR / "gpt_best.pt", map_location="cpu", weights_only=True)
    assert results["completed_iterations"] == latest["completed_iterations"] == 5000
    assert sha256_file(CHECKPOINT_DIR / "gpt_latest.pt") == results["latest_checkpoint_sha256"]
    assert sha256_file(CHECKPOINT_DIR / "gpt_best.pt") == results["best_checkpoint_sha256"]
    assert sha256_file(CHECKPOINT_DIR / "bigram_final.pt") == setup["bigram_final_sha256"]
    for name, expected in results["source_sha256"].items():
        assert sha256_file(BASE_DIR / name) == expected, name
    for name, expected in results["plan"]["input_sha256"].items():
        assert sha256_file(BASE_DIR / name) == expected, name
    evaluations = results["evaluation_history"]
    assert evaluations == latest["evaluation_history"]
    assert [row["update"] for row in evaluations] == list(range(0, 5001, 250))
    assert all(row["train_targets"] == row["validation_targets"] == 3276800 for row in evaluations)
    assert all(math.isfinite(row["train_cross_entropy"]) and math.isfinite(row["validation_cross_entropy"])
               for row in evaluations)
    selected = min(evaluations, key=lambda row: (row["validation_cross_entropy"], row["update"]))
    assert selected["update"] == results["selected_update"] == best["completed_iterations"]
    assert selected["validation_cross_entropy"] == results["selected_sampled_validation_ce"]
    assert latest["best_update"] == best["best_update"] == selected["update"]
    assert best["evaluation_history"] == evaluations[:selected["update"] // 250 + 1]

    history = latest["training_history"]
    assert [row["update"] for row in history] == list(range(1, 5001))
    assert all(row["learning_rate"] == learning_rate(i) for i, row in enumerate(history))
    assert sum(row["scaler_skipped_step"] for row in history) == results["scaler_skipped_steps"]
    if results["environment"]["autocast_dtype"] == "torch.bfloat16":
        assert results["scaler_skipped_steps"] == 0 and latest["scaler_state"] is None
    for checkpoint in (latest, best):
        assert checkpoint["optimizer_steps"] == checkpoint["completed_iterations"] - checkpoint["scaler_skipped_steps"]
        groups = checkpoint["optimizer_state"]["param_groups"]
        assert len(groups) == 2 and groups[0]["weight_decay"] == 0.1 and groups[1]["weight_decay"] == 0
        ids = [index for group in groups for index in group["params"]]
        assert len(ids) == len(set(ids)) == 39
        assert len(groups[0]["params"]) == 26 and len(groups[1]["params"]) == 13
        assert all(group["betas"] == (0.9, 0.99) and group["eps"] == 1e-8 for group in groups)
        if checkpoint["completed_iterations"]:
            assert all(group["lr"] == learning_rate(checkpoint["completed_iterations"] - 1) for group in groups)
        assert all(state["step"].item() == checkpoint["optimizer_steps"]
                   for state in checkpoint["optimizer_state"]["state"].values())
        assert torch.equal(checkpoint["model_state"]["output.weight"], checkpoint["model_state"]["embeddings.token.weight"])
        assert all(value.dtype == torch.float32 for value in checkpoint["model_state"].values())
        assert len(checkpoint["random_states"]["torch_cuda"]) == 1
        assert "evaluation_window_generator" in checkpoint["random_states"]

    data, _ = load_data()
    starts_record = torch.load(RESULTS_DIR / "gpt_training_starts.pt", weights_only=True)
    assert starts_record["starts"].shape == (5000, 64)
    assert torch.equal(starts_record["starts"], latest["training_window_starts"])
    training_generator = torch.Generator().manual_seed(SEED)
    best_training_state = training_generator.get_state().clone() if selected["update"] == 0 else None
    for i in range(5000):
        expected = torch.randint(data["train"].numel() - 256, (64,), generator=training_generator)
        assert torch.equal(expected, starts_record["starts"][i])
        if i + 1 == selected["update"]:
            best_training_state = training_generator.get_state().clone()
    assert torch.equal(training_generator.get_state(), latest["random_states"]["training_window_generator"])
    assert torch.equal(training_generator.get_state(), starts_record["generator_state_after_training"])
    assert torch.equal(best_training_state, best["random_states"]["training_window_generator"])
    assert torch.equal(best["training_window_starts"], starts_record["starts"][:selected["update"]])

    evaluation_record = torch.load(RESULTS_DIR / "gpt_evaluation_starts.pt", weights_only=True)
    assert evaluation_record["starts"].shape == (21, 2, 200, 64)
    assert evaluation_record["updates"].tolist() == list(range(0, 5001, 250))
    assert torch.equal(evaluation_record["starts"], latest["evaluation_window_starts"])
    evaluation_generator = torch.Generator().manual_seed(2026)
    for point, update in enumerate(evaluation_record["updates"].tolist()):
        for split_index, split in enumerate(("train", "validation")):
            expected = torch.stack([torch.randint(data[split].numel() - 256, (64,),
                                                  generator=evaluation_generator) for _ in range(200)])
            assert torch.equal(expected, evaluation_record["starts"][point, split_index])
        if update == selected["update"]:
            assert torch.equal(evaluation_generator.get_state(), best["random_states"]["evaluation_window_generator"])
    assert torch.equal(evaluation_generator.get_state(), latest["random_states"]["evaluation_window_generator"])
    assert torch.equal(evaluation_generator.get_state(), evaluation_record["generator_state_after_evaluations"])

    model = build_gpt()
    assert parameter_fingerprint(model) == setup["initial_parameter_sha256"]
    model.load_state_dict(best["model_state"])
    assert model.output.weight is model.embeddings.token.weight
    assert sum(p.numel() for p in model.parameters()) == 10745088
    prefix = (ASSIGNMENT_DIR / "Assignment2_submission.md").read_text().split("#### Bonus 3(b):", 1)[0]
    assert any(hashlib.sha256(text.encode()).hexdigest() == setup["earlier_report_sha256"]
               for text in (prefix, prefix.removesuffix("\n")))
    verification = {
        "all_checks_passed": True, "training_iterations": 5000,
        "training_windows": 320000, "evaluation_points": 21, "evaluation_windows": 537600,
        "training_and_evaluation_rng_streams_replayed_exactly": True,
        "learning_rate_schedule_verified": True, "optimizer_grouping_verified": True,
        "best_update": selected["update"], "sampled_validation_ce": selected["validation_cross_entropy"],
        "checkpoint_states_and_tied_weights_verified": True,
        "original_bigram_and_report_unchanged": True,
    }
    (RESULTS_DIR / "gpt_training_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print("PASS: 5,000 iterations, all learning rates, optimizer/scaler states, and 21 evaluations.")
    print("PASS: all 320,000 training windows and 537,600 evaluation windows replayed exactly.")
    print(f"PASS: validation-selected update {selected['update']}; loadable tied weights; earlier results unchanged.")


if __name__ == "__main__":
    main()
