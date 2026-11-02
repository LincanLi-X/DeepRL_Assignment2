"""Step 5: verify optimizer groups and all 5,000 scheduled learning rates."""

from dataclasses import asdict
import json

from gpt import GPTConfig, build_gpt
from gpt_optimization import build_optimizer, learning_rate
from paths import BASE_DIR, DATA_DIR, RESULTS_DIR, SEED, sha256_file


def main():
    checks = json.loads((RESULTS_DIR / "gpt_setup_checks.json").read_text())
    if not checks["all_checks_passed"]:
        raise RuntimeError("Complete all float32 model checks before configuring training.")
    model = build_gpt()
    optimizer, metadata = build_optimizer(model, fused=False)
    assert metadata["decay_parameter_tensors"] == 26 and metadata["no_decay_parameter_tensors"] == 13
    assert metadata["decay_parameter_count"] == 10740096 and metadata["no_decay_parameter_count"] == 4992
    assert "embeddings.token.weight" in metadata["decay_names"]
    assert "embeddings.position.weight" in metadata["decay_names"]
    assert all("norm" in name for name in metadata["no_decay_names"])
    rates = [learning_rate(i) for i in range(5000)]
    assert rates[0] == 1e-3 / 101 and rates[99] == 1e-3 * 100 / 101
    assert rates[100] == 0.001
    assert all(left < right for left, right in zip(rates[:100], rates[1:101]))
    assert all(left > right for left, right in zip(rates[100:], rates[101:]))
    plan = {"model": asdict(GPTConfig()), "updates": 5000, "batch_size": 64,
            "training_window_seed": SEED, "evaluation_window_seed": 2026,
            "evaluation_interval": 250, "evaluation_batches_per_split": 200,
            "evaluation_targets_per_split": 3276800, "gradient_norm_clip": 1.0,
            "optimizer": {"name": "AdamW", "betas": [0.9, 0.99], "eps": 1e-8,
                          "matrix_weight_decay": 0.1, "layernorm_weight_decay": 0.0},
            "optimizer_group_checks": metadata,
            "learning_rate_checks": {str(i): rates[i] for i in (0, 99, 100, 4999)},
            "input_sha256": {str(path.relative_to(BASE_DIR)): sha256_file(path) for path in (
                DATA_DIR / "text_splits.pt", DATA_DIR / "vocabulary.json")},
            "gpt_source_sha256": sha256_file(BASE_DIR / "gpt.py"),
            "earlier_report_sha256": checks["earlier_report_sha256"],
            "bigram_final_sha256": checks["bigram_final_sha256"],
            "all_checks_passed": True}
    (RESULTS_DIR / "gpt_training_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print("PASS: 26 decayed matrix tensors and 13 zero-decay LayerNorm tensors; tied weights occur once.")
    print(f"PASS: all 5,000 learning rates; boundaries {plan['learning_rate_checks']}")


if __name__ == "__main__":
    main()
