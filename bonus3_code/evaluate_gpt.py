"""Step 7 final diagnostic: reload the selected GPT and compare full-validation CE."""

import csv
import json

import torch

from gpt import build_gpt
from language_eval import full_validation
from paths import BASE_DIR, CHECKPOINT_DIR, RESULTS_DIR, sha256_file
from runtime import configure_float32_gpu
from text_data import load_data


def main():
    audit = json.loads((RESULTS_DIR / "gpt_training_verification.json").read_text())
    training = json.loads((RESULTS_DIR / "gpt_training_results.json").read_text())
    if not audit["all_checks_passed"] or training["completed_iterations"] != 5000:
        raise RuntimeError("Complete and audit all 5,000 iterations before full validation.")
    configure_float32_gpu()
    path = CHECKPOINT_DIR / "gpt_best.pt"
    before = sha256_file(path)
    if before != training["best_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed.")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["completed_iterations"] == training["selected_update"]
    model = build_gpt().to("cuda:0")
    model.load_state_dict(saved["model_state"])
    model.eval()
    dtype = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16}[training["environment"]["autocast_dtype"]]
    data, _ = load_data()
    metrics, starts, lengths = full_validation(model, data["validation"], batch_size=32, autocast_dtype=dtype)
    baseline_windows = torch.load(RESULTS_DIR / "full_validation_windows.pt", weights_only=True)
    assert torch.equal(starts, baseline_windows["starts"])
    assert torch.equal(lengths, baseline_windows["lengths"])
    assert metrics["targets"] == 111539 and metrics["all_targets_scored_once"]
    assert sha256_file(path) == before
    torch.save({"starts": starts, "lengths": lengths}, RESULTS_DIR / "gpt_full_validation_windows.pt")
    baseline = json.loads((RESULTS_DIR / "bigram_results.json").read_text())
    assert sha256_file(CHECKPOINT_DIR / "bigram_final.pt") == baseline["final_checkpoint_sha256"]
    result = {
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "selected_update": training["selected_update"],
        "selected_sampled_validation_ce": training["selected_sampled_validation_ce"],
        "full_validation": metrics, "same_targets_and_chunks_as_bigram": True,
        "autocast_dtype": str(dtype), "scaler_skipped_steps": training["scaler_skipped_steps"],
        "selected_checkpoint_sha256": before,
        "bigram_reference": {"parameters": baseline["parameters"],
                             "cross_entropy": baseline["validation"]["cross_entropy"],
                             "perplexity": baseline["validation"]["perplexity"]},
        "evaluation_source_sha256": {filename: sha256_file(BASE_DIR / filename)
                                     for filename in ("evaluate_gpt.py", "language_eval.py")},
    }
    (RESULTS_DIR / "gpt_results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (RESULTS_DIR / "language_model_comparison.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["model", "parameters", "full_validation_ce", "perplexity", "targets"])
        writer.writeheader()
        writer.writerow({"model": "Bigram", "parameters": baseline["parameters"],
                         "full_validation_ce": baseline["validation"]["cross_entropy"],
                         "perplexity": baseline["validation"]["perplexity"], "targets": 111539})
        writer.writerow({"model": "GPT", "parameters": result["parameters"],
                         "full_validation_ce": metrics["cross_entropy"],
                         "perplexity": metrics["perplexity"], "targets": 111539})
    print(f"Selected update {training['selected_update']}; selection CE {training['selected_sampled_validation_ce']:.6f}.")
    print(f"GPT full validation: CE {metrics['cross_entropy']:.8f}; perplexity {metrics['perplexity']:.8f}; 111,539 targets.")
    print("PASS: identical full-validation target/chunk protocol to bigram; selected weights unchanged.")


if __name__ == "__main__":
    main()
