"""Prepare the 512 fixed validation targets and freeze the part (c) protocol."""

import json

import torch

from paths import ASSIGNMENT_DIR, BASE_DIR, CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR, sha256_file
from text_data import load_data


def main():
    data, _ = load_data()
    previous = json.loads((RESULTS_DIR / "gpt_part_b_verification.json").read_text())
    results = json.loads((RESULTS_DIR / "gpt_results.json").read_text())
    if not previous["all_checks_passed"]:
        raise RuntimeError("Complete and verify part (b) first.")
    if sha256_file(CHECKPOINT_DIR / "gpt_best.pt") != results["selected_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed after part (b).")
    generator = torch.Generator().manual_seed(2027)
    sequence = data["validation"]
    # Exclusive high M-256 gives starts 0 through M-257, inclusive.
    starts = torch.randint(sequence.numel() - 256, (512,), generator=generator)
    target_indices = starts + 256
    targets = sequence[target_indices]
    assert bool(((starts >= 0) & (target_indices < sequence.numel())).all())
    path = RESULTS_DIR / "context_targets.pt"
    payload = {"seed": 2027, "starts": starts, "target_indices": target_indices,
               "targets": targets, "generator_state_after_sampling": generator.get_state()}
    if path.exists():
        existing = torch.load(path, weights_only=True)
        if not all(torch.equal(existing[key], payload[key]) for key in ("starts", "target_indices", "targets")):
            raise ValueError("Existing fixed target set differs; refusing to replace it.")
    else:
        torch.save(payload, path)
    protected = [*sorted(CHECKPOINT_DIR.glob("*.pt")), DATA_DIR / "text_splits.pt",
                 DATA_DIR / "vocabulary.json", BASE_DIR / "shakespeare.txt",
                 BASE_DIR / "gpt.py", BASE_DIR / "language_eval.py", BASE_DIR / "text_data.py",
                 RESULTS_DIR / "gpt_results.json", RESULTS_DIR / "gpt_training_results.json",
                 RESULTS_DIR / "bigram_results.json"]
    earlier_report = (ASSIGNMENT_DIR / "Assignment2_submission.md").read_text()
    if "#### Bonus 3(c):" in earlier_report:
        earlier_report = earlier_report.split("#### Bonus 3(c):", 1)[0]
        if earlier_report.endswith("\n\n"):
            earlier_report = earlier_report.removesuffix("\n")
    import hashlib
    protocol = {
        "selected_update": results["selected_update"],
        "selected_checkpoint_sha256": results["selected_checkpoint_sha256"],
        "context_seed": 2027, "context_lengths": [32, 128, 256], "context_targets": 512,
        "context_target_file_sha256": sha256_file(path),
        "temperatures": [0.7, 1.0, 1.3], "generation_seeds": [1, 2, 3],
        "prompt": "ROMEO:\n", "new_characters": 128,
        "autocast_dtype": results["autocast_dtype"],
        "full_validation_reference_perplexity": results["full_validation"]["perplexity"],
        "earlier_report_sha256": hashlib.sha256(earlier_report.encode()).hexdigest(),
        "protected_artifact_sha256": {str(p.relative_to(BASE_DIR)): sha256_file(p) for p in protected},
    }
    (RESULTS_DIR / "part_c_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"Frozen selected checkpoint: update {results['selected_update']}.")
    print("Saved 512 seed-2027 starts/targets; upper bound M-256, with replacement.")


if __name__ == "__main__":
    main()
