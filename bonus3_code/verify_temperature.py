"""Independently recompute all nine R4 values and three means from raw text."""

from collections import Counter
from fractions import Fraction
import hashlib
import json
import math

import torch

from part_c_common import assert_protected_unchanged, read_protocol
from paths import ASSIGNMENT_DIR, BASE_DIR, RESULTS_DIR, sha256_file
from text_data import decode, load_data


def main():
    protocol = read_protocol()
    assert_protected_unchanged(protocol)
    results = json.loads((RESULTS_DIR / "temperature_results.json").read_text())
    generated = torch.load(RESULTS_DIR / "temperature_generation_ids.pt", weights_only=True)
    _, vocabulary = load_data()
    assert len(results["records"]) == 9
    expected_pairs = {(temperature, seed) for temperature in (0.7, 1.0, 1.3) for seed in (1, 2, 3)}
    assert {(row["temperature"], row["seed"]) for row in results["records"]} == expected_pairs
    exact_rates = {}
    for row in results["records"]:
        path = BASE_DIR / row["file"]
        text = path.read_bytes().decode("utf-8")
        assert len(text) == row["new_characters"] == 128
        assert text == row["continuation"] and sha256_file(path) == row["file_sha256"]
        ids = generated[str(row["temperature"])][str(row["seed"])]["new_ids"]
        assert ids.shape == (128,) and ids.dtype == torch.long
        assert decode(ids, vocabulary) == text
        # Counter-based duplicate counting is independent of the generation
        # script's set-based calculation, and retains every whitespace character.
        counts = Counter(text[start:start + 4] for start in range(125))
        assert sum(counts.values()) == 125
        duplicate_windows = sum(count - 1 for count in counts.values())
        reference = Fraction(duplicate_windows, 125)
        assert row["unique_fourgrams"] == len(counts)
        assert row["fourgram_windows"] == 125
        assert math.isclose(float(reference), row["R4"], abs_tol=1e-12)
        exact_rates[(row["temperature"], row["seed"])] = reference
    for summary in results["summary"]:
        temperature = summary["temperature"]
        exact_mean = sum((exact_rates[(temperature, seed)] for seed in (1, 2, 3)), Fraction()) / 3
        assert math.isclose(float(exact_mean), summary["mean_R4"], abs_tol=1e-12)
    for filename, expected in results["source_sha256"].items():
        assert sha256_file(BASE_DIR / filename) == expected
    assert results["prompt"] == "ROMEO:\n"
    assert results["dropout_off"] and results["no_top_k_or_top_p"]
    assert results["selected_checkpoint_sha256"] == protocol["selected_checkpoint_sha256"]
    assert results["reference_tau1_full_validation_perplexity"] == protocol["full_validation_reference_perplexity"]
    prefix = (ASSIGNMENT_DIR / "Assignment2_submission.md").read_text().split("#### Bonus 3(c):", 1)[0]
    assert any(hashlib.sha256(text.encode()).hexdigest() == protocol["earlier_report_sha256"]
               for text in (prefix, prefix.removesuffix("\n")))
    audit = {"all_checks_passed": True, "continuations": 9, "new_characters_each": 128,
             "fourgram_windows_each": 125, "prompt_excluded_from_metrics": True,
             "raw_whitespace_preserved": True, "all_nine_rates_independently_verified": True,
             "all_three_means_verified_with_exact_fractions": True,
             "frozen_artifacts_and_earlier_report_unchanged": True}
    (RESULTS_DIR / "temperature_verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("PASS: all nine raw texts/IDs have 128 new characters; prompt excluded and whitespace preserved.")
    print("PASS: nine R4 values and three means independently verified using exact fractions.")
    print("PASS: selected model, original validation metric, and earlier report remain unchanged.")


if __name__ == "__main__":
    main()
