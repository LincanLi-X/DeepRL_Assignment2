"""Final integrity check of Bonus 3(b) diagnostic, generation, figure, and report."""

import ast
import hashlib
import json
import math
import re

import torch

from paths import ASSIGNMENT_DIR, BASE_DIR, CHECKPOINT_DIR, RESULTS_DIR, sha256_file


def main():
    setup = json.loads((RESULTS_DIR / "gpt_setup_checks.json").read_text())
    training = json.loads((RESULTS_DIR / "gpt_training_results.json").read_text())
    audit = json.loads((RESULTS_DIR / "gpt_training_verification.json").read_text())
    results = json.loads((RESULTS_DIR / "gpt_results.json").read_text())
    generation = json.loads((RESULTS_DIR / "gpt_generation_checks.json").read_text())
    baseline = json.loads((RESULTS_DIR / "bigram_results.json").read_text())
    assert setup["all_checks_passed"] and audit["all_checks_passed"] and generation["all_checks_passed"]
    assert training["completed_iterations"] == 5000 and len(training["evaluation_history"]) == 21
    assert results["parameters"] == 10745088
    assert results["selected_update"] == audit["best_update"] == training["selected_update"]
    assert results["selected_sampled_validation_ce"] == training["selected_sampled_validation_ce"]
    assert sha256_file(CHECKPOINT_DIR / "gpt_best.pt") == training["best_checkpoint_sha256"] == results["selected_checkpoint_sha256"]
    assert sha256_file(CHECKPOINT_DIR / "gpt_latest.pt") == training["latest_checkpoint_sha256"]
    assert sha256_file(CHECKPOINT_DIR / "bigram_final.pt") == setup["bigram_final_sha256"] == baseline["final_checkpoint_sha256"]
    for name, expected in training["source_sha256"].items():
        assert sha256_file(BASE_DIR / name) == expected
    for name, expected in results["evaluation_source_sha256"].items():
        assert sha256_file(BASE_DIR / name) == expected
    for name, expected in training["plan"]["input_sha256"].items():
        assert sha256_file(BASE_DIR / name) == expected
    assert results["full_validation"]["targets"] == baseline["validation"]["targets"] == 111539
    assert math.isclose(results["full_validation"]["perplexity"], math.exp(results["full_validation"]["cross_entropy"]), abs_tol=1e-12)
    baseline_windows = torch.load(RESULTS_DIR / "full_validation_windows.pt", weights_only=True)
    gpt_windows = torch.load(RESULTS_DIR / "gpt_full_validation_windows.pt", weights_only=True)
    assert all(torch.equal(baseline_windows[key], gpt_windows[key]) for key in ("starts", "lengths"))
    assert generation["records"]["long_prompt"]["forward_lengths"] == [256, 256]
    assert generation["parameters_and_checkpoint_unchanged"] and generation["dropout_off"]
    report = (ASSIGNMENT_DIR / "Assignment2_submission.md").read_text()
    prefix, section = report.split("#### Bonus 3(b):", 1)
    assert any(hashlib.sha256(text.encode()).hexdigest() == setup["earlier_report_sha256"]
               for text in (prefix, prefix.removesuffix("\n")))
    for number in (results["selected_sampled_validation_ce"], results["full_validation"]["cross_entropy"],
                   results["full_validation"]["perplexity"], baseline["validation"]["cross_entropy"],
                   baseline["validation"]["perplexity"]):
        assert f"${number:.6f}$" in section
    assert "$10745088$" in section and f"${results['selected_update']}$" in section
    assert "BF16" in section and "$5000$" in section
    for link in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', report):
        assert (ASSIGNMENT_DIR / link).is_file(), link
    in_display = in_code = False
    for line in report.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if line == "$$":
            in_display = not in_display
        elif not in_display:
            assert line.count("$") % 2 == 0, line
    assert not in_display and not in_code
    assert not any(delimiter in report for delimiter in ("\\(", "\\)", "\\[", "\\]"))
    for path in BASE_DIR.glob("*.py"):
        source = path.read_text()
        compile(source, str(path), "exec")
        # Inspect actual calls/imports so the check does not flag its own list
        # of prohibited APIs or explanatory strings.
        prohibited = {"Transformer", "TransformerEncoder", "TransformerDecoder",
                      "MultiheadAttention", "scaled_dot_product_attention"}
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call):
                called = node.func.attr if isinstance(node.func, ast.Attribute) else (
                    node.func.id if isinstance(node.func, ast.Name) else "")
                assert called not in prohibited, path
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("transformers"), path
    record = {"all_checks_passed": True, "training_iterations": 5000, "evaluations": 21,
              "selected_update": results["selected_update"], "full_validation_targets": 111539,
              "full_validation_ce": results["full_validation"]["cross_entropy"],
              "perplexity": results["full_validation"]["perplexity"],
              "generation_interface_verified": True, "report_metrics_and_image_links_verified": True,
              "earlier_report_and_bigram_unchanged": True,
              "figure_sha256": sha256_file(RESULTS_DIR / "figures" / "gpt_sampled_loss.png")}
    (RESULTS_DIR / "gpt_part_b_verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print("PASS: all eight GPT steps, completed budget, selected/full-validation metrics, and generation interface.")
    print("PASS: original data, bigram, earlier report, source/checkpoint hashes, figures, and math formatting.")


if __name__ == "__main__":
    main()
