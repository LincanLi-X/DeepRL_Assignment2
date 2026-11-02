"""Audit paired-test statistics from the persisted predictions, without GPU work."""

import hashlib
import json

import torch

from prepare_data import BASE_DIR
from prepare_training import sha256_file
from spatial_data import CONDITIONS, load_paired_test_split


def main() -> None:
    torch.set_num_threads(4)
    directory = BASE_DIR / "results"
    results = json.loads((directory / "position_results.json").read_text())
    placements = json.loads((directory / "position_placements.json").read_text())
    assert placements["visual_inspection_passed"]
    assert sha256_file(BASE_DIR / placements["visual_check_figure"]) == placements["visual_check_figure_sha256"]
    assert sha256_file(directory / "position_predictions.pt") == results["predictions_sha256"]
    saved = torch.load(directory / "position_predictions.pt", map_location="cpu", weights_only=True)
    dataset = load_paired_test_split()
    assert torch.equal(saved["official_test_indices"], dataset.original_indices)
    assert torch.equal(saved["labels"], dataset.labels)
    assert len(results["rows"]) == 10
    assert {(row["model"], row["condition"]) for row in results["rows"]} == {
        (name, condition) for name in ("MLP", "CNN") for condition, _, _ in CONDITIONS
    }
    labels = saved["labels"].tolist()
    for row in results["rows"]:
        predictions = saved["predictions"][row["model"]][row["condition"]]
        logits = saved["raw_logits"][row["model"]][row["condition"]]
        assert predictions.shape == (2000,) and logits.shape == (2000, 10)
        assert logits.dtype == torch.float32 and bool(torch.isfinite(logits).all())
        assert torch.equal(predictions, logits.argmax(dim=1))
        centered = saved["predictions"][row["model"]]["centered"].tolist()
        predictions = predictions.tolist()
        # Recompute the reported fractions independently with Python counts.
        correct = sum(prediction == target for prediction, target in zip(predictions, labels))
        consistent = sum(prediction == original for prediction, original in zip(predictions, centered))
        stable_wrong = sum(prediction == original and prediction != target
                           for prediction, original, target in zip(predictions, centered, labels))
        assert row["examples"] == len(labels) == 2000
        assert row["correct"] == correct and row["accuracy"] == correct / 2000
        assert row["consistent_with_centered"] == consistent and row["consistency"] == consistent / 2000
        assert row["prediction_changes"] == 2000 - consistent
        assert row["consistent_but_incorrect"] == stable_wrong
        if row["condition"] == "centered":
            assert row["consistency"] == 1
    for path, expected in results["protected_training_artifact_sha256"].items():
        assert sha256_file(BASE_DIR / path) == expected
    for filename, expected in results["source_sha256"].items():
        assert sha256_file(BASE_DIR / filename) == expected
    report = (BASE_DIR.parent / "Assignment2_submission.md").read_text()
    prefix = report.split("#### Question 2(c):", 1)[0]
    assert any(hashlib.sha256(text.encode()).hexdigest() == placements["earlier_report_sha256"]
               for text in (prefix, prefix.removesuffix("\n")))
    assert placements["checked_example"]["maximum_absolute_pixel_sum_difference"] == 0
    assert all(row["maximum_pixel_sum_difference_all_examples"] == 0 for row in placements["conditions"])

    means = {}
    for name in ("MLP", "CNN"):
        shifted = [row for row in results["rows"] if row["model"] == name and row["condition"] != "centered"]
        centered = next(row for row in results["rows"] if row["model"] == name and row["condition"] == "centered")
        means[name] = {
            "mean_shifted_accuracy": sum(row["accuracy"] for row in shifted) / 4,
            "mean_shifted_consistency": sum(row["consistency"] for row in shifted) / 4,
            "centered_accuracy": centered["accuracy"],
            "minimum_shifted_accuracy": min(row["accuracy"] for row in shifted),
            "maximum_shifted_accuracy": max(row["accuracy"] for row in shifted),
        }
    verification = {
        "all_checks_passed": True, "number_of_rows": 10,
        "paired_sample_count_per_row": 2000, "mean_shifted_metrics": means,
        "saved_logits_and_predictions_agree": True,
        "original_test_indices_and_labels_preserved": True,
        "all_training_artifacts_unchanged": True,
        "earlier_report_verified_before_append": True,
    }
    case_path = directory / "position_change_cases.json"
    if case_path.exists():
        cases = json.loads(case_path.read_text())
        assert cases["number_of_pairs"] == len(cases["cases"]) <= 4
        for case in cases["cases"]:
            position, model, condition = case["test_position"], case["model"], case["condition"]
            assert case["official_test_index"] == saved["official_test_indices"][position].item()
            assert case["true_class_id"] == saved["labels"][position].item()
            assert case["centered_prediction_id"] == saved["predictions"][model]["centered"][position].item()
            assert case["shifted_prediction_id"] == saved["predictions"][model][condition][position].item()
            assert case["centered_prediction_id"] != case["shifted_prediction_id"]
            for field in ("true_class", "centered_prediction", "shifted_prediction"):
                assert case[field + "_name"] == saved["class_names"][case[field + "_id"]]
        if cases["cases"]:
            assert sha256_file(BASE_DIR / cases["figure"]) == cases["figure_sha256"]
        verification["changed_prediction_pairs_verified"] = len(cases["cases"])
    (directory / "position_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print("PASS: all ten metrics rows independently match saved logits/predictions and original labels.")
    print("PASS: test pairing, centered consistency, placement checks, and training artifact hashes.")
    for name, mean in means.items():
        print(f"{name}: mean shifted accuracy {mean['mean_shifted_accuracy']:.4%}; "
              f"mean shifted consistency {mean['mean_shifted_consistency']:.4%}.")


if __name__ == "__main__":
    main()
