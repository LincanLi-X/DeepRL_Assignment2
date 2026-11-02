"""Step 3: show one deterministically selected prediction change per model."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from prepare_data import BASE_DIR
from prepare_training import sha256_file
from spatial_data import CONDITIONS, CONDITION_LABELS, load_paired_test_split, place_images


def main() -> None:
    directory = BASE_DIR / "results"
    audit = json.loads((directory / "position_verification.json").read_text())
    if not audit["all_checks_passed"]:
        raise RuntimeError("Verify the paired metrics before selecting examples.")
    results = json.loads((directory / "position_results.json").read_text())
    if sha256_file(directory / "position_predictions.pt") != results["predictions_sha256"]:
        raise RuntimeError("The saved predictions changed.")
    saved = torch.load(directory / "position_predictions.pt", weights_only=True)
    dataset = load_paired_test_split()
    cases = []
    models_without_changes = []
    for name in ("MLP", "CNN"):
        centered = saved["predictions"][name]["centered"]
        for condition, dr, dc in CONDITIONS[1:]:
            shifted = saved["predictions"][name][condition]
            changed = torch.nonzero(shifted != centered, as_tuple=True)[0]
            if changed.numel():
                position = changed[0].item()
                true_label = dataset.labels[position].item()
                before, after = centered[position].item(), shifted[position].item()
                cases.append({
                    "model": name, "condition": condition,
                    "delta_row": dr, "delta_column": dc,
                    "test_position": position,
                    "official_test_index": dataset.original_indices[position].item(),
                    "true_class_id": true_label, "true_class_name": dataset.classes[true_label],
                    "centered_prediction_id": before, "centered_prediction_name": dataset.classes[before],
                    "shifted_prediction_id": after, "shifted_prediction_name": dataset.classes[after],
                })
                break
        else:
            models_without_changes.append(name)

    case_record = {
        "selection_rule": "One example per model: scan shifts left, right, up, down, "
                          "and select the first changed prediction in saved test order. "
                          "No selection by correctness or class.",
        "number_of_pairs": len(cases), "cases": cases,
        "models_without_prediction_changes": models_without_changes,
    }
    if cases:
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "pdf.fonttype": 42, "ps.fonttype": 42})
        fig, axes = plt.subplots(len(cases), 2, figsize=(6.8, 3.2 * len(cases)),
                                 squeeze=False, layout="constrained")
        for row, case in enumerate(cases):
            raw_image = dataset.images[case["test_position"]]
            for column, condition, dr, dc, prediction_name, prediction_id in (
                (0, "centered", 0, 0, case["centered_prediction_name"], case["centered_prediction_id"]),
                (1, case["condition"], case["delta_row"], case["delta_column"],
                 case["shifted_prediction_name"], case["shifted_prediction_id"]),
            ):
                axis = axes[row, column]
                image = place_images(raw_image, dr, dc)
                axis.imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                axis.set_title(f"{case['model']} | {CONDITION_LABELS[condition]} ({dr}, {dc})\n"
                               f"Prediction: {prediction_name} ({prediction_id})", fontsize=10)
                axis.set_xlabel(f"True: {case['true_class_name']} ({case['true_class_id']})\n"
                                f"Official test index: {case['official_test_index']}", fontsize=10)
                axis.set_xticks([])
                axis.set_yticks([])
        fig.suptitle("Prediction changes after a 3-pixel translation", fontsize=12)
        figure_dir = directory / "figures"
        for extension in ("png", "pdf"):
            fig.savefig(figure_dir / f"q2c_prediction_changes.{extension}", dpi=220, facecolor="white")
        plt.close(fig)
        case_record["figure"] = "results/figures/q2c_prediction_changes.png"
        case_record["figure_sha256"] = sha256_file(BASE_DIR / case_record["figure"])
    (directory / "position_change_cases.json").write_text(json.dumps(case_record, indent=2) + "\n")
    for case in cases:
        print(f"{case['model']}: official test index {case['official_test_index']}, "
              f"true {case['true_class_name']}; {case['condition']}: "
              f"{case['centered_prediction_name']} -> {case['shifted_prediction_name']}")
    print(f"Saved {len(cases)} centered/shifted pairs in total.")


if __name__ == "__main__":
    main()
