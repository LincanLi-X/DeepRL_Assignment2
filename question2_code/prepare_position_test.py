"""Step 1: validate all placements and render the five-condition visual check.

After inspecting the image, run again with --confirm-visual-check to record the
completed visual check before advancing to GPU inference.
"""

import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

from data import centered_canvases
from models import CNN, MLP
from prepare_data import BASE_DIR, SEED, SPLIT_DIR
from prepare_training import sha256_file
from spatial_data import CONDITIONS, CONDITION_LABELS, load_paired_test_split, place_images


RESULTS_DIR = BASE_DIR / "results"
PLACEMENT_PATH = RESULTS_DIR / "position_placements.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-visual-check", action="store_true")
    args = parser.parse_args()
    if args.confirm_visual_check:
        checks = json.loads(PLACEMENT_PATH.read_text())
        figure = BASE_DIR / checks["visual_check_figure"]
        if sha256_file(figure) != checks["visual_check_figure_sha256"]:
            raise ValueError("Placement figure changed; inspect the current image first.")
        checks["visual_inspection_passed"] = True
        checks["visual_inspection_note"] = (
            "Inspected all five canvases: the same complete image is centered or "
            "moved three pixels in the indicated direction, with no clipping or wrap."
        )
        PLACEMENT_PATH.write_text(json.dumps(checks, indent=2) + "\n")
        print("Recorded completed visual inspection.")
        return

    torch.set_num_threads(4)
    dataset = load_paired_test_split()
    training = json.loads((RESULTS_DIR / "training_results.json").read_text())
    if sha256_file(SPLIT_DIR / "fashion_mnist_indices.pt") != training["training_plan"]["split_file_sha256"]:
        raise ValueError("The shared split file changed after training.")
    earlier_report = (BASE_DIR.parent / "Assignment2_submission.md").read_text()
    if "#### Question 2(c):" in earlier_report:
        earlier_report = earlier_report.split("#### Question 2(c):", 1)[0]
        if earlier_report.endswith("\n\n"):
            earlier_report = earlier_report.removesuffix("\n")
    checks = {
        "test_examples": len(dataset),
        "test_index_order_sha256": hashlib.sha256(
            dataset.original_indices.numpy().astype("<i8").tobytes()
        ).hexdigest(),
        "split_file_sha256": sha256_file(SPLIT_DIR / "fashion_mnist_indices.pt"),
        "earlier_report_sha256": hashlib.sha256(earlier_report.encode()).hexdigest(),
        "checkpoint_sha256": {},
        "selected_epochs": {},
        "conditions": [],
        "visual_inspection_passed": False,
        "pixel_dtype": "float32",
        "sum_reduction_dtype": "float64",
        "checked_example": {
            "test_position": 0,
            "official_test_index": dataset.original_indices[0].item(),
            "true_class_id": dataset.labels[0].item(),
            "true_class_name": dataset.classes[dataset.labels[0].item()],
        },
    }
    for name, model_class in (("MLP", MLP), ("CNN", CNN)):
        path = BASE_DIR / training["models"][name]["best_checkpoint"]
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["completed_epoch"] != training["models"][name]["selected_epoch"]:
            raise ValueError("Selected checkpoint does not match validation selection.")
        torch.manual_seed(SEED)
        model = model_class()
        model.load_state_dict(checkpoint["model_state"])
        checks["checkpoint_sha256"][name] = sha256_file(path)
        checks["selected_epochs"][name] = checkpoint["completed_epoch"]

    scaled_originals = dataset.images.to(torch.float32) / 255.0
    reference_sums = scaled_originals.to(torch.float64).sum(dim=(-2, -1))
    example_canvases = []
    for condition, dr, dc in CONDITIONS:
        canvases = place_images(dataset.images, dr, dc)
        row, column = 4 + dr, 4 + dc
        if not torch.equal(canvases[:, 0, row:row + 28, column:column + 28], scaled_originals):
            raise ValueError(f"Image contents changed for {condition}.")
        support = torch.zeros(36, 36, dtype=torch.bool)
        support[row:row + 28, column:column + 28] = True
        if bool((canvases[:, 0, ~support] != 0).any()):
            raise ValueError(f"Nonzero background for {condition}.")
        if condition == "centered" and not torch.equal(canvases, centered_canvases(dataset.images)):
            raise ValueError("Centered test transform differs from training.")
        sums = canvases.to(torch.float64).sum(dim=(1, 2, 3))
        difference = (sums - reference_sums).abs()
        if bool((difference != 0).any()):
            raise ValueError(f"Pixel-sum preservation failed for {condition}.")
        checks["conditions"].append({
            "name": condition, "delta_row": dr, "delta_column": dc,
            "canvas_shape": list(canvases.shape),
            "full_originals_preserved": True, "outside_placement_zero": True,
            "maximum_pixel_sum_difference_all_examples": difference.max().item(),
            "checked_example_pixel_sum": sums[0].item(),
            "checked_example_absolute_sum_difference": difference[0].item(),
        })
        example_canvases.append(canvases[0].clone())
    checks["checked_example"]["original_scaled_pixel_sum"] = reference_sums[0].item()
    checks["checked_example"]["maximum_absolute_pixel_sum_difference"] = max(
        condition["checked_example_absolute_sum_difference"] for condition in checks["conditions"]
    )
    example_tensor = torch.stack(example_canvases)
    torch.save({"canvases": example_tensor, "original_uint8": dataset.images[0],
                "official_test_index": dataset.original_indices[0],
                "true_label": dataset.labels[0]}, RESULTS_DIR / "position_checked_example.pt")

    figure_dir = RESULTS_DIR / "figures"
    figure_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 5, figsize=(10.8, 3.0), layout="constrained")
    for axis, image, (condition, dr, dc) in zip(axes, example_tensor, CONDITIONS):
        axis.imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axis.add_patch(Rectangle((3.5 + dc, 3.5 + dr), 28, 28, fill=False,
                                 edgecolor="#56B4E9", linewidth=1))
        axis.set_title(f"{CONDITION_LABELS[condition]}\n(row, col) = ({dr}, {dc})")
        axis.set_xticks([0, 35])
        axis.set_yticks([0, 35])
    example = checks["checked_example"]
    fig.suptitle(f"Official test index {example['official_test_index']} | "
                 f"True class: {example['true_class_name']} ({example['true_class_id']})", fontsize=12)
    for extension in ("png", "pdf"):
        figure_path = figure_dir / f"q2c_placements.{extension}"
        fig.savefig(figure_path, dpi=220, facecolor="white")
    plt.close(fig)
    checks["visual_check_figure"] = "results/figures/q2c_placements.png"
    checks["visual_check_figure_sha256"] = sha256_file(BASE_DIR / checks["visual_check_figure"])
    PLACEMENT_PATH.write_text(json.dumps(checks, indent=2) + "\n")
    print(f"PASS: all {len(dataset)} images preserved in all five conditions; zero pixel-sum differences.")
    print(f"Checked example: official test index {example['official_test_index']}; "
          f"maximum absolute sum difference {example['maximum_absolute_pixel_sum_difference']}.")
    print(f"Inspect {BASE_DIR / checks['visual_check_figure']} before confirming the visual check.")


if __name__ == "__main__":
    main()
