"""Step 4: append the verified Question 2(c) evidence and interpretation."""

import hashlib
import json

from prepare_data import BASE_DIR
from prepare_training import sha256_file
from spatial_data import CONDITION_LABELS


def main() -> None:
    directory = BASE_DIR / "results"
    results = json.loads((directory / "position_results.json").read_text())
    audit = json.loads((directory / "position_verification.json").read_text())
    placements = json.loads((directory / "position_placements.json").read_text())
    cases = json.loads((directory / "position_change_cases.json").read_text())
    if not audit["all_checks_passed"] or not placements["visual_inspection_passed"]:
        raise RuntimeError("Complete the numerical and visual checks first.")
    for path, expected in results["protected_training_artifact_sha256"].items():
        if sha256_file(BASE_DIR / path) != expected:
            raise RuntimeError(f"Protected training artifact changed: {path}")
    example = placements["checked_example"]
    means = audit["mean_shifted_metrics"]
    stable_wrong = next(row["consistent_but_incorrect"] for row in results["rows"]
                        if row["model"] == "CNN" and row["condition"] == "up")
    table = "\n".join(
        rf"| {row['model']} | {CONDITION_LABELS[row['condition']]} $({row['delta_row']},{row['delta_column']})$ | "
        rf"${100 * row['accuracy']:.2f}\%$ | ${100 * row['consistency']:.2f}\%$ |"
        for row in results["rows"]
    )
    if cases["cases"]:
        case_text = (
            f"Figure 4 shows {cases['number_of_pairs']} centered/shifted pairs in total, "
            "one per model. For each model, we scanned shifts in the fixed order "
            "left, right, up, down and selected the first changed prediction in saved "
            "test order, without filtering by correctness. "
        )
        case_text += " ".join(
            f"For {case['model']}, official test index ${case['official_test_index']}$ "
            f"has true class {case['true_class_name']}; moving "
            f"{case['condition']} changes the prediction from {case['centered_prediction_name']} "
            f"to {case['shifted_prediction_name']}."
            for case in cases["cases"]
        )
        case_text += (
            "\n\n![Centered and shifted examples whose predictions change]"
            "(question2_code/results/figures/q2c_prediction_changes.png)\n\n"
            "*Figure 4. Each row is one paired example. Model, displacement, true class, "
            "official test index, and both predictions are labeled.*"
        )
        if cases["models_without_prediction_changes"]:
            case_text += " No prediction changes occurred for " + ", ".join(cases["models_without_prediction_changes"]) + "."
    else:
        case_text = "Neither model changed any prediction under the four tested shifts, so no changed-prediction pairs are shown."

    section = rf"""#### Question 2(c): Sensitivity to Object Position [30 points]

**Question.** Evaluate both best-validation models on the same test images at five prescribed positions. Report accuracy, consistency with centered predictions, the pixel-sum check, and changed-prediction pairs. Explain the architectural effects and the limits of the evidence.

**Answer.** **Paired conditions and placement check.** We reloaded each model's best-validation checkpoint from epoch $5$ and evaluated the same $2000$ saved test examples in the same order, in evaluation mode without gradients. Inference used the single server GPU and `float32` settings described in part (b); no model was retrained. For each condition, the complete original image was copied onto a fresh zero-filled canvas, with its label unchanged:

$$
(T_\delta X)_{{b,0,r+4+\Delta r,c+4+\Delta c}} = \frac{{x_{{b,r,c}}}}{{255}},
\qquad 0 \leq r,c < 28.
$$

Here $x$ contains raw integer pixels, positive row shifts move down, and positive column shifts move right. The displacements were $(0,0)$, $(0,-3)$, $(0,3)$, $(-3,0)$, and $(3,0)$. Placement used no resizing, interpolation, cropping, or wrapping.

![The same test image at all five prescribed positions](question2_code/results/figures/q2c_placements.png)

*Figure 3. Official test index ${example['official_test_index']}$, true class {example['true_class_name']}. Cyan outlines mark the complete original image's placement within each canvas.*

Visual inspection confirmed all five placements. For this example, the maximum absolute difference between a canvas's pixel sum and the scaled original's pixel sum was

$$
\max_\delta \left|\sum T_\delta X - \sum (x/255)\right| = {example['maximum_absolute_pixel_sum_difference']:g}.
$$

Sums were accumulated in `float64` over the unchanged `float32` pixel values. The same check returned $0$ for every selected test image and condition, and all pixels outside each placement were zero.

**Accuracy and consistency.** For $N=2000$, we computed

$$
A_\delta = \frac{{1}}{{N}}\sum_{{i=1}}^N \mathbf{{1}}[\hat{{y}}(T_\delta X_i)=y_i],
\qquad
C_\delta = \frac{{1}}{{N}}\sum_{{i=1}}^N \mathbf{{1}}[\hat{{y}}(T_\delta X_i)=\hat{{y}}(X_i)].
$$

Each consistency comparison uses the same example and model's centered prediction. Both metrics are percentages below.

| Model | Condition $(\Delta r,\Delta c)$ | Accuracy | Consistency |
| --- | --- | --- | --- |
{table}

The CNN had higher accuracy and consistency at every nonzero shift, despite lower centered accuracy. Averaged equally over the four nonzero directions, consistency was ${100 * means['MLP']['mean_shifted_consistency']:.2f}\%$ for the MLP and ${100 * means['CNN']['mean_shifted_consistency']:.2f}\%$ for the CNN, supporting the original position-sensitivity prediction in part (a).

**Changed predictions.** {case_text}

**Architectural interpretation.** The MLP assigns separate weights to pixel coordinates, so a shift changes which weights receive an object's features. Weights connected to always-zero training-border pixels receive zero data gradient; translated objects can activate these untrained connections. The CNN instead learns local filters shared across locations. Pooling can reduce sensitivity to small feature displacements, and global average pooling reduces dependence on feature location. However, stride-$2$ pooling changes alignment under a $3$-pixel shift, and finite boundaries and zero padding can change features, so these operations do not guarantee exact invariance.

**Consistency versus correctness.** High consistency can coexist with low accuracy because two predictions can agree while both are wrong. A classifier that always predicts one class would have $100\%$ consistency but only $10\%$ accuracy on this balanced test set. In our results, the CNN's upward-shift condition included ${stable_wrong}$ examples with an unchanged but incorrect prediction.

**Invariance and scope.** Every nonzero-shift consistency was below $100\%$ for both models, so these checkpoints already fail exact prediction invariance on the tested examples. Even perfect consistency on a finite set would not prove invariance for all inputs or translations, or equality of raw logits. Each architecture was trained once, so the comparison does not estimate variation across initializations or training orders. Similar parameter counts control model size, but depth, connectivity, weight sharing, pooling, and aggregation change together; their individual contributions cannot be isolated without controlled ablations.
"""
    report_path = BASE_DIR.parent / "Assignment2_submission.md"
    original = report_path.read_text(encoding="utf-8")
    marker = "#### Question 2(c):"
    if marker in original:
        prefix, remainder = original.split(marker, 1)
        next_section = remainder.find("\n#### ")
        suffix = remainder[next_section:] if next_section >= 0 else ""
        earlier_candidates = [prefix, prefix.removesuffix("\n")]
        updated = prefix + section + suffix
    else:
        earlier_candidates = [original]
        updated = original + ("\n" if not original.endswith("\n\n") else "") + section
    if not any(hashlib.sha256(text.encode()).hexdigest() == placements["earlier_report_sha256"]
               for text in earlier_candidates):
        raise RuntimeError("Earlier report content changed; refusing to overwrite it.")
    report_path.write_text(updated, encoding="utf-8")
    print(f"Updated {report_path}")


if __name__ == "__main__":
    main()
