"""Insert the verified Question 2(b) results into the assignment report."""

import json

from prepare_data import BASE_DIR
from prepare_training import PREDICTION_PATH, sha256_file


def main() -> None:
    results_dir = BASE_DIR / "results"
    results = json.loads((results_dir / "training_results.json").read_text())
    audit = json.loads((results_dir / "training_verification.json").read_text())
    if not audit["all_checks_passed"]:
        raise RuntimeError("Verify the completed run before writing the report.")
    if sha256_file(PREDICTION_PATH) != results["training_plan"]["pretraining_prediction_sha256"]:
        raise RuntimeError("The original pre-training prediction has changed.")
    for metric in ("loss", "accuracy"):
        if not (results_dir / "figures" / f"q2b_{metric}.png").is_file():
            raise FileNotFoundError("Generate both training figures first.")

    mlp, cnn = results["models"]["MLP"], results["models"]["CNN"]
    mlast, clast = mlp["history"][-1], cnn["history"][-1]
    mgap = 100 * (mlast["train_accuracy"] - mlast["validation_accuracy"])
    cgap = 100 * (clast["train_accuracy"] - clast["validation_accuracy"])
    ratio = cnn["elapsed_seconds"] / mlp["elapsed_seconds"]
    environment = results["environment"]
    rows = "\n".join(
        rf"| {name} | ${entry['selected_epoch']}$ | ${entry['selected_validation_loss']:.6f}$ | "
        rf"${100 * entry['selected_validation_accuracy']:.2f}\%$ | ${entry['elapsed_seconds']:.3f}$ |"
        for name, entry in results["models"].items()
    )
    section = rf"""#### Question 2(b): Training and Validation [25 points]

**Question.** Train both models under the prescribed shared training order and budget. Plot training and validation loss and accuracy, report each model's selected epoch, validation performance, and elapsed time, and explain differences in optimization, generalization, and training time.

**Answer.** **Training and checkpoint selection.** A dedicated `torch.Generator` seeded with $1337$ generated five consecutive permutations of the $10000$ training positions before training. Both models used these saved permutations for five epochs with batch size $128$, retaining the final batch of $16$: $79$ updates per epoch and $395$ per model. Each model used a fresh Adam optimizer with learning rate $0.001$, default moment settings $(0.9,0.999)$, default $\epsilon=10^{{-8}}$, and zero weight decay. Each update cleared gradients, computed mean cross-entropy from raw logits and integer labels, backpropagated, and updated the parameters.

After each epoch, we evaluated the complete training and validation splits in evaluation mode without gradients. Dataset loss was the sample-weighted mean

$$
\bar{{\ell}} = \frac{{\sum_b n_b\ell_b}}{{\sum_b n_b}},
$$

where $\ell_b$ is the batch mean loss and $n_b$ its size. Accuracy used the largest logit. We saved a new best checkpoint only when validation loss strictly decreased, retaining the earlier epoch on ties. Both best and latest checkpoints include model, optimizer, completed-epoch, and random-generator states. The test split was neither loaded nor evaluated.

**Environment and timing.** Both models ran on one {environment['gpu_name']} GPU using PyTorch `{environment['torch_version']}` and `float32`, with mixed precision and TF32 disabled. This used a server GPU instead of the prescribed free Google Colab GPU. Inputs were prepared and placed on the GPU before timing. We synchronized CUDA before each model's timer started and after its final epoch-end evaluation. Elapsed time includes training, all five full-training/validation evaluations, and intermediate checkpoint writes; it excludes data preparation, model/optimizer construction, plotting, and final checkpoint writes.

**Results.** The curves show full-split metrics measured after each epoch.

![Training and validation cross-entropy for the MLP and CNN](question2_code/results/figures/q2b_loss.png)

*Figure 1. Cross-entropy on the complete training and validation splits. Solid lines show training; dashed lines show validation.*

![Training and validation accuracy for the MLP and CNN](question2_code/results/figures/q2b_accuracy.png)

*Figure 2. Classification accuracy on the same splits, expressed as percentages.*

| Model | Selected epoch | Validation CE (nats/image) | Validation accuracy | Elapsed time (s) |
| --- | --- | --- | --- | --- |
{rows}

For both models, validation loss reached its minimum at epoch $5$. Reloading the saved best checkpoints reproduced the logged training and validation metrics.

**Interpretation.** Both models improved throughout training, but the MLP fit the centered training images better within the fixed budget: its final training CE was ${mlast['train_loss']:.4f}$ with ${100 * mlast['train_accuracy']:.2f}\%$ accuracy, compared with ${clast['train_loss']:.4f}$ and ${100 * clast['train_accuracy']:.2f}\%$ for the CNN. The MLP can use fixed pixel coordinates directly, whereas the CNN classifies spatially averaged local features; these parameterizations need not learn equally quickly with the same optimizer. The CNN's loss was still decreasing at epoch $5$, consistent with incomplete fitting under this budget.

Validation loss decreased at every epoch for both models, although the MLP's training/validation loss gap widened modestly toward the end. The final training-minus-validation accuracy gaps were ${mgap:.2f}$ percentage points for the MLP and ${cgap:.2f}$ for the CNN. The CNN's smaller gap does not imply better classification: its training and validation accuracies were both lower. These centered-image results do not yet measure position sensitivity.

The CNN took ${ratio:.2f}$ times as long despite the similar parameter counts. Shared convolutional filters are applied at many spatial locations, creating more computation and intermediate activations than the MLP's small dense layers; backpropagation and full-split evaluation incur these costs as well. These timings describe one short run on this GPU, rather than a general speed ratio across devices.
"""
    # Raw strings keep the LaTeX source readable above; emit single backslashes.
    section = section.replace("\\\\", "\\")
    report_path = BASE_DIR.parent / "Assignment2_submission.md"
    original = report_path.read_text(encoding="utf-8")
    marker = "#### Question 2(b):"
    if marker in original:
        prefix, remainder = original.split(marker, 1)
        next_section = remainder.find("\n#### ")
        suffix = remainder[next_section:] if next_section >= 0 else ""
        updated = prefix + section + suffix
    else:
        updated = original + ("\n" if not original.endswith("\n\n") else "") + section
    prediction = PREDICTION_PATH.read_text(encoding="utf-8").strip()
    if prediction not in updated:
        raise RuntimeError("Report must retain the original pre-training prediction verbatim.")
    report_path.write_text(updated, encoding="utf-8")
    print(f"Updated {report_path}")


if __name__ == "__main__":
    main()
