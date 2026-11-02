"""Write Bonus 3(b) from the completed, audited run and selected-model diagnostic."""

import hashlib
import json

from paths import ASSIGNMENT_DIR, CHECKPOINT_DIR, RESULTS_DIR, sha256_file


def scientific(value):
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.6e}".split("e")
    return mantissa + r"\times10^{" + str(int(exponent)) + "}"


def main():
    setup = json.loads((RESULTS_DIR / "gpt_setup_checks.json").read_text())
    training = json.loads((RESULTS_DIR / "gpt_training_results.json").read_text())
    audit = json.loads((RESULTS_DIR / "gpt_training_verification.json").read_text())
    results = json.loads((RESULTS_DIR / "gpt_results.json").read_text())
    generation = json.loads((RESULTS_DIR / "gpt_generation_checks.json").read_text())
    if not (audit["all_checks_passed"] and generation["all_checks_passed"]):
        raise RuntimeError("Finish all training, evaluation, and generation checks first.")
    if sha256_file(CHECKPOINT_DIR / "gpt_best.pt") != results["selected_checkpoint_sha256"]:
        raise RuntimeError("The selected checkpoint changed.")
    if not (RESULTS_DIR / "figures" / "gpt_sampled_loss.png").is_file():
        raise FileNotFoundError("Render and inspect the sampled-loss plot first.")
    selected = next(row for row in training["evaluation_history"] if row["update"] == training["selected_update"])
    last = training["evaluation_history"][-1]
    baseline = results["bigram_reference"]
    full = results["full_validation"]
    row_error = scientific(setup["maximum_attention_row_sum_error"])
    section = rf"""#### Bonus 3(b): Build and Train a GPT [70 points]

**Question.** Implement the prescribed GPT, verify its shapes, masking, prefix causality, and weight sharing, and train it for the full fixed budget. Report sampled loss curves and checkpoint selection, compare full-validation results with the bigram baseline, record precision and skipped updates, and implement autoregressive generation.

**Answer.** **Architecture and initialization.** We reused part (a)'s vocabulary, ordered splits, sampler, and next-character loss. The GPT has $6$ pre-LayerNorm blocks, embedding width $384$, $6$ attention heads of width $64$, maximum context $256$, and dropout $0.2$. For $1\leq T\leq256$, it maps $(B,T)$ character IDs to $(B,T,65)$ raw logits. Token and learned absolute position embeddings are added, using default positions $0,\ldots,T-1$ or explicit position IDs. Each block uses a handwritten causal attention module and a $384\to1536\to384$ GELU feed-forward network with separate residual additions. Attention explicitly computes scaled query-key scores, masks future positions before softmax over keys, applies attention dropout, combines values, concatenates heads, and projects the result. Dropout is applied only to embedding sums, attention weights, and the two residual-branch outputs.

All linear layers are bias-free. LayerNorm has a trainable scale, no offset, and $\epsilon=10^{{-5}}$. The final normalized features are projected using the same parameter tensor as the token embedding. Immediately before construction, we set seed $1337$; after tying weights, unique linear/embedding weights were initialized from $\mathcal{{N}}(0,0.02^2)$ and LayerNorm scales to $1$. Attention output projections and second FFN projections were then reinitialized with standard deviation $0.02/\sqrt{{12}}$.

**Pre-training checks.** These checks ran in evaluation mode, without gradients or autocast, using float32 parameters and TF32 disabled. The two length-$256$ inputs matched at positions $0,\ldots,127$ and differed afterward; the prefix check compared their raw logits. All six blocks and all heads were checked.

| Check | Observed result |
| --- | --- |
| Short-input logits | $(2,17,65)$ |
| Maximum-length logits | $(2,256,65)$ |
| Explicit retained positions $224,\ldots,255$ | Logits $(2,32,65)$ |
| Unique trainable parameters | $10745088$ |
| Embedding/output weight is the same parameter | Yes |
| Maximum absolute attention row-sum error | ${row_error}$ |
| Maximum absolute future-position weight | ${setup['maximum_future_attention_weight']:g}$ |
| Maximum prefix-logit difference $\Delta$ | ${setup['prefix_logit_delta']:g}$ |

Initialization was reproducible, and the model was restored to training mode after these checks.

**Optimization and precision.** Each iteration used one batch of $64$ length-$256$ windows. AdamW used $(\beta_1,\beta_2)=(0.9,0.99)$ and $\epsilon=10^{{-8}}$. The $26$ unique matrix tensors, including both embeddings, received weight decay $0.1$; the $13$ LayerNorm scale tensors received zero decay. The tied weight appeared once. Before zero-based iteration $i$, both groups used

$$
\eta_i=
\begin{{cases}}
10^{{-3}}(i+1)/101, & 0\leq i<100,\\
10^{{-4}}+\dfrac{{9\times10^{{-4}}}}{{2}}\left[1+\cos\left(\pi\dfrac{{i-100}}{{4900}}\right)\right], & 100\leq i<5000.
\end{{cases}}
$$

The run used the single server GPU environment reported in Question 2(b), with float32 parameters and BF16 autocast for forward passes and loss computation. Gradients were backpropagated normally and their global norm clipped to $1.0$ before the optimizer step. Fused AdamW was used; no GradScaler was needed. All $5000$ iterations were completed, with ${training['scaler_skipped_steps']}$ skipped updates.

**Sampled evaluation and selection.** The training-window generator was reset to $1337$. A separate generator seeded with $2026$ at initialization advanced continuously, without reseeding, over evaluations at update $0$ and every $250$ completed iterations through $5000$. Each evaluation drew $200$ fresh training batches followed by $200$ validation batches, scoring $3276800$ targets per split. Losses were aggregated by target count in evaluation mode without gradients, using the same autocast precision; evaluation did not advance the training sampler. Training mode was restored afterward. Strictly lower sampled validation CE selected a new checkpoint, retaining the earlier checkpoint on ties. Both selected and latest checkpoints retain optimizer, random-generator, and training-progress states.

![GPT sampled training and validation cross-entropy with the selected checkpoint marked](bonus3_code/results/figures/gpt_sampled_loss.png)

*Figure 5. All $21$ sampled evaluations. The star and vertical line mark the checkpoint selected by sampled validation cross-entropy. Training curves here also use evaluation mode, with dropout off.*

| Checkpoint | Completed iterations | Sampled training CE | Sampled validation CE |
| --- | --- | --- | --- |
| Selected | ${selected['update']}$ | ${selected['train_cross_entropy']:.6f}$ | ${selected['validation_cross_entropy']:.6f}$ |
| Latest | ${last['update']}$ | ${last['train_cross_entropy']:.6f}$ | ${last['validation_cross_entropy']:.6f}$ |

All CE values are in nats/character. Training loss continued to fall after the selected checkpoint while validation loss rose, consistent with overfitting; completing the budget therefore did not make the last checkpoint the best validation model.

**Full-validation comparison.** We reloaded the selected checkpoint and used part (a)'s identical chunk starts, lengths, and $111539$ targets, resetting positions to $0,\ldots,L_s-1$ within each chunk. These are full-validation diagnostics, separate from the sampled checkpoint-selection metric above.

| Model | Unique parameters | Full-validation CE (nats/character) | Perplexity |
| --- | --- | --- | --- |
| Bigram, update $500$ | ${baseline['parameters']}$ | ${baseline['cross_entropy']:.6f}$ | ${baseline['perplexity']:.6f}$ |
| Selected GPT, update ${selected['update']}$ | ${results['parameters']}$ | ${full['cross_entropy']:.6f}$ | ${full['perplexity']:.6f}$ |

The GPT achieved lower held-out next-character loss under this protocol. Its greater capacity and different training budget mean this comparison does not isolate the contribution of attention or context alone. Sampled and full-validation losses need not match because their contexts and target weights differ. The full diagnostic does not select checkpoints, and reuse of validation data means it is not an independent test estimate.

**Causality and generation.** Falling training loss alone cannot establish causality: incorrectly unmasked attention could exploit later input characters to predict the targets. Correct mask-before-softmax attention enforces prefix-only dependence, and the float32 prefix-logit check above provides an implementation check independent of training loss.

The handwritten autoregressive procedure feeds the last $256$ IDs, or the full prefix if shorter, with positions $0,\ldots,T-1$. It divides the final-position logits by temperature $\tau>0$, applies vocabulary softmax, samples with a dedicated generator seeded once per continuation, and appends the ID to the full prefix. Inference uses the same autocast precision, with dropout off and no gradients. Checks confirmed actual prefix appends, truncation of a longer-than-$256$ prompt, and position resets, without modifying weights. The selected checkpoint is retained for part (c).
"""
    path = ASSIGNMENT_DIR / "Assignment2_submission.md"
    original = path.read_text(encoding="utf-8")
    marker = "#### Bonus 3(b):"
    if marker in original:
        prefix, remainder = original.split(marker, 1)
        following = remainder.find("\n#### ")
        suffix = remainder[following:] if following >= 0 else ""
        candidates = [prefix, prefix.removesuffix("\n")]
        updated = prefix + section + suffix
    else:
        candidates = [original]
        updated = original + ("\n" if not original.endswith("\n\n") else "") + section
    if not any(hashlib.sha256(value.encode()).hexdigest() == setup["earlier_report_sha256"] for value in candidates):
        raise RuntimeError("Earlier answers changed; refusing to overwrite them.")
    path.write_text(updated, encoding="utf-8")
    print(f"Updated {path}")


if __name__ == "__main__":
    main()
