"""Write the completed context/temperature experiments without changing earlier answers."""

import hashlib
import json

from part_c_common import assert_protected_unchanged, read_protocol
from paths import ASSIGNMENT_DIR, RESULTS_DIR


def text_block(text):
    return "```text\n" + text + ("" if text.endswith("\n") else "\n") + "```"


def main():
    protocol = read_protocol()
    assert_protected_unchanged(protocol)
    context = json.loads((RESULTS_DIR / "context_results.json").read_text())
    context_audit = json.loads((RESULTS_DIR / "context_verification.json").read_text())
    temperatures = json.loads((RESULTS_DIR / "temperature_results.json").read_text())
    temperature_audit = json.loads((RESULTS_DIR / "temperature_verification.json").read_text())
    if not context_audit["all_checks_passed"] or not temperature_audit["all_checks_passed"]:
        raise RuntimeError("Finish and verify both experiments before writing the report.")
    context_rows = "\n".join(
        rf"| ${row['context_length']}$ | ${row['first_position_id']},\ldots,{row['last_position_id']}$ | ${row['cross_entropy']:.6f}$ |"
        for row in context["rows"]
    )
    temperature_rows = "\n".join(
        rf"| ${row['temperature']:.1f}$ | ${row['mean_R4']:.6f}$ |"
        for row in temperatures["summary"]
    )
    sample_sections = []
    for temperature in protocol["temperatures"]:
        sample = next(row for row in temperatures["records"] if row["temperature"] == temperature and row["seed"] == 1)
        sample_sections.append(rf"**Temperature ${temperature:.1f}$, seed $1$.**" + "\n\n" + text_block(sample["continuation"]))
    samples = "\n\n".join(sample_sections)
    context_order = "decreased monotonically" if context["monotonically_nonincreasing"] else "did not decrease monotonically"
    means = [row["mean_R4"] for row in temperatures["summary"]]
    repetition_order = "decreased monotonically" if all(left >= right for left, right in zip(means, means[1:])) else "were not monotonic"
    middle_sample = next(row for row in temperatures["records"] if row["temperature"] == 1.0 and row["seed"] == 1)
    headings = middle_sample["continuation"].count("CATESBY:")
    section = rf"""#### Bonus 3(c): Context and Temperature [50 points]

**Question.** Using only the selected GPT checkpoint, compare context lengths $32$, $128$, and $256$ on $512$ fixed validation targets while preserving retained position IDs. Report losses and a curve, then test temperatures $0.7$, $1.0$, and $1.3$ with seeds $1$, $2$, and $3$. Report mean four-character repetition rates, show the three seed-$1$ continuations, and explain the findings and limitations.

**Answer.** Both experiments used the frozen update-${protocol['selected_update']}$ checkpoint from part (b), in evaluation mode without gradients, on one server GPU with the same BF16 forward precision. No training or parameter tuning was performed, and checkpoint hashes remained unchanged.

**Experiment 1: earlier context.** For validation sequence $a$ of length $M$, a separate generator seeded with $2027$ sampled $512$ starts with replacement from $0,\ldots,M-257$. We saved these starts and held each target $y_i=a[s_i+256]$ fixed. For each $C\in\{{32,128,256\}}$, the input and explicit position IDs were

$$
X_i^{{(C)}}=a[s_i+256-C:s_i+256],
\qquad q^{{(C)}}=(256-C,\ldots,255).
$$

Each cropped input received a fresh forward pass with its own length-$C$ causal mask. Only the last output position predicted the fixed target:

$$
\ell_C=-\frac{{1}}{{512}}\sum_{{i=0}}^{{511}}\log p_\theta\!\left(y_i\mid X_i^{{(C)}},q^{{(C)}}\right).
$$

Forward passes used BF16 autocast; last-position logits were explicitly converted to float32 for cross-entropy, and the $512$ losses were averaged in float64. Independent CPU calculations from the saved logits verified these means. No full-window hidden states were reused.

| Context length $C$ | Retained position IDs | CE (nats/target character) |
| --- | --- | --- |
{context_rows}

![GPT context-length losses on the same 512 fixed validation targets](bonus3_code/results/figures/gpt_context_loss.png)

*Figure 6. All three losses are labeled; target characters and retained position IDs are held fixed across context lengths.*

The losses {context_order} across these three lengths. Earlier tokens therefore improved mean prediction on this target set, including context beyond $128$ characters. This does not guarantee improvement for every target or another sample. Cropping a model trained with context $256$ tests that model's inference-time use of history; it does not compare models trained separately at different context lengths, and removing history also changes retained tokens' hidden representations. Even $\ell_{{256}}$ need not equal either metric in part (b), because this experiment scores only $512$ fixed final targets.

**Experiment 2: sampling temperature.** We used the same prompt for every continuation, `ROMEO:` followed by a newline:

```text
ROMEO:
```

For each temperature $\tau\in\{{0.7,1.0,1.3\}}$ and seed $r\in\{{1,2,3\}}$, a fresh dedicated CPU sampling generator was seeded once and generated $128$ new characters. Generation used part (b)'s procedure, ordinary position IDs $0,\ldots,T-1$, dropout off, and $p_\tau=\operatorname{{softmax}}(z/\tau)$, without top-k or top-p filtering.

We excluded the prompt and counted all $125$ overlapping four-character substrings in each continuation, including spaces, punctuation, and newlines. With $U_4$ distinct substrings, we computed

$$
R_4=1-\frac{{U_4}}{{128-3}}=1-\frac{{U_4}}{{125}},
\qquad
\overline R_4(\tau)=\frac{{1}}{{3}}\sum_{{r=1}}^3 R_4(g_{{\tau,r}}).
$$

The table reports dimensionless means over the three seeds; all nine raw continuations were retained for the calculations.

| Temperature $\tau$ | Mean $R_4$ |
| --- | --- |
{temperature_rows}

Only seed-$1$ continuations are shown below. Each contains $128$ new characters, excluding the common prompt; spelling, whitespace, and the fixed-length endings are preserved.

{samples}

**Interpretation.** The mean repetition rates {repetition_order} as temperature increased. At temperature $1.0$, the displayed continuation repeats `CATESBY:` ${headings}$ times, illustrating how a short sample can raise the mean; three seeds do not establish a universal temperature/repetition ordering. At a fixed prefix, increasing temperature flattens the next-character distribution, but sampled prefixes then diverge, so repetition is not determined by temperature alone.

The temperature-$0.7$ sample contains recognizable phrasing and a `FRIAR LAURENCE:` heading, although it remains imperfect. The temperature-$1.3$ sample is less repetitive but includes less readable fragments such as `sacraft faction` and `gratisfy`. Lower $R_4$ therefore need not mean better language: it measures local substring reuse, not grammar, meaning, or coherence, and nonsensical text can have mostly unique four-grams.

Generation temperature changes only the sampling distribution. The weights and the temperature-$1$ full-validation definition remain fixed, so the previously measured full-validation perplexity remains ${protocol['full_validation_reference_perplexity']:.6f}$; the three sampling temperatures do not change that metric.
"""
    path = ASSIGNMENT_DIR / "Assignment2_submission.md"
    original = path.read_text(encoding="utf-8")
    marker = "#### Bonus 3(c):"
    if marker in original:
        prefix, remainder = original.split(marker, 1)
        following = remainder.find("\n#### ")
        suffix = remainder[following:] if following >= 0 else ""
        candidates = [prefix, prefix.removesuffix("\n")]
        updated = prefix + section + suffix
    else:
        candidates = [original]
        updated = original + ("\n" if not original.endswith("\n\n") else "") + section
    if not any(hashlib.sha256(value.encode()).hexdigest() == protocol["earlier_report_sha256"] for value in candidates):
        raise RuntimeError("Earlier answers changed; refusing to overwrite them.")
    path.write_text(updated, encoding="utf-8")
    print(f"Updated {path}")


if __name__ == "__main__":
    main()
