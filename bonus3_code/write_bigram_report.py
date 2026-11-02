"""Step 6: write the verified baseline results and explain its dependencies."""

import hashlib
import json

from paths import ASSIGNMENT_DIR, RESULTS_DIR
from text_data import visible_whitespace


def window_block(text):
    return "\n".join(f"{start:03d}:{start + 32:03d}  {visible_whitespace(text[start:start + 32])}"
                     for start in range(0, len(text), 32))


def main() -> None:
    text_checks = json.loads((RESULTS_DIR / "text_checks.json").read_text())
    results = json.loads((RESULTS_DIR / "bigram_results.json").read_text())
    audit = json.loads((RESULTS_DIR / "bigram_verification.json").read_text())
    sample = json.loads((RESULTS_DIR / "sample_pair.json").read_text())
    if not audit["all_checks_passed"] or results["completed_updates"] != 500:
        raise RuntimeError("Complete all training and verification before writing the report.")
    continuation = results["generation"]["continuation"]
    if len(continuation) != 128:
        raise RuntimeError("The continuation must contain exactly 128 new characters.")
    input_block, target_block = window_block(sample["input_text"]), window_block(sample["target_text"])
    generation_block = "```text\n" + continuation + ("" if continuation.endswith("\n") else "\n") + "```"
    section = rf"""#### Bonus 3(a): Data and a Bigram Baseline [30 points]

**Question.** Encode and split the supplied Shakespeare text, construct next-character batches, and implement and train the prescribed bigram model. Report the data and shape checks, a decoded input/target pair, parameter count, full-validation cross-entropy and perplexity, and a $128$-character continuation. Explain one dependency the model can learn and one it cannot.

**Answer.** **Text encoding and ordered splits.** We read the supplied text as UTF-8, preserving capitalization, punctuation, and all whitespace. Sorting its distinct characters produced one shared vocabulary with IDs $0,\ldots,64$. Encoding and decoding reproduced the full original text and its bytes exactly.

| Item | Checked count |
| --- | --- |
| Full text | $1115394$ characters |
| Vocabulary | $65$ characters |
| Training prefix | $1003854$ characters |
| Validation suffix | $111540$ characters |

For the encoded sequence $d$ of length $N$, we set $n=\lfloor0.9N\rfloor$ and used consecutive slices $d[0:n]$ and $d[n:N]$. We split before sampling and retained sequence order, with no separate test set. For a split $a$ of length $M$, a dedicated training-window generator seeded once with $1337$ sampled starts uniformly with replacement from $0,\ldots,M-257$. Each start supplied

$$
X=a[s:s+256],\qquad Y=a[s+1:s+257].
$$

Checks confirmed that windows remain inside their own split, that $Y$ is shifted by one character, and that the final target is $a[s+256]$.

**Model and training.** The model is a single `Embedding(65,65)` table: each current character selects a row of next-character logits. It has no hidden state or position embedding. We reset the initialization seed to $1337$ immediately before construction and initialized its weights from $\mathcal{{N}}(0,0.02^2)$. The checked tensors were:

| Tensor | Shape | Data type |
| --- | --- | --- |
| Inputs | $(32,256)$ | `torch.long` |
| Targets | $(32,256)$ | `torch.long` |
| Raw logits | $(32,256,65)$ | `torch.float32` |

Cross-entropy reshaped logits to $(8192,65)$ and targets to $(8192,)$, averaging over all $8192$ predictions without applying softmax first. Training used a fresh Adam optimizer with learning rate $0.01$, zero weight decay, batch size $32$, and exactly $500$ updates. Each update sampled a new batch, cleared gradients, backpropagated mean loss, and stepped the optimizer. We retained the final checkpoint, without validation-based selection, together with optimizer and random-generator states and all sampled starts. Computation used `float32` on the same single-server-GPU environment reported in Question 2(b), rather than free Google Colab.

**Decoded input/target pair.** The first sampled pair starts at training index ${sample['start']}$, using zero-based indices. Each row below represents $32$ original characters, with an end-exclusive offset range. The notation `\s` denotes one space and `\n` one newline; row breaks and offset labels are display formatting. All other characters are literal.

Input:

```text
{input_block}
```

Target:

```text
{target_block}
```

The last input character is `{sample['input_text'][-1]}`; the final target is `{sample['final_target_character']}` at training index ${sample['final_target_split_index']}$, confirming that the last prediction is included.

**Full-validation results.** In evaluation mode without gradients, we used starts $0,256,512,\ldots$, scoring $435$ full chunks and one final chunk of length $179$. The summed negative log-likelihood was divided by $111539$, so every validation character after the first was scored exactly once. Perplexity is $\exp(\ell)$ for the mean cross-entropy $\ell$.

| Model | Trainable parameters | Full-validation CE (nats/character) | Perplexity |
| --- | --- | --- | --- |
| Bigram, update $500$ | ${results['parameters']}$ | ${results['validation']['cross_entropy']:.6f}$ | ${results['validation']['perplexity']:.6f}$ |

An independent `float64` calculation based on validation character-transition counts agreed with the chunked evaluation within $10^{{-6}}$ nats/character.

**Generated continuation.** A separate CPU sampling generator was seeded once with $1337$. At temperature $1$, we repeatedly sampled from the latest character's next-character distribution and appended the sampled ID, without updating weights. The prompt, shown separately, is `ROMEO:` followed by a newline:

```text
ROMEO:
```

The following continuation contains $128$ newly sampled characters; spelling and internal whitespace are preserved:

{generation_block}

**Learnable and unavailable dependencies.** The model can learn adjacent-character preferences: its trained probability of `u` after `q` was ${results['learned_transition_example']['probability']:.6f}$. It cannot condition on a speaker name earlier in the prefix: `ROMEO:\n` and `JULIET:\n` both end in the same newline character and therefore receive exactly the same next-character distribution. Earlier context beyond the current character is unavailable to the model.
"""
    path = ASSIGNMENT_DIR / "Assignment2_submission.md"
    original = path.read_text(encoding="utf-8")
    marker = "#### Bonus 3(a):"
    if marker in original:
        prefix, remainder = original.split(marker, 1)
        following = remainder.find("\n#### ")
        suffix = remainder[following:] if following >= 0 else ""
        candidates = [prefix, prefix.removesuffix("\n")]
        updated = prefix + section + suffix
    else:
        candidates = [original]
        updated = original + ("\n" if not original.endswith("\n\n") else "") + section
    if not any(hashlib.sha256(value.encode()).hexdigest() == text_checks["earlier_report_sha256"]
               for value in candidates):
        raise RuntimeError("Earlier answers changed; refusing to overwrite them.")
    path.write_text(updated, encoding="utf-8")
    print(f"Updated {path}")


if __name__ == "__main__":
    main()
