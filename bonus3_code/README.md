# Bonus 3: character-level language modeling

## Completed part (a): data and bigram baseline

All code and artifacts for this part are inside `bonus3_code/`. The only
nonstandard dependency is PyTorch. The run used one NVIDIA A100 80GB PCIe on
the server, PyTorch 2.11.0+cu128, float32, no autocast, and no TF32. The report
records the server-GPU environment in place of the prescribed free Colab GPU.
No hyperparameter search or additional model training was performed.

The scripts resolve paths relative to their own location. Run from the assignment
directory, completing each step before proceeding:

```bash
# 1. Byte-preserving UTF-8 read, sorted vocabulary, and exact roundtrip.
python bonus3_code/prepare_text.py
# 2. Ordered 90/10 split before window sampling.
python bonus3_code/split_text.py
# 3. Input/target alignment, boundaries, loss, and model shape checks.
python bonus3_code/check_bigram_setup.py
# 4. Exactly 500 training updates on one selected GPU.
bash bonus3_code/run_bigram.sh GPU-8b270cf8-6bb4-cee0-7060-88eba83d2fb0
# 5. Prescribed full validation and 128-character continuation.
bash bonus3_code/run_bigram_evaluation.sh GPU-8b270cf8-6bb4-cee0-7060-88eba83d2fb0
# 6. Independent result audit and report with model-capability explanations.
python bonus3_code/verify_bigram.py
python bonus3_code/write_bigram_report.py
```

Use an available GPU UUID on another server. Each launcher exposes exactly one
GPU and holds a project lock against duplicate simultaneous runs. This server
had no discovered batch scheduler, so access was direct. The processes exit and
release GPU memory when finished. Training refuses to overwrite an existing
run. To continue after an interruption at the latest saved update:

```bash
bash bonus3_code/run_bigram.sh GPU-8b270cf8-6bb4-cee0-7060-88eba83d2fb0 --resume
```

For the already completed run, read the saved results or rerun the CPU verifier;
there is no need to retrain.

## Data and sampling

`shakespeare.txt` is a byte-identical working copy of the supplied
`../shakespeare-1.txt`. UTF-8 is decoded directly from bytes to avoid newline
conversion. The text has 1,115,394 characters and 65 distinct characters.
One sorted vocabulary is shared across splits. The first 1,003,854 characters
form training, and the remaining 111,540 validation; there is no independent
test set.

`text_data.py` samples windows with replacement from one split at a time.
For length M and T=256, `torch.randint(M-T, ...)` draws starts from 0 through
M-T-1. Inputs use `[s:s+T]`, targets `[s+1:s+T+1]`, including the last target.
The training generator is a dedicated CPU `torch.Generator` seeded once with
1337. Setup previews use a separate fresh generator, and training resets its
own sampler to 1337. Only current batches are moved to the GPU.

## Model, training, and evaluation

`bigram.py` defines one trainable `Embedding(65,65)` table and returns raw
`(B,T,65)` logits. `build_bigram` resets the model seed immediately before
construction and initializes weights from a normal distribution with mean 0
and standard deviation 0.02. The model has 4,225 parameters and no position
embedding or hidden state.

`losses.py` reshapes logits and targets and uses built-in cross-entropy on raw
logits. `train_bigram.py` uses Adam, learning rate 0.01, zero weight decay,
batch size 32, and exactly 500 updates. It logs every batch loss and records all
16,000 sampled starts. Model/optimizer states, completed updates, Python and
PyTorch CPU/CUDA RNG states, and the dedicated sampler state are saved initially,
every 100 updates, and at the final update. NumPy randomness is not used. State
tensors are copied, and checkpoint replacement is atomic. The final checkpoint
is retained without choosing an update from validation performance.

`language_eval.py` implements the full-validation diagnostic and autoregressive
generation using basic tensor operations. Evaluation uses 435 length-256 chunks
and one length-179 chunk, with contexts reset within each chunk. Every validation
target except the first is scored once: 111,539 targets. Per-token negative
log-likelihoods are summed in float64 and divided by target count; perplexity is
the exponential of that mean, not an average of per-batch perplexities.

Generation uses prompt `ROMEO:\n`, temperature 1, and a separate CPU generator
seeded once with 1337. Probabilities are moved to the generator's device for
`torch.multinomial`, and sampled IDs back to the model's device. For the bigram,
only the current character is forwarded. The saved continuation contains exactly
128 new characters, excluding the prompt, without any text cleanup.

`verify_bigram.py` checks optimizer steps, replays all sampled starts, verifies
chunk coverage and generated IDs, and independently recomputes validation NLL
from adjacent-character counts and the saved table in CPU float64. This audit
does not train or reselect the model.

## Saved artifacts

- `data/vocabulary.json`, `encoded_text.pt`, `text_splits.pt`: shared vocabulary,
  encoded corpus, and ordered CPU int64 splits.
- `results/text_checks.json`, `split_checks.json`, `bigram_setup_checks.json`:
  counts, byte/encoding checks, boundaries, shapes, target alignment, and
  initialization checks.
- `results/sample_pair.json`: the full 256-character input/target pair from
  training start 354057, including final target `g` at index 354313.
- `checkpoints/bigram_final.pt`, `bigram_latest.pt`: final/latest resumable states.
- `results/bigram_training.json`, `bigram_training_history.csv`,
  `bigram_training_starts.pt`: configuration, GPU/software/source hashes, all 500
  training losses, and all sampled starts.
- `results/full_validation_windows.pt`: starts and lengths for the diagnostic.
- `results/bigram_results.json`: parameter count, validation CE/perplexity,
  generation settings/text, and learned transition example.
- `results/bigram_continuation.txt`, `bigram_generation.pt`: exact continuation,
  generated IDs, and the sampling generator's final state.
- `results/bigram_verification.json`: independent verification results.

Measured full-validation cross-entropy was 2.5211652412 nats/character and
perplexity 12.4430874173. The independent transition-count CE differed by about
9.02e-9. `write_bigram_report.py` inserts the complete part (a) answer into the
assignment Markdown, preserving all earlier answers. Whitespace in the decoded
pair is explicitly escaped; generated text is shown separately from the prompt.

## Part (b): prescribed GPT

The GPT uses six independent pre-LayerNorm blocks, width 384, six heads of
width 64, context length 256, and dropout 0.2. `gpt.py` implements combined QKV
projection, head splitting, scaled dot products, a lower-triangular mask before
softmax, attention dropout, concatenation, and output projection directly.
There are no ready-made Transformer/attention modules or SDPA calls. The FFN
uses `384 -> 1536 -> 384` with GELU. Linear layers have no biases; LayerNorm has
scale only and epsilon 1e-5. Dropout appears only at embedding sums, attention
weights, and the two residual outputs per block (19 modules total).

The output layer references the exact token-embedding parameter tensor. After
tying, each unique matrix is initialized from N(0,0.02^2), LayerNorm scales to 1,
and each attention output / second FFN projection is reinitialized with standard
deviation 0.02/sqrt(12). `build_gpt` resets seed 1337 immediately before model
construction. The model has 10,745,088 unique trainable parameters.

The ordered setup/training commands are:

```bash
python bonus3_code/check_gpt_components.py embeddings
python bonus3_code/check_gpt_components.py attention
python bonus3_code/check_gpt_assembly.py
CUDA_VISIBLE_DEVICES=GPU-8b270cf8-6bb4-cee0-7060-88eba83d2fb0 CUBLAS_WORKSPACE_CONFIG=:4096:8 flock -n bonus3_code/.gpu.lock python bonus3_code/check_gpt.py
python bonus3_code/prepare_gpt_training.py
bash bonus3_code/run_gpt.sh GPU-8b270cf8-6bb4-cee0-7060-88eba83d2fb0
python bonus3_code/verify_gpt_training.py
CUDA_VISIBLE_DEVICES=GPU-8b270cf8-6bb4-cee0-7060-88eba83d2fb0 CUBLAS_WORKSPACE_CONFIG=:4096:8 flock -n bonus3_code/.gpu.lock python bonus3_code/evaluate_gpt.py
python bonus3_code/plot_gpt_training.py
```

Use `--resume` with the training launcher to continue an interrupted run at its
latest saved evaluation. Training rejects an existing run without this flag.
Exact continuation checks the original source, data, hardware, and precision.

`check_gpt.py` runs the prescribed checks in float32 without autocast, with
dropout off. It checks short `(2,17,65)` and full `(2,256,65)` logits, attention
row sums in all six blocks/heads, zero future attention, and prefix-logit
equality when the two inputs differ only after position 127. Explicit retained
positions 224-255 are also supported. Model checks complete before training.

`gpt_optimization.py` creates unique-parameter AdamW groups. All 26 matrix
tensors, including both embeddings, use weight decay 0.1; the 13 LayerNorm
scales use zero decay. Betas are (0.9,0.99), epsilon 1e-8. The exact schedule
uses `1e-3*(i+1)/101` for i<100 and
`1e-4 + 9e-4/2*(1+cos(pi*(i-100)/4900))` afterward, for i=0,...,4999.

`train_gpt.py` keeps parameters float32, uses BF16 autocast when supported
(otherwise FP16 with GradScaler), and clips unscaled global gradient norm to 1.
It completes all 5,000 iterations, counting any scaler-skipped step in the
budget. Only the current batch is transferred to the GPU. No hyperparameter
search, compilation, distributed training, or gradient accumulation is used.

Training starts use an independent CPU generator seeded with 1337. A separate
generator, seeded with 2026 at initialization, continuously supplies fresh
evaluation windows: at update 0 and each multiple of 250, draw 200 training
batches followed by 200 validation batches, each of shape (64,256). The target
count is 3,276,800 per split per evaluation. Evaluation disables dropout, does
not advance the training sampler, and restores training mode. The smallest
sampled validation CE selects the checkpoint, including update 0 and retaining
the earlier checkpoint on ties. Latest and best states include both samplers,
global CPU/CUDA RNG states, optimizer/scaler states, completed iterations,
histories, and sampled starts. Snapshots are copied and saved atomically.

The independent CPU verifier replays every training/evaluation random draw,
checks every scheduled learning rate and optimizer step, and validates selection
and state loading. The selected checkpoint's full-validation diagnostic uses
`language_eval.full_validation` with the same 111,539 targets, chunk starts,
lengths, and context resets as the bigram. This diagnostic does not reselect the
checkpoint and is separate from the sampled checkpoint-selection metric.

GPT artifacts:

- `results/gpt_*checks.json`: ordered component/architecture/numerical checks.
- `results/gpt_training_plan.json`: prescribed configuration, optimizer grouping,
  exact schedule boundaries, and protected input hashes.
- `checkpoints/gpt_best.pt`, `gpt_latest.pt`: selected and latest resumable states.
- `results/gpt_training.log`, `gpt_training_history.csv`: online batch CE,
  learning rates, unclipped gradient norms, skipped steps, and progress.
- `results/gpt_sampled_evaluations.csv`, `gpt_training_results.json`: all 21
  sampled evaluations, selection, software/hardware, precision, and source hashes.
- `results/gpt_training_starts.pt`, `gpt_evaluation_starts.pt`: all 320,000
  training windows and 537,600 evaluation windows, plus final generator states.
- `results/gpt_training_verification.json`: independent completed-run audit.
- `results/gpt_results.json`, `language_model_comparison.csv`: selected-model
  full-validation CE/perplexity and comparison with the saved bigram baseline.
- `results/gpt_full_validation_windows.pt`: full-validation coverage record.
- `results/figures/gpt_sampled_loss.{png,pdf}`: training/validation curves with
  the selected checkpoint marked.

Complete step 8 and write/check the report after the diagnostic:

```bash
CUDA_VISIBLE_DEVICES=GPU-8b270cf8-6bb4-cee0-7060-88eba83d2fb0 CUBLAS_WORKSPACE_CONFIG=:4096:8 flock -n bonus3_code/.gpu.lock python bonus3_code/check_gpt_generation.py
python bonus3_code/write_gpt_report.py
python bonus3_code/verify_gpt_results.py
```

Generation reuses the handwritten `language_eval.generate` implementation from
part (a), now with maximum context 256 and the training autocast precision.
The selected GPT accepts optional explicit position IDs for part (c); ordinary
generation uses positions 0 through T-1 on each cropped prefix. Checks use hooks
to confirm that every model input is exactly the latest at-most-256 IDs of the
actual growing prefix, including a prompt longer than 256, and that positions
reset correctly. A dedicated sampling generator is seeded once per continuation;
no targets, optimizer steps, top-k, or top-p filters are used. The checks leave
both parameters and checkpoint files unchanged.

`results/gpt_generation_checks.json` and `gpt_generation_check_ids.pt` retain
these interface checks. They are short verification samples, not the prescribed
part (c) temperature experiment. Plotting additionally requires Matplotlib.
`write_gpt_report.py` appends part (b), its numerical-check table, loss figure,
sampled checkpoint metrics, full-validation comparison, and interpretation.

The completed run used BF16 autocast with float32 parameters, fused AdamW, and
zero scaler-skipped steps. All 5,000 iterations and 21 evaluations completed in
about 1,002 seconds (including sampled evaluation and checkpoint I/O). The
selected checkpoint was update 1,750, with sampled validation CE 1.4679888155.
Its full-validation CE was 1.4688662585 and perplexity 4.3443070270, on the same
111,539 targets as the bigram (CE 2.5211652412, perplexity 12.4430874173).
Latest-update sampled train/validation CE were 0.6225373888/1.7033205516, so the
selected model is the earlier checkpoint, not the final update. The original
bigram checkpoint, data, and earlier report sections remain unchanged.
