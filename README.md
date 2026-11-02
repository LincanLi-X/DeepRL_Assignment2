# Deep and Reinforcement Learning - Assignment 2

Code and recorded experiments for CAP 4613/5619 Homework 2. The repository covers two programming tasks:

- **Question 2:** compare an MLP and a CNN on Fashion-MNIST, including their sensitivity to changes in object position.
- **Bonus 3:** train a character-level bigram baseline and a GPT on Shakespeare, then study context length and sampling temperature.

The repository includes data, saved splits, experiment configurations, evaluation results, figures, and the smaller model checkpoints. The two GPT checkpoints are distributed separately below.

## Required download: GPT checkpoints

> [!IMPORTANT]
> **`gpt_best.pt` and `gpt_latest.pt` are not included in the GitHub repository.** Download them from Google Drive and place them directly in `bonus3_code/checkpoints/`, keeping their filenames unchanged. Git LFS is not required.

| File | Download | Size | Purpose |
| --- | --- | ---: | --- |
| `gpt_best.pt` | [Download from Google Drive](https://drive.google.com/file/d/1cMiE1Mhio4FSG6Z2smE8-tvfPUDXL1uu/view?usp=sharing) | 125.66 MiB | Checkpoint selected at update 1,750; used for GPT evaluation, generation, and the context/temperature experiments. |
| `gpt_latest.pt` | [Download from Google Drive](https://drive.google.com/file/d/1TI7N9OR_CjcRI7oMCpa6s-BoTJ8YEdGA/view?usp=sharing) | 129.98 MiB | State after all 5,000 updates; retained for training-state verification and resumption. |

The resulting layout must be:

```text
bonus3_code/
└── checkpoints/
    ├── bigram_final.pt
    ├── bigram_latest.pt
    ├── gpt_best.pt       # Download from Google Drive.
    └── gpt_latest.pt     # Download from Google Drive.
```

These are complete training checkpoints containing model weights, optimizer state, progress, and random-generator states. A standalone generation example below needs only `gpt_best.pt`; the saved context/temperature protocol also verifies the latest checkpoint, so download **both** before using that workflow.

Optional download verification, from the repository root:

```bash
sha256sum bonus3_code/checkpoints/gpt_best.pt bonus3_code/checkpoints/gpt_latest.pt
```

Expected SHA-256 values:

```text
80a21e79c6e03fbd8e82aac5e8e9e75476697f6d072f4535bd0e68a55601cb9e  bonus3_code/checkpoints/gpt_best.pt
dc1221c6a8c5df8e268591c7639a99d18650fce25fa00214213a1cdf88fcf4c9  bonus3_code/checkpoints/gpt_latest.pt
```

## Repository structure

```text
question2_code/
├── prepare_data.py              # Download Fashion-MNIST and save shared splits.
├── data.py                      # Scale pixels and construct centered canvases.
├── models.py                    # Prescribed MLP and CNN.
├── check_setup.py               # Input/model checks and pre-training prediction.
├── prepare_training.py          # Save the shared five-epoch training orders.
├── train.py / run_training.sh   # Training, validation, timing, and checkpoints.
├── spatial_data.py              # Five paired image placements.
├── prepare_position_test.py     # Geometry, pixel sums, and visual-check record.
├── evaluate_position.py         # Accuracy and prediction consistency.
├── verify_*.py                  # Independent checks of saved experiment evidence.
├── plot_*.py / write_*_report.py # Figures and assignment-report sections.
├── data/                        # Official Fashion-MNIST cache.
├── splits/                      # Original indices and epoch permutations.
├── checkpoints/                 # Best/latest MLP and CNN states.
└── results/                     # JSON/CSV results, predictions, logs, and figures.

bonus3_code/
├── shakespeare.txt              # Supplied text, preserving all original bytes.
├── prepare_text.py / split_text.py
├── text_data.py / losses.py     # Shared vocabulary, windows, and cross-entropy.
├── bigram.py / train_bigram.py  # Character bigram model and training.
├── gpt.py                       # Embeddings, causal attention, blocks, tied head.
├── gpt_optimization.py          # AdamW groups and learning-rate schedule.
├── gpt_evaluation.py            # Sampled training/validation evaluation.
├── train_gpt.py / run_gpt.sh    # Mixed-precision GPT training and resumption.
├── language_eval.py             # Full validation and autoregressive generation.
├── evaluate_bigram.py / evaluate_gpt.py
├── check_*.py / verify_*.py     # Architecture, causality, and result checks.
├── prepare_context.py          # Fixed validation targets and frozen protocol.
├── context_experiment.py        # Context lengths 32, 128, and 256.
├── temperature_experiment.py    # Three temperatures, three seeds, and R4.
├── plot_*.py / write_*_report.py
├── data/                        # Vocabulary, encoded text, and ordered splits.
├── checkpoints/                 # Bigram states and downloaded GPT checkpoints.
└── results/                     # Metrics, sampled starts, text, logs, and figures.
```

The directory-level READMEs provide further implementation details: [Question 2](question2_code/README.md) and [Bonus 3](bonus3_code/README.md). Question 1 and Bonuses 1-2 are written derivations rather than separate training programs.

## Environment and installation

The original experiments used Python 3.11, PyTorch `2.11.0+cu128`, torchvision `0.26.0+cu128`, NumPy `1.26.4`, and Matplotlib `3.10.0`, on one NVIDIA A100 80GB PCIe GPU. Question 2 and the bigram use float32; the recorded GPT run uses BF16 autocast with float32 parameters and zero scaler-skipped updates.

The launchers require Linux, Bash, and `flock` from `util-linux`. GPU experiment scripts require CUDA and expose exactly one GPU. CPU-only machines can inspect the saved results and run CPU checks that do not require GPU inference.

```bash
git clone https://github.com/LincanLi-X/DeepRL_Assignment2.git
cd DeepRL_Assignment2

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy matplotlib
```

Install matching CUDA-enabled PyTorch and torchvision builds using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). For a machine compatible with the CUDA 12.8 wheels, for example:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Check GPU availability and choose an available GPU UUID:

```bash
python -c 'import torch; print(torch.__version__); print("CUDA available:", torch.cuda.is_available())'
nvidia-smi --query-gpu=index,uuid,name,memory.free --format=csv

# Replace the placeholder with a UUID printed by nvidia-smi.
export RL_GPU="GPU-REPLACE_WITH_YOUR_GPU_UUID"
export CUDA_VISIBLE_DEVICES="$RL_GPU"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Run the commands below from the repository root in the same activated environment. GPU launchers select a device and take a project file lock; on a managed cluster, obtain the GPU through the local scheduler first.

## Quick start: generate text from the selected GPT

After downloading `gpt_best.pt`, run this example from the repository root. It uses the saved vocabulary and the existing handwritten generation function, without training or changing experiment artifacts.

```bash
python - <<'PY'
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path("bonus3_code").resolve()))
from gpt import build_gpt
from language_eval import generate
from text_data import decode, encode, load_data

assert torch.cuda.is_available(), "Install a CUDA-enabled PyTorch build and select a GPU."
checkpoint = torch.load(
    "bonus3_code/checkpoints/gpt_best.pt",
    map_location="cpu", weights_only=True,
)
model = build_gpt()
model.load_state_dict(checkpoint["model_state"])
model = model.to("cuda:0").eval()
_, vocabulary = load_data()

prompt = "ROMEO:\n"
generator = torch.Generator(device="cpu").manual_seed(1)
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
new_ids = generate(
    model, encode(prompt, vocabulary), new_characters=128,
    temperature=1.0, generator=generator, max_context=256,
    autocast_dtype=dtype,
)
print(prompt + decode(new_ids, vocabulary))
PY
```

Generation reads the latest at-most-256 characters, uses ordinary position IDs, disables dropout, and samples without top-k or top-p filtering. The example selects BF16 where supported and FP16 otherwise; exact sampled text can differ across hardware, precision, and library versions.

## Running the experiment workflows

**The repository contains a completed run.** Reading its results or using the downloaded model does not require retraining. The commands in the training sections describe the ordered workflow for a new run. Existing checkpoints/logs are protected against accidental overwriting; use a separate working copy with the previous run's checkpoint and result directories moved aside before starting a new experiment. Complete each stage and inspect its required checks before continuing.

Some preparation, verification, and report-writing scripts also read `Assignment2_submission.md` from the repository root. If that document is not included in your checkout, supply the corresponding assignment working report before using those scripts. For a fresh staged run, the report should contain the completed preceding sections when each stage captures its report hash. The standalone generation example does not require the report.

### Question 2: Fashion-MNIST and spatial generalization

Both models share 10,000 training, 2,000 validation, and 2,000 test images, with equal class counts. Pixels are scaled to `[0,1]`; each original `28x28` image is placed on a `36x36` zero canvas. Both models train for five epochs, batch size 128, using Adam with learning rate 0.001 and shared saved epoch permutations.

| Model | Architecture | Trainable parameters |
| --- | --- | ---: |
| MLP | Flatten -> Linear(1296,72) -> ReLU -> Linear(72,10) | 94,114 |
| CNN | Three convolution/ReLU stages, channels 32/64/128; max pooling after the first two; global average pooling and a classifier | 93,962 |

Prepare data and models, then train and evaluate:

```bash
python question2_code/prepare_data.py
python question2_code/check_setup.py
python question2_code/prepare_training.py
bash question2_code/run_training.sh "$RL_GPU"
python question2_code/verify_training.py
python question2_code/plot_training.py
```

The selected checkpoint has the lowest validation loss, with earlier epochs retained on ties. Test results are not used for selection.

For the paired position experiment:

```bash
python question2_code/prepare_position_test.py
```

Inspect `question2_code/results/figures/q2c_placements.png`, then record the visual check and continue:

```bash
python question2_code/prepare_position_test.py --confirm-visual-check
bash question2_code/run_position_test.sh "$RL_GPU"
python question2_code/verify_position_results.py
python question2_code/plot_position_cases.py
```

The five conditions are centered, left/right by 3 pixels, and up/down by 3 pixels. Complete images are copied onto fresh canvases without cropping, interpolation, or wrapping. Accuracy uses the true label; consistency compares a shifted prediction with the same model's centered prediction for the same example.

### Bonus 3(a): Shakespeare and the bigram baseline

The corpus contains 1,115,394 characters and a sorted vocabulary of 65 characters. The ordered 90/10 split contains 1,003,854 training and 111,540 validation characters. Windows never cross the split boundary.

`prepare_text.py` expects the original course file at `shakespeare-1.txt` in the repository root. If it is absent, the bundled byte-preserving copy can supply it:

```bash
test -f shakespeare-1.txt || cp bonus3_code/shakespeare.txt shakespeare-1.txt

python bonus3_code/prepare_text.py
python bonus3_code/split_text.py
python bonus3_code/check_bigram_setup.py
bash bonus3_code/run_bigram.sh "$RL_GPU"
bash bonus3_code/run_bigram_evaluation.sh "$RL_GPU"
python bonus3_code/verify_bigram.py
```

The baseline is `Embedding(65,65)`, with 4,225 parameters. It trains for 500 Adam updates, batch size 32, learning rate 0.01. Evaluation scores all 111,539 validation targets after the first character, and generation produces 128 new characters from `ROMEO:\n`.

### Bonus 3(b): implement and train GPT

The GPT has six pre-LayerNorm blocks, embedding width 384, six attention heads, maximum context 256, and dropout 0.2. Linear layers are bias-free; LayerNorm has a trainable scale and no bias. The token embedding and output projection share one parameter tensor. The model has **10,745,088 unique trainable parameters**.

Attention scores, the causal mask before softmax, head concatenation, Transformer blocks, and autoregressive generation are implemented explicitly using basic PyTorch operations. No ready-made Transformer, MultiheadAttention, scaled-dot-product-attention API, or pretrained model weights are used.

Run the ordered checks and training stages after the bigram workflow:

```bash
python bonus3_code/check_gpt_components.py embeddings
python bonus3_code/check_gpt_components.py attention
python bonus3_code/check_gpt_assembly.py
flock -n bonus3_code/.gpu.lock python bonus3_code/check_gpt.py
python bonus3_code/prepare_gpt_training.py
bash bonus3_code/run_gpt.sh "$RL_GPU"
python bonus3_code/verify_gpt_training.py
flock -n bonus3_code/.gpu.lock python bonus3_code/evaluate_gpt.py
python bonus3_code/plot_gpt_training.py
flock -n bonus3_code/.gpu.lock python bonus3_code/check_gpt_generation.py
```

Training uses batch size 64 and completes all 5,000 updates with AdamW, 100 warmup iterations, cosine decay, and gradient-norm clipping at 1.0. Parameters remain float32; forward/loss computation uses BF16 where supported, otherwise FP16 with GradScaler.

Sampled evaluation runs at update 0 and every 250 updates, drawing 200 training and 200 validation batches from a separate continuously advancing generator. The lowest sampled validation cross-entropy selects `gpt_best.pt`; `gpt_latest.pt` records the final training state. Full-validation CE and perplexity use the same 111,539 targets as the bigram and do not select the checkpoint.

### Bonus 3(c): context length and temperature

For the **bundled completed run**, download both GPT files and retain the saved `part_c_protocol.json`, targets, and preceding result files. Then run inference only, in this order:

```bash
flock -n bonus3_code/.gpu.lock python bonus3_code/context_experiment.py
python bonus3_code/verify_context.py
python bonus3_code/plot_context.py
flock -n bonus3_code/.gpu.lock python bonus3_code/temperature_experiment.py
```

- **Context experiment:** the same 512 validation targets, sampled with seed 2027, are scored with 32, 128, and 256 characters of context. Explicit retained position IDs end at 255 for every length. Each cropped sequence gets a fresh forward pass; only its last position is scored. Last-position logits are converted to float32 for CE, with a float64 mean.
- **Temperature experiment:** temperatures 0.7, 1.0, and 1.3 are each combined with seeds 1, 2, and 3. All nine continuations contain 128 new characters after `ROMEO:\n`. Four-character repetition is `R4 = 1 - U4/125`, counting overlapping raw substrings, including whitespace and punctuation, and excluding the prompt.

`verify_temperature.py` independently checks all nine samples and the three means; it also requires the original report prefix described below. For a newly completed training run, establish the new protocol with `prepare_context.py` only after part (b)'s report and verification have been completed.

The saved protocol protects preceding metrics as well as checkpoint bytes. Re-running earlier full-validation scripts rewrites result files and can invalidate that protocol even when the model is unchanged. Keep an untouched copy of the bundled artifacts when reproducing part (c).

### Resuming interrupted training

Use `--resume` with the corresponding launcher:

```bash
bash question2_code/run_training.sh "$RL_GPU" --resume
bash bonus3_code/run_bigram.sh "$RL_GPU" --resume
bash bonus3_code/run_gpt.sh "$RL_GPU" --resume
```

These commands restore the saved run rather than start an additional training budget. Exact continuation requires the matching data, source, configuration, and supported original runtime; GPT resumption also checks the recorded GPU selection and software environment. A completed 5,000-update run has no remaining GPT updates to perform.

## Results and saved outputs

The recorded run selected epoch 5 for both image classifiers:

| Model | Validation CE (nats/image) | Validation accuracy | Training/evaluation time (s) |
| --- | ---: | ---: | ---: |
| MLP | 0.496752 | 82.70% | 1.079 |
| CNN | 0.784496 | 72.05% | 1.757 |

The MLP fit centered images better within the fixed budget, while the CNN had higher accuracy and consistency at every tested nonzero shift. Mean consistency over the four shifted conditions was 47.3750% for the MLP and 85.7375% for the CNN.

| Language model | Parameters | Full-validation CE (nats/character) | Perplexity |
| --- | ---: | ---: | ---: |
| Bigram | 4,225 | 2.521165 | 12.443087 |
| Selected GPT, update 1,750 | 10,745,088 | 1.468866 | 4.344307 |

The selected GPT's **sampled** validation CE was 1.467989. This is a different metric from the full-validation diagnostic in the table.

| Context length | CE on 512 fixed targets |
| --- | ---: |
| 32 | 1.699378 |
| 128 | 1.566800 |
| 256 | 1.457959 |

| Sampling temperature | Mean R4 across three seeds |
| --- | ---: |
| 0.7 | 0.064000 |
| 1.0 | 0.088000 |
| 1.3 | 0.010667 |

These are observations from one training run per architecture. Lower repetition does not by itself establish better grammar or coherence, and generation temperature does not change the model's full-validation perplexity measured at temperature 1.

Key output files:

| Location | Contents |
| --- | --- |
| `question2_code/results/training_summary.csv` | Selected epoch, validation metrics, and elapsed time. |
| `question2_code/results/position_metrics.csv` | Ten model/position accuracy and consistency rows. |
| `question2_code/results/figures/` | Learning curves, placement check, and changed-prediction pairs. |
| `bonus3_code/results/language_model_comparison.csv` | Bigram/GPT full-validation comparison. |
| `bonus3_code/results/gpt_sampled_evaluations.csv` | All 21 sampled training/validation evaluations. |
| `bonus3_code/results/context_losses.csv` | Three context-length losses. |
| `bonus3_code/results/temperature_summary.csv` | Mean repetition rate for each temperature. |
| `bonus3_code/results/temperature_samples/` | All nine unedited continuations. |
| `bonus3_code/results/figures/` | GPT sampled-loss and context-length curves. |

## Reproducibility and report utilities

Data selection, model initialization, and training sampling use independent streams seeded with 1337. GPT sampled evaluation uses seed 2026, context targets use 2027, and the temperature experiment uses sampling seeds 1/2/3. Checkpoints retain the states required for resumption. Matching seeds do not guarantee identical results across hardware or PyTorch versions.

The `verify_*.py` scripts audit saved indices, generator streams, checkpoint/source hashes, metric calculations, and, in several cases, the original report prefix. They are checks of the recorded experiment: a missing report, later editorial changes to it, or regenerated protected files can make an audit stop even when model loading succeeds. Use the matching original report snapshot for those checks; standalone generation does not depend on its hashes.

The `write_*_report.py` scripts insert verified results into the root `Assignment2_submission.md`:

- Question 2: `write_training_report.py`, `write_position_report.py`.
- Bonus 3: `write_bigram_report.py`, `write_gpt_report.py`, `write_part_c_report.py`.

Run them only when the preceding stage and its required verification are complete. They modify the working assignment report; they are not required to load a model or inspect the saved results.
