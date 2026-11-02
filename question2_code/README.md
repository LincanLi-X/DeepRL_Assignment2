# Question 2 code

## Step 1: download and split Fashion-MNIST

From the assignment directory, run:

```bash
python question2_code/prepare_data.py
```

Dependencies: PyTorch, torchvision, and NumPy. The script works from any working
directory and saves everything under `question2_code/`:

- `data/FashionMNIST/raw/`: the four official compressed IDX files and their
  extracted image/label files. The data root is `data` within this code directory.
- `splits/fashion_mnist_indices.pt`: seed and original official indices for
  `train`, `validation`, and `test`, stored as CPU `torch.long` tensors.
- `splits/split_summary.json`: actual counts, disjointness checks, index hashes,
  source datasets, and library versions.

The script first uses `torchvision.datasets.FashionMNIST(download=True)`. If its
default download fails, it downloads identical files from the dataset author's
official `zalandoresearch/fashion-mnist` GitHub repository, checking torchvision's
MD5 checksums. Both source splits are loaded using `FashionMNIST`.

Selection uses two independent CPU `torch.Generator` instances seeded with 1337.
For each label from 0 to 9, ascending official training indices are shuffled once:
the first 1,000 are training examples and the next 200 are validation examples.
The separate test generator selects 200 official test indices per label. Neither
generator is reset within the class loop. Results are concatenated in class order.
Unselected examples are unused.

Both models must load this **same saved split file** rather than generate their
own splits. For subsequent code in this directory:

```python
from prepare_data import DATA_ROOT, SPLIT_DIR
import torch
from torchvision.datasets import FashionMNIST

indices = torch.load(SPLIT_DIR / "fashion_mnist_indices.pt", weights_only=True)
official_train = FashionMNIST(root=str(DATA_ROOT), train=True, download=False)
official_test = FashionMNIST(root=str(DATA_ROOT), train=False, download=False)

# Train and validation indices refer to official_train; test indices refer to
# official_test. Equal numerical indices across these two sources are unrelated.
train_images = official_train.data[indices["train"]]
validation_images = official_train.data[indices["validation"]]
test_images = official_test.data[indices["test"]]
```

The step-1 script only prepares the data selection. Raw images remain uint8, with no
transforms or augmentation. Scaling and centered canvases belong to the next
step. This preparation script uses CPU and performs no model training.

## Steps 2-4: centered inputs, model checks, and pre-training prediction

Run after step 1:

```bash
python question2_code/check_setup.py
```

- `data.py` loads the saved shared splits and provides a PyTorch Dataset for
  each. It retains original uint8 images, labels, and official indices. The
  transform divides raw pixels by 255 exactly once and places the image at
  `canvas[0, 4:32, 4:32]`, with a four-pixel zero border and no augmentation.
- `models.py` defines the prescribed MLP and CNN. Call `build_models()` to
  reset the initialization seed to 1337 separately before constructing each
  model. It uses default initializations and returns CPU float32 models.
- `check_setup.py` checks every selected image's scaling, placement, range,
  label format, and zero border. It checks DataLoader batches from all splits,
  dummy and real-image model outputs, actual parameter counts, and repeatable
  initialization. These are CPU float32 checks, with no training.
- `results/setup_checks.json` records measured input/model checks, layer
  parameter counts, architectures, and the PyTorch version.
- `results/pretraining_prediction.md` holds the original three-sentence
  prediction. The checker creates this file before any model checks and
  preserves its contents on later runs. Keep it unchanged after training.

The measured trainable parameter counts are 94,114 for MLP and 93,962 for CNN.
Both return raw `(B, 10)` logits, without softmax.

The shared data interface is:

```python
from torch.utils.data import DataLoader
from data import load_shared_splits
from models import build_models

splits = load_shared_splits()
images, labels = next(iter(DataLoader(splits["train"], batch_size=4)))
models = build_models()
# images: (4, 1, 36, 36), float32; labels: (4,), torch.long
# Later training must use the prescribed saved epoch permutations, rather than
# enabling independent random shuffling for the two models.
```

`Assignment2_submission.md` contains the full English Question 2(a) answer,
including the measured checks and the original prediction.

## Question 2(b): training and validation

The completed run used one NVIDIA A100 80GB PCIe on this server, with PyTorch
2.11.0+cu128 and float32. The report explicitly records use of a server GPU in
place of the assignment's free Google Colab GPU. No hyperparameter search was
performed. The test split is not loaded by the training or verification scripts.

The steps, in order, are:

```bash
python question2_code/prepare_training.py
bash question2_code/run_training.sh GPU-8b270cf8-6bb4-cee0-7060-88eba83d2fb0
CUDA_VISIBLE_DEVICES=GPU-8b270cf8-6bb4-cee0-7060-88eba83d2fb0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python question2_code/verify_training.py
python question2_code/plot_training.py
python question2_code/write_training_report.py
```

Use the UUID of an available GPU when moving the project to another machine.
The launcher exposes exactly one GPU and holds a project file lock to prevent
duplicate concurrent runs; the server had no discovered batch scheduler.
After the completed run, the GPU process exits. Launching training again without
`--resume` refuses to overwrite checkpoints. To continue an interrupted run at
its last completed epoch:

```bash
bash question2_code/run_training.sh GPU-8b270cf8-6bb4-cee0-7060-88eba83d2fb0 --resume
```

`prepare_training.py` creates `splits/epoch_permutations.pt`: five consecutive
permutations from a separate CPU generator seeded once with 1337. Both models
load these same orders. Each epoch visits all 10,000 selected training examples
in 79 batches, including a final batch of 16; each model receives 395 updates.

`train.py` constructs each model with initialization seed 1337 and a fresh Adam
optimizer (learning rate 0.001, default moments and epsilon, zero weight decay).
It uses mean cross-entropy on raw logits, no autocast, and no TF32. Deterministic
PyTorch/cuDNN operations and the launcher's cuBLAS workspace setting support
continuation on the same software/hardware. Centered training and validation
tensors and shared epoch orders are placed on the GPU before timing.

After every epoch, complete training and validation metrics are computed in
evaluation mode without gradients. Batch mean losses are weighted by sample
count. Strictly lower validation loss selects the best checkpoint, so ties
retain the earlier epoch. Timers synchronize CUDA and span training start to
the final epoch-end evaluation, including intermediate checkpoint I/O. Data
preparation, model/optimizer construction, plotting, and final checkpoint writes
are outside the interval.

Saved artifacts:

- `results/training_plan.json`: fixed configuration and input/order hashes.
- `results/training_environment.json`: GPU UUID, software, precision, timing
  definition, and hashes of the training source files.
- `results/training.log`, `mlp_history.csv`, and `cnn_history.csv`: all five
  epoch-end training/validation metrics and cumulative elapsed seconds.
- `results/training_results.json` and `training_summary.csv`: full results and
  selected-checkpoint table. Accuracy values in these files are fractions.
- `checkpoints/{mlp,cnn}_{best,latest}.pt`: model and optimizer states, completed
  epochs/updates, best validation result, history, elapsed time, Python/NumPy/
  PyTorch CPU/CUDA RNG states, and saved epoch orders. State tensors are
  snapshotted and files are replaced atomically. No gradient scaler is used.
- `results/training_verification.json`: checks of update counts, shared orders,
  optimizer/RNG states, selection, and reloaded best-checkpoint metrics.
- `results/figures/q2b_loss.{png,pdf}` and `q2b_accuracy.{png,pdf}`: two plots,
  each showing training and validation curves for both models.

`write_training_report.py` inserts the verified Question 2(b) section and image
links into the assignment Markdown. It preserves the earlier answers and the
original pre-training prediction. Plotting and report generation do not train
or change any model.

## Question 2(c): paired position sensitivity

This part reuses the frozen best-validation checkpoints from part (b), without
training or checkpoint selection on test data. The completed inference run used
the same single server A100 GPU and float32 settings as part (b).

Complete the steps in order:

```bash
python question2_code/prepare_position_test.py
# Inspect question2_code/results/figures/q2c_placements.png before confirming:
python question2_code/prepare_position_test.py --confirm-visual-check
bash question2_code/run_position_test.sh GPU-8b270cf8-6bb4-cee0-7060-88eba83d2fb0
python question2_code/verify_position_results.py
python question2_code/plot_position_cases.py
# Inspect the prediction-change figure before completing the report:
python question2_code/write_position_report.py
```

`spatial_data.py` reads the same 2,000 saved official test indices, preserving
their order and labels. `place_images` constructs a fresh zero-filled float32
canvas for each condition and copies the entire original image after dividing
by 255. Conditions are centered `(0,0)`, left `(0,-3)`, right `(0,3)`, up
`(-3,0)`, and down `(3,0)`, with positive rows moving down. There is no image
resizing, interpolation, clipping, or wrapping.

`prepare_position_test.py` checks all image contents, zero backgrounds, and
pixel sums under all five conditions, confirms that the centered transform
matches training, and renders one example. Pixel sums accumulate float32 pixel
values in float64. The `--confirm-visual-check` flag records the inspection only
after the saved figure has been viewed. It does not redo inference.

`evaluate_position.py` requires the completed visual check and exactly one
visible CUDA GPU. It reloads each selected checkpoint, evaluates in batches of
128 in evaluation mode without gradients, and saves all raw logits/predictions.
The last test batch contains 80 examples. Accuracy compares against original
labels; consistency compares against that model's own centered predictions for
the same saved test positions. The launcher shares the project file lock with
training, and the GPU process exits when inference finishes.

`verify_position_results.py` independently recomputes the ten rows with Python
counts, checks saved indices, labels and argmax predictions, and confirms that
all prior checkpoints and training artifacts are unchanged. It also checks the
example pairs once they exist, and can be rerun after the report is appended.

`plot_position_cases.py` shows one changed-prediction pair per model, scanning
directions left/right/up/down and selecting the first change in saved test
order. Selection does not filter for correct predictions. Two pairs were found:
MLP index 4060 changes T-shirt/top to Shirt; CNN index 6094 changes Shirt to
T-shirt/top. Both true labels are T-shirt/top.

Saved artifacts:

- `results/position_placements.json` and `position_checked_example.pt`: geometry,
  pixel-sum results, visual-check record, and checked example canvases.
- `results/position_predictions.pt`: original test indices, labels, class names,
  and CPU logits/predictions for both models at all five positions.
- `results/position_results.json` and `position_metrics.csv`: ten-row accuracy/
  consistency results, counts, GPU metadata, and protected artifact hashes.
  Metric values in these files are fractions.
- `results/position_verification.json`: independent checks and equally weighted
  means over the four nonzero shifts.
- `results/position_change_cases.json`: exact sample IDs, labels, predictions,
  selection rule, and figure checksum.
- `results/figures/q2c_placements.{png,pdf}`: the five-placement visual check.
- `results/figures/q2c_prediction_changes.{png,pdf}`: the two prediction-change
  pairs, labeled with model, shift, true class, and before/after predictions.

`write_position_report.py` appends Question 2(c), its complete ten-row table,
both figures, and the interpretation to `Assignment2_submission.md`. Earlier
answers and the original pre-training prediction are preserved. All measured
pixel-sum differences were zero. Mean shifted consistency was 47.3750% for MLP
and 85.7375% for CNN; the complete directional results are in the CSV/report.
