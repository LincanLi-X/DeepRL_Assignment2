"""Fixed paired test conditions for Question 2(c), with no image resampling."""

import torch
from torchvision.datasets import FashionMNIST

from data import FashionMNISTSplit
from prepare_data import DATA_ROOT, SEED, SPLIT_DIR


# Positive row offsets move down; positive column offsets move right.
CONDITIONS = (
    ("centered", 0, 0),
    ("left", 0, -3),
    ("right", 0, 3),
    ("up", -3, 0),
    ("down", 3, 0),
)
CONDITION_LABELS = {
    "centered": "Centered", "left": "Left 3 px", "right": "Right 3 px",
    "up": "Up 3 px", "down": "Down 3 px",
}


def place_images(images: torch.Tensor, delta_row: int, delta_column: int) -> torch.Tensor:
    """Copy complete raw images onto fresh zero canvases, scaling only by 255."""
    if images.ndim not in (2, 3) or images.shape[-2:] != (28, 28):
        raise ValueError("Expected (28,28) or (B,28,28) images.")
    if images.dtype != torch.uint8:
        raise TypeError("Expected raw uint8 images.")
    row, column = 4 + delta_row, 4 + delta_column
    if row < 0 or column < 0 or row + 28 > 36 or column + 28 > 36:
        raise ValueError("The complete image must fit on the canvas.")
    canvas = torch.zeros(
        (*images.shape[:-2], 1, 36, 36), dtype=torch.float32, device=images.device
    )
    canvas[..., 0, row:row + 28, column:column + 28] = images.to(torch.float32) / 255.0
    return canvas


def load_paired_test_split() -> FashionMNISTSplit:
    """Use the saved official test indices in their original saved order."""
    saved = torch.load(
        SPLIT_DIR / "fashion_mnist_indices.pt", map_location="cpu", weights_only=True
    )
    if saved["seed"] != SEED:
        raise ValueError("Unexpected split seed.")
    source = FashionMNIST(root=str(DATA_ROOT), train=False, download=False)
    indices = saved["test"]
    if indices.shape != (2000,) or indices.dtype != torch.long:
        raise ValueError("Unexpected test index format.")
    if indices.unique().numel() != 2000 or not bool(((indices >= 0) & (indices < len(source))).all()):
        raise ValueError("Invalid original test indices.")
    selected = FashionMNISTSplit(source, indices)
    if not torch.equal(selected.labels, torch.arange(10).repeat_interleave(200)):
        raise ValueError("Test class counts/order differ from the prescribed split.")
    return selected
