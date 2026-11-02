"""Shared Fashion-MNIST splits and the prescribed centered input transform."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torchvision.datasets import FashionMNIST

from prepare_data import DATA_ROOT, SEED, SPLIT_DIR, check_splits


def centered_canvases(images: torch.Tensor) -> torch.Tensor:
    """Map uint8 (28,28) or (B,28,28) images to float32 centered canvases.

    The only pixel transform is division by 255. The extra channel dimension
    gives (1,36,36) for one image or (B,1,36,36) for a batch.
    """
    if images.ndim not in (2, 3) or images.shape[-2:] != (28, 28):
        raise ValueError("Expected image shape (28,28) or (B,28,28).")
    if images.dtype != torch.uint8:
        raise TypeError("Expected raw uint8 pixels; do not scale the images twice.")
    canvases = torch.zeros(
        (*images.shape[:-2], 1, 36, 36), dtype=torch.float32, device=images.device
    )
    canvases[..., 0, 4:32, 4:32] = images.to(torch.float32) / 255.0
    return canvases


class FashionMNISTSplit(Dataset):
    """One saved split; retain original images for later placement experiments."""

    def __init__(self, source: FashionMNIST, indices: torch.Tensor) -> None:
        self.original_indices = indices.clone()
        self.images = source.data[self.original_indices]
        self.labels = source.targets[self.original_indices]
        self.classes = source.classes

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        return centered_canvases(self.images[position]), self.labels[position]


def load_shared_splits() -> dict[str, FashionMNISTSplit]:
    """Load the saved selection without resampling or changing its order."""
    saved = torch.load(
        SPLIT_DIR / "fashion_mnist_indices.pt", map_location="cpu", weights_only=True
    )
    if saved["seed"] != SEED:
        raise ValueError("Unexpected seed in the saved split definition.")
    official_train = FashionMNIST(root=str(DATA_ROOT), train=True, download=False)
    official_test = FashionMNIST(root=str(DATA_ROOT), train=False, download=False)
    indices = {name: saved[name] for name in ("train", "validation", "test")}
    check_splits((official_train, official_test), indices)
    return {
        name: FashionMNISTSplit(
            official_test if name == "test" else official_train, split_indices
        )
        for name, split_indices in indices.items()
    }
