"""Download Fashion-MNIST and save the shared Question 2(a)(1) splits.

Run from any directory: python question2_code/prepare_data.py
All paths are relative to this script, so the dataset stays in question2_code/data.
This step preserves the original uint8 images; input scaling is a later step.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torchvision
from torchvision.datasets import FashionMNIST
from torchvision.datasets.utils import download_and_extract_archive


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"
SPLIT_DIR = BASE_DIR / "splits"
SEED = 1337
OFFICIAL_GITHUB_MIRROR = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
)


def load_official_datasets() -> tuple[FashionMNIST, FashionMNIST]:
    """Use torchvision's downloader, with the dataset author's mirror as fallback."""
    try:
        train = FashionMNIST(root=str(DATA_ROOT), train=True, download=True)
        test = FashionMNIST(root=str(DATA_ROOT), train=False, download=True)
    except RuntimeError as error:
        print(f"Default download could not complete: {error}")
        print("Trying the official zalandoresearch/fashion-mnist GitHub mirror.")
        # The directory and the four MD5 values are torchvision's own. Download
        # the identical official files without changing FashionMNIST globally.
        raw_dir = DATA_ROOT / "FashionMNIST" / "raw"
        for filename, md5 in FashionMNIST.resources:
            download_and_extract_archive(
                OFFICIAL_GITHUB_MIRROR + filename,
                download_root=str(raw_dir),
                filename=filename,
                md5=md5,
            )
        train = FashionMNIST(root=str(DATA_ROOT), train=True, download=True)
        test = FashionMNIST(root=str(DATA_ROOT), train=False, download=True)
    return train, test


def make_split_indices(
    train_targets: torch.Tensor, test_targets: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return original official indices, concatenated in label order 0 through 9."""
    train_generator = torch.Generator().manual_seed(SEED)
    test_generator = torch.Generator().manual_seed(SEED)
    selected: dict[str, list[torch.Tensor]] = {
        "train": [], "validation": [], "test": []
    }

    for class_id in range(10):
        # nonzero on these 1-D CPU tensors returns indices in ascending order.
        train_candidates = torch.nonzero(
            train_targets == class_id, as_tuple=True
        )[0]
        train_order = torch.randperm(
            train_candidates.numel(), generator=train_generator
        )
        shuffled_train = train_candidates[train_order]
        selected["train"].append(shuffled_train[:1000])
        selected["validation"].append(shuffled_train[1000:1200])

        test_candidates = torch.nonzero(test_targets == class_id, as_tuple=True)[0]
        test_order = torch.randperm(
            test_candidates.numel(), generator=test_generator
        )
        selected["test"].append(test_candidates[test_order[:200]])

    return {name: torch.cat(parts) for name, parts in selected.items()}


def check_splits(
    datasets: tuple[FashionMNIST, FashionMNIST],
    indices: dict[str, torch.Tensor],
) -> dict:
    """Check source data, split sizes, class order, uniqueness, and disjointness."""
    train, test = datasets
    for dataset, size, per_class in ((train, 60000, 6000), (test, 10000, 1000)):
        if dataset.data.shape != (size, 28, 28) or dataset.data.dtype != torch.uint8:
            raise ValueError("Unexpected official image shape or dtype.")
        if dataset.targets.shape != (size,) or dataset.targets.dtype != torch.long:
            raise ValueError("Unexpected official target shape or dtype.")
        if torch.bincount(dataset.targets, minlength=10).tolist() != [per_class] * 10:
            raise ValueError("Unexpected official class counts.")

    summary = {
        "seed": SEED,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "selection_device": "cpu",
        "class_names": train.classes,
        "official_sizes": {"train": len(train), "test": len(test)},
        "splits": {},
    }
    for name, expected_count in (("train", 1000), ("validation", 200), ("test", 200)):
        source = test if name == "test" else train
        split_indices = indices[name]
        if split_indices.dtype != torch.long or split_indices.ndim != 1:
            raise ValueError(f"Invalid index format for {name}.")
        if split_indices.numel() != 10 * expected_count:
            raise ValueError(f"Incorrect size for {name}.")
        if not bool(((split_indices >= 0) & (split_indices < len(source))).all()):
            raise ValueError(f"Out-of-range official index in {name}.")
        if split_indices.unique().numel() != split_indices.numel():
            raise ValueError(f"Duplicate original indices in {name}.")
        labels = source.targets[split_indices]
        expected_labels = torch.arange(10).repeat_interleave(expected_count)
        if not torch.equal(labels, expected_labels):
            raise ValueError(f"Incorrect per-class count or class order for {name}.")
        summary["splits"][name] = {
            "source": "official_test" if name == "test" else "official_train",
            "size": split_indices.numel(),
            "per_class_counts": torch.bincount(labels, minlength=10).tolist(),
            "unique_indices": True,
            "concatenated_in_class_order": True,
            "indices_sha256": hashlib.sha256(
                split_indices.numpy().astype("<i8").tobytes()
            ).hexdigest(),
        }

    overlap = len(set(indices["train"].tolist()) & set(indices["validation"].tolist()))
    if overlap != 0:
        raise ValueError("Training and validation splits overlap.")
    summary["train_validation_overlap_count"] = overlap
    summary["train_validation_disjoint"] = True
    summary["unused_official_examples"] = {"train": 48000, "test": 8000}
    return summary


def main() -> None:
    datasets = load_official_datasets()
    indices = make_split_indices(datasets[0].targets, datasets[1].targets)
    summary = check_splits(datasets, indices)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = SPLIT_DIR / "fashion_mnist_indices.pt"
    torch.save({"seed": SEED, **indices}, index_path)
    summary_path = SPLIT_DIR / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\nSplit        Source           Size    Per-class counts (labels 0-9)")
    for name, details in summary["splits"].items():
        print(
            f"{name:<12} {details['source']:<16} {details['size']:>5}   "
            f"{details['per_class_counts']}"
        )
    print(f"Training/validation overlap: {summary['train_validation_overlap_count']}")
    print(f"Shared indices: {index_path}")
    print(f"Check results:  {summary_path}")


if __name__ == "__main__":
    main()
