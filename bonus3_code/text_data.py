"""Shared character data and independent-generator window sampling."""

import json

import torch

from paths import CONTEXT_LENGTH, DATA_DIR, RESULTS_DIR, TEXT_PATH, sha256_file


def load_data():
    checks = json.loads((RESULTS_DIR / "split_checks.json").read_text())
    if sha256_file(DATA_DIR / "text_splits.pt") != checks["split_file_sha256"]:
        raise ValueError("Saved splits changed after validation.")
    if sha256_file(DATA_DIR / "vocabulary.json") != checks["vocabulary_sha256"]:
        raise ValueError("Saved vocabulary changed after validation.")
    data = torch.load(DATA_DIR / "text_splits.pt", map_location="cpu", weights_only=True)
    if sha256_file(TEXT_PATH) != data["source_sha256"]:
        raise ValueError("Source text changed after encoding.")
    vocabulary = json.loads((DATA_DIR / "vocabulary.json").read_text())
    return data, vocabulary


def encode(text, vocabulary):
    return torch.tensor([vocabulary["stoi"][character] for character in text], dtype=torch.long)


def decode(ids, vocabulary):
    return "".join(vocabulary["itos"][index] for index in ids.tolist())


def sample_batch(sequence, generator, batch_size, context_length=CONTEXT_LENGTH):
    """Sample with replacement within one CPU split, including its final target."""
    if sequence.ndim != 1 or sequence.dtype != torch.long or sequence.device.type != "cpu":
        raise ValueError("Use a one-dimensional CPU int64 sequence.")
    upper_bound = sequence.numel() - context_length
    if upper_bound <= 0 or context_length < 1:
        raise ValueError("The split must contain at least context_length + 1 characters.")
    starts = torch.randint(upper_bound, (batch_size,), generator=generator, device="cpu")
    positions = starts[:, None] + torch.arange(context_length)
    return sequence[positions], sequence[positions + 1], starts


def visible_whitespace(text):
    """ASCII representation: each space is \\s, each newline \\n, and each tab \\t."""
    escapes = {"\\": "\\\\", " ": "\\s", "\n": "\\n", "\t": "\\t", "\r": "\\r"}
    return "".join(escapes.get(character, character) for character in text)
