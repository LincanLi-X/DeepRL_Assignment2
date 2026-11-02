"""Step 2: split the encoded sequence in order, before any window sampling."""

import json

import torch

from paths import DATA_DIR, RESULTS_DIR, TEXT_PATH, sha256_file


def main() -> None:
    checks = json.loads((RESULTS_DIR / "text_checks.json").read_text())
    vocabulary = json.loads((DATA_DIR / "vocabulary.json").read_text())
    encoded = torch.load(DATA_DIR / "encoded_text.pt", weights_only=True)
    if sha256_file(TEXT_PATH) != encoded["source_sha256"] or not checks["encoding_roundtrip_exact"]:
        raise ValueError("Complete the text encoding check before splitting.")
    tokens = encoded["tokens"]
    boundary = 9 * tokens.numel() // 10
    train, validation = tokens[:boundary].clone(), tokens[boundary:].clone()
    if train.numel() != 1003854 or validation.numel() != 111540:
        raise ValueError("Unexpected ordered split sizes.")
    if not torch.equal(torch.cat((train, validation)), tokens):
        raise ValueError("Splitting changed the sequence order.")
    torch.save({"train": train, "validation": validation, "boundary": boundary,
                "vocabulary_size": vocabulary["size"], "source_sha256": encoded["source_sha256"]},
               DATA_DIR / "text_splits.pt")
    summary = {
        "train_characters": train.numel(), "validation_characters": validation.numel(),
        "train_global_slice": [0, boundary],
        "validation_global_slice": [boundary, tokens.numel()],
        "slice_right_end_exclusive": True, "sequence_order_preserved": True,
        "one_shared_vocabulary": True, "separate_test_set": False,
        "split_file_sha256": sha256_file(DATA_DIR / "text_splits.pt"),
        "vocabulary_sha256": sha256_file(DATA_DIR / "vocabulary.json"),
    }
    (RESULTS_DIR / "split_checks.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Training: {train.numel():,}; validation: {validation.numel():,} characters.")
    print("PASS: consecutive disjoint slices reconstruct the original sequence exactly.")


if __name__ == "__main__":
    main()
