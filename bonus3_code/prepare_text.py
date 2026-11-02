"""Step 1: preserve, encode, and verify the supplied Shakespeare text."""

import hashlib
import json
import shutil

import torch

from paths import ASSIGNMENT_DIR, DATA_DIR, RESULTS_DIR, TEXT_PATH, ensure_directories, sha256_file


def main() -> None:
    ensure_directories()
    source = ASSIGNMENT_DIR / "shakespeare-1.txt"
    source_bytes = source.read_bytes()
    if TEXT_PATH.exists():
        if TEXT_PATH.read_bytes() != source_bytes:
            raise ValueError("Existing shakespeare.txt differs from the supplied text.")
    else:
        shutil.copyfile(source, TEXT_PATH)
    # Decode bytes directly to avoid universal-newline conversion.
    text = TEXT_PATH.read_bytes().decode("utf-8")
    characters = sorted(set(text))
    if len(text) != 1115394 or len(characters) != 65:
        raise ValueError("Unexpected character count or vocabulary size.")
    stoi = {character: index for index, character in enumerate(characters)}
    tokens = torch.tensor([stoi[character] for character in text], dtype=torch.long)
    decoded = "".join(characters[index] for index in tokens.tolist())
    if decoded != text or decoded.encode("utf-8") != source_bytes:
        raise ValueError("Encoding/decoding did not reproduce the original bytes.")
    vocabulary = {"size": len(characters), "itos": characters, "stoi": stoi}
    (DATA_DIR / "vocabulary.json").write_text(
        json.dumps(vocabulary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    torch.save({"tokens": tokens, "source_sha256": sha256_file(source)},
               DATA_DIR / "encoded_text.pt")
    earlier_report = (ASSIGNMENT_DIR / "Assignment2_submission.md").read_text()
    if "#### Bonus 3(a):" in earlier_report:
        earlier_report = earlier_report.split("#### Bonus 3(a):", 1)[0]
        if earlier_report.endswith("\n\n"):
            earlier_report = earlier_report.removesuffix("\n")
    results = {
        "characters": len(text), "vocabulary_size": len(characters),
        "encoding_roundtrip_exact": True, "source_bytes_preserved": True,
        "source_sha256": sha256_file(source), "local_copy_sha256": sha256_file(TEXT_PATH),
        "encoded_tensor_shape": list(tokens.shape), "encoded_dtype": str(tokens.dtype),
        "whitespace_characters": [character for character in characters if character.isspace()],
        "earlier_report_sha256": hashlib.sha256(earlier_report.encode()).hexdigest(),
    }
    (RESULTS_DIR / "text_checks.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Characters: {len(text):,}; vocabulary: {len(characters)} sorted characters.")
    print("PASS: exact character/byte roundtrip; case, punctuation, and whitespace preserved.")
    print(f"Byte-identical working copy: {TEXT_PATH}")


if __name__ == "__main__":
    main()
