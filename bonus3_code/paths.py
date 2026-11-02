"""Project-relative paths and fixed constants for Bonus 3."""

from pathlib import Path
import hashlib

BASE_DIR = Path(__file__).resolve().parent
ASSIGNMENT_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
TEXT_PATH = BASE_DIR / "shakespeare.txt"
SEED = 1337
CONTEXT_LENGTH = 256


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_directories() -> None:
    for directory in (DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
