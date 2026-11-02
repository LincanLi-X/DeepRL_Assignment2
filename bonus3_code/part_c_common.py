"""Frozen-model inference setup and integrity checks shared by both experiments."""

import json
import os

import torch

from gpt import build_gpt
from paths import BASE_DIR, CHECKPOINT_DIR, RESULTS_DIR, sha256_file
from runtime import configure_float32_gpu


def read_protocol():
    return json.loads((RESULTS_DIR / "part_c_protocol.json").read_text())


def assert_protected_unchanged(protocol):
    for filename, expected in protocol["protected_artifact_sha256"].items():
        if sha256_file(BASE_DIR / filename) != expected:
            raise ValueError(f"Frozen part (a)/(b) artifact changed: {filename}")
    if sha256_file(RESULTS_DIR / "context_targets.pt") != protocol["context_target_file_sha256"]:
        raise ValueError("The fixed context targets changed.")


def load_selected_model():
    protocol = read_protocol()
    assert_protected_unchanged(protocol)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Use the documented single-GPU launch environment.")
    configure_float32_gpu()
    saved = torch.load(CHECKPOINT_DIR / "gpt_best.pt", map_location="cpu", weights_only=True)
    if saved["completed_iterations"] != protocol["selected_update"]:
        raise ValueError("Wrong checkpoint for the prescribed inference experiments.")
    model = build_gpt().to("cuda:0")
    model.load_state_dict(saved["model_state"])
    model.eval()
    dtype = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16}[protocol["autocast_dtype"]]
    return model, dtype, protocol
