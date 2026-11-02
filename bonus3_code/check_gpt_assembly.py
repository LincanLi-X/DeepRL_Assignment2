"""Step 3: validate the assembled architecture before prescribed initialization."""

import json

import torch
from torch import nn

from gpt import GPT
from paths import RESULTS_DIR


def main():
    if not (RESULTS_DIR / "gpt_attention_component_checks.json").is_file():
        raise RuntimeError("Complete the attention component checks first.")
    torch.set_num_threads(4)
    model = GPT().eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, dtype=torch.long))
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count == 10745088 and logits.shape == (1, 3, 65)
    assert model.output.weight is model.embeddings.token.weight
    assert all(layer.bias is None for layer in model.modules() if isinstance(layer, nn.Linear))
    norms = [layer for layer in model.modules() if isinstance(layer, nn.LayerNorm)]
    assert len(norms) == 13 and all(layer.bias is None and layer.eps == 1e-5 for layer in norms)
    assert len([layer for layer in model.modules() if isinstance(layer, nn.Dropout)]) == 19
    checks = {"parameters": count, "raw_logit_shape": list(logits.shape),
              "weight_tensor_shared": True, "all_linear_biases_absent": True,
              "layernorm_scale_only": True, "dropout_module_count": 19,
              "independent_block_count": len(model.blocks)}
    (RESULTS_DIR / "gpt_assembly_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(f"PASS: full architecture, true weight sharing, raw logits, and {count:,} unique parameters.")


if __name__ == "__main__":
    main()
