"""Run the ordered embedding/attention component checks before GPT assembly."""

import argparse
import json

import torch

from gpt import GPTConfig, TokenPositionEmbedding
from paths import RESULTS_DIR, SEED


def check_embeddings():
    torch.manual_seed(SEED)
    embedding = TokenPositionEmbedding(GPTConfig()).eval()
    inputs = torch.arange(34).reshape(2, 17) % 65
    with torch.inference_mode():
        default = embedding(inputs)
        explicit = embedding(inputs, torch.arange(17))
        cropped = embedding(torch.zeros(2, 32, dtype=torch.long), torch.arange(224, 256))
    assert default.shape == (2, 17, 384) and default.dtype == torch.float32
    assert torch.equal(default, explicit)
    assert cropped.shape == (2, 32, 384)
    return {"default_embedding_shape": list(default.shape),
            "explicit_position_shape": list(cropped.shape),
            "default_equals_explicit_zero_based_positions": True,
            "supports_retained_positions_224_through_255": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["embeddings", "attention"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.step == "embeddings":
        result = check_embeddings()
        filename = "gpt_embedding_checks.json"
    else:
        from gpt import CausalMultiHeadAttention
        if not (RESULTS_DIR / "gpt_embedding_checks.json").is_file():
            raise RuntimeError("Complete embedding checks before attention checks.")
        torch.manual_seed(SEED)
        attention = CausalMultiHeadAttention(GPTConfig()).eval()
        hidden = torch.randn(2, 17, 384)
        with torch.inference_mode():
            output, weights = attention(hidden, return_attention=True)
        row_error = (weights.sum(dim=-1) - 1).abs().max().item()
        future_weight = weights.triu(diagonal=1).abs().max().item()
        assert output.shape == (2, 17, 384) and weights.shape == (2, 6, 17, 17)
        assert row_error < 1e-5 and future_weight == 0
        result = {"output_shape": list(output.shape), "attention_shape": list(weights.shape),
                  "maximum_row_sum_error": row_error, "maximum_future_weight": future_weight}
        filename = "gpt_attention_component_checks.json"
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS {args.step}: {result}")


if __name__ == "__main__":
    main()
