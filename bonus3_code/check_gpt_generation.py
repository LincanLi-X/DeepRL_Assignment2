"""Step 8: verify autoregressive sampling and last-256 cropping on the selected GPT."""

import json

import torch

from gpt import build_gpt, parameter_fingerprint
from language_eval import generate
from paths import CHECKPOINT_DIR, RESULTS_DIR, SEED, sha256_file
from runtime import configure_float32_gpu
from text_data import decode, encode, load_data


def main():
    results = json.loads((RESULTS_DIR / "gpt_results.json").read_text())
    if not results["same_targets_and_chunks_as_bigram"]:
        raise RuntimeError("Complete the selected GPT full-validation diagnostic first.")
    configure_float32_gpu()
    path = CHECKPOINT_DIR / "gpt_best.pt"
    before_file_hash = sha256_file(path)
    assert before_file_hash == results["selected_checkpoint_sha256"]
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = build_gpt().to("cuda:0")
    model.load_state_dict(saved["model_state"])
    model.eval()
    before_parameters = parameter_fingerprint(model)
    dtype = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16}[results["autocast_dtype"]]
    _, vocabulary = load_data()
    token_windows, position_windows = [], []
    def token_hook(module, arguments):
        token_windows.append(arguments[0].detach().cpu().clone())
    def position_hook(module, arguments):
        position_windows.append(arguments[0].detach().cpu().clone())
    handles = [model.embeddings.token.register_forward_pre_hook(token_hook),
               model.embeddings.position.register_forward_pre_hook(position_hook)]
    records = {}
    generated_ids = {}
    try:
        for name, prompt, count in (("short_prompt", "ROMEO:\n", 16),
                                    ("long_prompt", "ROMEO:\n" * 40, 2)):
            token_windows.clear()
            position_windows.clear()
            prompt_ids = encode(prompt, vocabulary)
            generator = torch.Generator(device="cpu").manual_seed(SEED)
            new_ids = generate(model, prompt_ids, count, temperature=1.0,
                               generator=generator, max_context=256, autocast_dtype=dtype)
            assert new_ids.shape == (count,) and len(token_windows) == count
            for step, (window, positions) in enumerate(zip(token_windows, position_windows)):
                full_prefix = torch.cat((prompt_ids, new_ids[:step]))
                expected = full_prefix[-256:]
                assert torch.equal(window[0], expected)
                assert torch.equal(positions, torch.arange(expected.numel()))
            records[name] = {"prompt_characters": len(prompt), "new_characters": count,
                             "forward_lengths": [window.shape[1] for window in token_windows],
                             "ordinary_positions_reset_each_forward": True,
                             "each_input_is_last_256_of_actual_prefix": True,
                             "seed": SEED, "temperature": 1.0,
                             "continuation": decode(new_ids, vocabulary)}
            generated_ids[name] = new_ids
    finally:
        for handle in handles:
            handle.remove()
    assert records["long_prompt"]["forward_lengths"] == [256, 256]
    assert parameter_fingerprint(model) == before_parameters
    assert sha256_file(path) == before_file_hash
    checks = {"all_checks_passed": True, "records": records,
              "autocast_dtype": str(dtype), "dropout_off": not model.training,
              "selected_checkpoint_sha256": before_file_hash,
              "parameters_and_checkpoint_unchanged": True,
              "dedicated_generator_device": "cpu", "no_top_k_or_top_p": True}
    (RESULTS_DIR / "gpt_generation_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    torch.save(generated_ids, RESULTS_DIR / "gpt_generation_check_ids.pt")
    print("PASS: short-prompt sampling, last-256 truncation, position resets, and actual prefix appends.")
    print("PASS: BF16/FP16 evaluation policy reused, dropout off, and selected weights unchanged.")


if __name__ == "__main__":
    main()
