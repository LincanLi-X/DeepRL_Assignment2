"""Step 5: full validation and the prescribed bigram continuation, without training."""

import json

import torch

from bigram import build_bigram
from language_eval import full_validation, generate
from paths import BASE_DIR, CHECKPOINT_DIR, RESULTS_DIR, SEED, sha256_file
from runtime import configure_float32_gpu
from text_data import decode, encode, load_data


def main() -> None:
    configure_float32_gpu()
    training = json.loads((RESULTS_DIR / "bigram_training.json").read_text())
    path = CHECKPOINT_DIR / "bigram_final.pt"
    before_hash = sha256_file(path)
    if before_hash != training["final_checkpoint_sha256"]:
        raise ValueError("The final checkpoint changed after training.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["completed_updates"] != 500:
        raise ValueError("Complete all 500 training updates before evaluation.")
    data, vocabulary = load_data()
    model = build_bigram(vocabulary["size"]).to("cuda:0")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    validation, starts, lengths = full_validation(model, data["validation"], batch_size=32)
    if validation["targets"] != 111539:
        raise ValueError("Expected exactly 111,539 validation targets.")
    torch.save({"starts": starts, "lengths": lengths}, RESULTS_DIR / "full_validation_windows.pt")

    prompt = "ROMEO:\n"
    # This fresh CPU generator is separate from the training-window generator.
    sampling_generator = torch.Generator(device="cpu").manual_seed(SEED)
    continuation_ids = generate(model, encode(prompt, vocabulary), new_characters=128,
                                temperature=1.0, generator=sampling_generator, max_context=1)
    continuation = decode(continuation_ids, vocabulary)
    if len(continuation) != 128:
        raise ValueError("Expected 128 newly generated characters, excluding the prompt.")
    (RESULTS_DIR / "bigram_continuation.txt").write_text(continuation, encoding="utf-8")
    torch.save({"new_ids": continuation_ids, "generator_state_after_generation": sampling_generator.get_state()},
               RESULTS_DIR / "bigram_generation.pt")
    if sha256_file(path) != before_hash or not torch.equal(model.table.weight.detach().cpu(), checkpoint["model_state"]["table.weight"]):
        raise RuntimeError("Evaluation/generation changed the model.")
    q_id, u_id = vocabulary["stoi"]["q"], vocabulary["stoi"]["u"]
    q_distribution = model.table.weight[q_id].detach().cpu().softmax(dim=-1)
    results = {
        "parameters": training["parameters"], "completed_updates": 500,
        "validation": validation,
        "generation": {"prompt": prompt, "temperature": 1.0, "seed": SEED,
                       "generator_device": "cpu", "new_characters": len(continuation),
                       "continuation": continuation},
        "learned_transition_example": {
            "current_character": "q", "next_character": "u",
            "probability": q_distribution[u_id].item(),
            "most_likely_next_character": vocabulary["itos"][q_distribution.argmax().item()],
        },
        "checkpoint_unchanged": True, "final_checkpoint_sha256": before_hash,
        "environment": training["environment"],
        "evaluation_source_sha256": {filename: sha256_file(BASE_DIR / filename)
                                     for filename in ("evaluate_bigram.py", "language_eval.py")},
    }
    (RESULTS_DIR / "bigram_results.json").write_text(json.dumps(results, ensure_ascii=True, indent=2) + "\n")
    print(f"Full validation: {validation['targets']:,} targets, "
          f"CE={validation['cross_entropy']:.8f} nats/character, "
          f"perplexity={validation['perplexity']:.8f}.")
    print(f"Chunks: {validation['full_chunks']} full chunks and one final length-{validation['last_chunk_length']} chunk.")
    print(f"Generated 128 characters, seed 1337, temperature 1; p(u|q)={q_distribution[u_id].item():.6f}.")
    print("Continuation (Python repr, whitespace preserved):", repr(continuation))
    print("PASS: final checkpoint and model parameters unchanged.")


if __name__ == "__main__":
    main()
