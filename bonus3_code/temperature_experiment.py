"""Experiment 2: nine prescribed continuations and per-continuation R4 values."""

import csv
import json
import math

import torch

from gpt import parameter_fingerprint
from language_eval import generate
from part_c_common import assert_protected_unchanged, load_selected_model
from paths import BASE_DIR, RESULTS_DIR, sha256_file
from text_data import decode, encode, load_data


def repetition_rate(text):
    if len(text) != 128:
        raise ValueError("Score exactly 128 new characters, excluding the prompt.")
    fourgrams = [text[start:start + 4] for start in range(len(text) - 3)]
    unique = len(set(fourgrams))
    return 1 - unique / len(fourgrams), unique, len(fourgrams)


def main():
    verification = json.loads((RESULTS_DIR / "context_verification.json").read_text())
    if not verification["all_checks_passed"] or not (RESULTS_DIR / "figures" / "gpt_context_loss.png").is_file():
        raise RuntimeError("Complete and verify the context experiment/plot before temperature generation.")
    model, dtype, protocol = load_selected_model()
    before = parameter_fingerprint(model)
    _, vocabulary = load_data()
    prompt_ids = encode(protocol["prompt"], vocabulary)
    output_dir = RESULTS_DIR / "temperature_samples"
    output_dir.mkdir(exist_ok=True)
    records, saved_ids = [], {}
    for temperature in protocol["temperatures"]:
        saved_ids[str(temperature)] = {}
        for seed in protocol["generation_seeds"]:
            # A new independent generator is seeded once per continuation.
            generator = torch.Generator(device="cpu").manual_seed(seed)
            new_ids = generate(model, prompt_ids, protocol["new_characters"],
                               temperature=temperature, generator=generator,
                               max_context=256, autocast_dtype=dtype)
            continuation = decode(new_ids, vocabulary)
            rate, unique, windows = repetition_rate(continuation)
            filename = output_dir / f"temperature_{str(temperature).replace('.', 'p')}_seed{seed}.txt"
            filename.write_text(continuation, encoding="utf-8")
            records.append({"temperature": temperature, "seed": seed,
                            "new_characters": len(continuation), "unique_fourgrams": unique,
                            "fourgram_windows": windows, "R4": rate,
                            "continuation": continuation,
                            "file": str(filename.relative_to(BASE_DIR)),
                            "file_sha256": sha256_file(filename)})
            saved_ids[str(temperature)][str(seed)] = {
                "new_ids": new_ids, "generator_state_after_generation": generator.get_state()}
            print(f"Temperature {temperature:.1f}, seed {seed}: {len(continuation)} new characters; "
                  f"unique fourgrams {unique}/{windows}; R4={rate:.6f}", flush=True)
    summary = [{"temperature": temperature,
                "mean_R4": math.fsum(record["R4"] for record in records if record["temperature"] == temperature) / 3,
                "seeds": [1, 2, 3]}
               for temperature in protocol["temperatures"]]
    assert parameter_fingerprint(model) == before
    assert_protected_unchanged(protocol)
    torch.save(saved_ids, RESULTS_DIR / "temperature_generation_ids.pt")
    results = {"selected_update": protocol["selected_update"],
               "selected_checkpoint_sha256": protocol["selected_checkpoint_sha256"],
               "prompt": protocol["prompt"], "records": records, "summary": summary,
               "autocast_dtype": str(dtype), "sampling_generator_device": "cpu",
               "dropout_off": not model.training, "no_top_k_or_top_p": True,
               "model_and_frozen_artifacts_unchanged": True,
               "reference_tau1_full_validation_perplexity": protocol["full_validation_reference_perplexity"],
               "source_sha256": {name: sha256_file(BASE_DIR / name) for name in (
                   "temperature_experiment.py", "language_eval.py", "part_c_common.py")}}
    (RESULTS_DIR / "temperature_results.json").write_text(json.dumps(results, ensure_ascii=True, indent=2) + "\n")
    with (RESULTS_DIR / "temperature_samples.csv").open("w", newline="", encoding="utf-8") as output:
        fields = [key for key in records[0] if key != "continuation"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: record[key] for key in fields} for record in records)
    with (RESULTS_DIR / "temperature_summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["temperature", "mean_R4"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in ("temperature", "mean_R4")} for row in summary)
    print("Saved all nine raw continuations/IDs, nine R4 values, and three temperature means.")


if __name__ == "__main__":
    main()
