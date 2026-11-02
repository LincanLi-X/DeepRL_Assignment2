"""Steps 6-7: one complete 5,000-iteration GPT run and 21 prescribed evaluations."""

import argparse
import csv
import json
import math
import os
import random
import time

import torch

from gpt import build_gpt, parameter_fingerprint
from gpt_evaluation import sampled_evaluation
from gpt_optimization import build_optimizer, learning_rate
from losses import next_character_loss
from paths import BASE_DIR, CHECKPOINT_DIR, RESULTS_DIR, SEED, sha256_file
from runtime import atomic_save, capture_rng, configure_float32_gpu, cpu_snapshot, restore_rng
from text_data import load_data, sample_batch


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    plan = json.loads((RESULTS_DIR / "gpt_training_plan.json").read_text())
    setup = json.loads((RESULTS_DIR / "gpt_setup_checks.json").read_text())
    if not plan["all_checks_passed"] or not setup["all_checks_passed"]:
        raise RuntimeError("Pass all pre-training and optimizer checks first.")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Use the provided GPU launcher.")
    configure_float32_gpu()
    for filename, expected in plan["input_sha256"].items():
        if sha256_file(BASE_DIR / filename) != expected:
            raise ValueError("Shared training data changed.")
    if sha256_file(BASE_DIR / "gpt.py") != plan["gpt_source_sha256"]:
        raise ValueError("GPT implementation changed after checks.")
    latest_path, best_path = CHECKPOINT_DIR / "gpt_latest.pt", CHECKPOINT_DIR / "gpt_best.pt"
    if latest_path.exists() and not args.resume:
        raise FileExistsError("A GPT run already exists; use --resume for continuation.")
    data, _ = load_data()
    random.seed(SEED)
    model = build_gpt()
    # Created at initialization and advanced continuously over all evaluations.
    evaluation_generator = torch.Generator().manual_seed(2026)
    training_generator = torch.Generator().manual_seed(SEED)
    if parameter_fingerprint(model) != setup["initial_parameter_sha256"]:
        raise ValueError("Training initialization differs from the checked initialization.")
    model = model.to("cuda:0")
    optimizer, group_metadata = build_optimizer(model)
    for key, value in plan["optimizer_group_checks"].items():
        if key != "fused" and group_metadata[key] != value:
            raise ValueError("Optimizer grouping differs from the checked plan.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("Keep model parameters float32.")
    source_hashes = {filename: sha256_file(BASE_DIR / filename) for filename in (
        "train_gpt.py", "gpt.py", "gpt_optimization.py", "gpt_evaluation.py",
        "text_data.py", "losses.py", "runtime.py", "paths.py")}
    environment = {"gpu_name": torch.cuda.get_device_name(0),
                   "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                   "visible_device_count": torch.cuda.device_count(),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "torch_version": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                   "parameter_dtype": "float32", "autocast_dtype": str(dtype),
                   "grad_scaler_used": scaler is not None, "tf32": False,
                   "fused_adamw": group_metadata["fused"], "deterministic_algorithms": True,
                   "hardware_deviation": "one server GPU instead of prescribed free Google Colab GPU"}
    completed = skips = 0
    best_loss, best_update, elapsed_before = float("inf"), None, 0.0
    history, evaluations, training_starts, evaluation_starts = [], [], [], []
    if args.resume and latest_path.exists():
        saved = torch.load(latest_path, map_location="cpu", weights_only=True)
        if saved["plan"] != plan or saved["source_sha256"] != source_hashes:
            raise ValueError("Source or configuration changed; cannot exactly resume.")
        for key in ("gpu_name", "cuda_visible_devices", "torch_version", "cuda_runtime", "autocast_dtype"):
            if saved["environment"][key] != environment[key]:
                raise ValueError("Use the original hardware/software environment for continuation.")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        if scaler is not None:
            scaler.load_state_dict(saved["scaler_state"])
        restore_rng(saved["random_states"], training_generator)
        evaluation_generator.set_state(saved["random_states"]["evaluation_window_generator"])
        completed, skips = saved["completed_iterations"], saved["scaler_skipped_steps"]
        best_loss, best_update = saved["best_validation_cross_entropy"], saved["best_update"]
        elapsed_before = saved["elapsed_seconds"]
        history, evaluations = saved["training_history"], saved["evaluation_history"]
        training_starts = list(saved["training_window_starts"].unbind(0))
        evaluation_starts = list(saved["evaluation_window_starts"].unbind(0))
        del saved

    log = (RESULTS_DIR / "gpt_training.log").open("a" if args.resume else "x", encoding="utf-8")
    def emit(message):
        print(message, flush=True)
        log.write(message + "\n")
        log.flush()

    torch.cuda.synchronize()
    started = time.perf_counter()
    def elapsed():
        return elapsed_before + time.perf_counter() - started

    def save_checkpoint(improved=False):
        states = capture_rng(training_generator)
        states["evaluation_window_generator"] = evaluation_generator.get_state()
        payload = {
            "model_state": cpu_snapshot(model.state_dict()),
            "optimizer_state": cpu_snapshot(optimizer.state_dict()),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "completed_iterations": completed, "optimizer_steps": completed - skips,
            "scaler_skipped_steps": skips, "best_update": best_update,
            "best_validation_cross_entropy": best_loss,
            "random_states": states, "plan": plan, "environment": environment,
            "source_sha256": source_hashes, "elapsed_seconds": elapsed(),
            "training_history": list(history), "evaluation_history": list(evaluations),
            "training_window_starts": torch.stack(training_starts) if training_starts else torch.empty(0, 64, dtype=torch.long),
            "evaluation_window_starts": torch.stack(evaluation_starts) if evaluation_starts else torch.empty(0, 2, 200, 64, dtype=torch.long),
        }
        if improved:
            atomic_save(payload, best_path)
        atomic_save(payload, latest_path)
        write_csv(RESULTS_DIR / "gpt_training_history.csv", history)
        write_csv(RESULTS_DIR / "gpt_sampled_evaluations.csv", evaluations)

    def evaluate_and_select():
        nonlocal best_loss, best_update
        evaluation_started = time.perf_counter()
        metrics, starts = sampled_evaluation(model, data, evaluation_generator, training_generator, dtype)
        evaluation_starts.append(starts)
        metrics = {"update": completed, **metrics, "elapsed_seconds": elapsed()}
        evaluations.append(metrics)
        improved = metrics["validation_cross_entropy"] < best_loss
        if improved:
            best_loss, best_update = metrics["validation_cross_entropy"], completed
        save_checkpoint(improved)
        emit(f"Evaluation {completed}/5000: train CE={metrics['train_cross_entropy']:.6f}; "
             f"val CE={metrics['validation_cross_entropy']:.6f}; "
             f"best={best_update} ({best_loss:.6f}); evaluation {time.perf_counter() - evaluation_started:.1f}s; "
             f"elapsed {elapsed():.1f}s")

    emit(f"GPU: {environment['gpu_name']}; autocast {dtype}; unique parameters 10,745,088; "
         f"resume update {completed}.")
    if not evaluations:
        save_checkpoint()
        evaluate_and_select()  # Include update zero in selection.
    model.train()
    for iteration in range(completed, 5000):
        rate = learning_rate(iteration)
        for group in optimizer.param_groups:
            group["lr"] = rate
        inputs, targets, starts = sample_batch(data["train"], training_generator, 64)
        inputs, targets = inputs.to("cuda:0"), targets.to("cuda:0")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            logits = model(inputs)
            loss = next_character_loss(logits, targets)
        value = loss.item()
        if not math.isfinite(value):
            raise RuntimeError("Non-finite GPT training loss.")
        skipped = False
        if scaler is not None:
            previous_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < previous_scale
        else:
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        skips += int(skipped)
        completed = iteration + 1
        training_starts.append(starts)
        history.append({"update": completed, "learning_rate": rate, "batch_cross_entropy": value,
                        "unclipped_gradient_norm": norm.item(), "scaler_skipped_step": skipped})
        if completed % 100 == 0:
            emit(f"Update {completed}/5000: batch CE={value:.6f}; lr={rate:.9f}; "
                 f"skips={skips}; elapsed={elapsed():.1f}s")
        if completed % 250 == 0:
            evaluate_and_select()

    if completed != 5000 or len(evaluations) != 21:
        raise RuntimeError("Incomplete training budget or evaluation schedule.")
    torch.save({"seed": SEED, "starts": torch.stack(training_starts),
                "generator_state_after_training": training_generator.get_state()},
               RESULTS_DIR / "gpt_training_starts.pt")
    torch.save({"seed": 2026, "updates": torch.tensor([entry["update"] for entry in evaluations]),
                "starts": torch.stack(evaluation_starts),
                "generator_state_after_evaluations": evaluation_generator.get_state()},
               RESULTS_DIR / "gpt_evaluation_starts.pt")
    summary = {"plan": plan, "environment": environment, "completed_iterations": completed,
               "optimizer_steps": completed - skips, "scaler_skipped_steps": skips,
               "selected_update": best_update, "selected_sampled_validation_ce": best_loss,
               "evaluation_history": evaluations, "elapsed_seconds": elapsed(),
               "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
               "source_sha256": source_hashes,
               "best_checkpoint_sha256": sha256_file(best_path),
               "latest_checkpoint_sha256": sha256_file(latest_path)}
    (RESULTS_DIR / "gpt_training_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    emit(f"Completed all 5,000 iterations and 21 evaluations; selected update {best_update}, "
         f"sampled validation CE {best_loss:.6f}; skipped steps {skips}.")
    log.close()


if __name__ == "__main__":
    main()
