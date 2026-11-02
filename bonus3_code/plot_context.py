"""Plot the three losses on the identical 512 fixed target characters."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import RESULTS_DIR


def main():
    verification = json.loads((RESULTS_DIR / "context_verification.json").read_text())
    if not verification["all_checks_passed"]:
        raise RuntimeError("Verify the context experiment before plotting.")
    results = json.loads((RESULTS_DIR / "context_results.json").read_text())
    lengths = [row["context_length"] for row in results["rows"]]
    losses = [row["cross_entropy"] for row in results["rows"]]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axis = plt.subplots(figsize=(6.6, 4.0), layout="constrained")
    axis.plot(lengths, losses, color="#0072B2", marker="o", linewidth=2, markersize=6)
    for length, loss in zip(lengths, losses):
        axis.annotate(f"{loss:.6f}", (length, loss), xytext=(0, 10), textcoords="offset points", ha="center")
    margin = max((max(losses) - min(losses)) * 0.2, 0.04)
    axis.set(xlabel="Context length (characters)", ylabel="Cross-entropy (nats/target character)",
             title="GPT loss on the same 512 validation targets", xlim=(12, 276),
             ylim=(min(losses) - margin, max(losses) + margin))
    axis.set_xticks(lengths)
    axis.grid(axis="y", alpha=0.22)
    directory = RESULTS_DIR / "figures"
    directory.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        path = directory / f"gpt_context_loss.{extension}"
        fig.savefig(path, dpi=220, facecolor="white")
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
