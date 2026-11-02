"""Plot the completed 21-point sampled training/validation evaluation history."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import RESULTS_DIR


def main():
    results = json.loads((RESULTS_DIR / "gpt_training_results.json").read_text())
    audit = json.loads((RESULTS_DIR / "gpt_training_verification.json").read_text())
    if results["completed_iterations"] != 5000 or not audit["all_checks_passed"]:
        raise RuntimeError("Plot only the completed, audited run.")
    history = results["evaluation_history"]
    updates = [row["update"] for row in history]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axis = plt.subplots(figsize=(7.2, 4.5), layout="constrained")
    axis.plot(updates, [row["train_cross_entropy"] for row in history], color="#0072B2",
              marker="o", markersize=3.5, linewidth=2, label="Sampled training CE")
    axis.plot(updates, [row["validation_cross_entropy"] for row in history], color="#D55E00",
              marker="s", markersize=3.5, linewidth=2, label="Sampled validation CE")
    selected = results["selected_update"]
    minimum = results["selected_sampled_validation_ce"]
    axis.axvline(selected, color="#444444", linestyle=":", linewidth=1.3, alpha=0.8)
    axis.scatter([selected], [minimum], marker="*", s=150, color="#E6AB02",
                 edgecolors="#333333", linewidths=0.7, zorder=5,
                 label=f"Selected checkpoint: update {selected:,}")
    axis.set(xlabel="Completed training iterations", ylabel="Cross-entropy (nats/character)",
             title="GPT sampled training and validation loss", xlim=(-75, 5075))
    axis.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, loc="upper right", fontsize=9.5)
    directory = RESULTS_DIR / "figures"
    directory.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        path = directory / f"gpt_sampled_loss.{extension}"
        fig.savefig(path, dpi=220, facecolor="white")
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
