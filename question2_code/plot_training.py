"""Plot the completed Question 2(b) run; this script performs no model updates."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

from prepare_data import BASE_DIR


def main() -> None:
    results_dir = BASE_DIR / "results"
    results = json.loads((results_dir / "training_results.json").read_text())
    audit = json.loads((results_dir / "training_verification.json").read_text())
    if not audit["all_checks_passed"] or len(results["models"]) != 2:
        raise RuntimeError("Verify both completed training runs before plotting.")
    figure_dir = results_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
        "axes.labelsize": 11, "legend.fontsize": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    colors = {"MLP": "#0072B2", "CNN": "#D55E00"}
    for metric, title, ylabel in (
        ("loss", "Training and validation cross-entropy", "Cross-entropy (nats/image)"),
        ("accuracy", "Training and validation accuracy", "Accuracy (%)"),
    ):
        fig, ax = plt.subplots(figsize=(7.0, 4.3), layout="constrained")
        for name, entry in results["models"].items():
            history = entry["history"]
            epochs = [row["epoch"] for row in history]
            for split, linestyle, marker in (("train", "-", "o"), ("validation", "--", "s")):
                values = [row[f"{split}_{metric}"] for row in history]
                ax.plot(epochs, values, color=colors[name], linestyle=linestyle,
                        marker=marker, markersize=5, linewidth=2,
                        markerfacecolor=colors[name] if split == "train" else "white",
                        label=f"{name} {'training' if split == 'train' else 'validation'}")
        ax.set(title=title, xlabel="Completed epoch", ylabel=ylabel, xlim=(0.85, 5.15))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(axis="y", alpha=0.22, linewidth=0.7)
        if metric == "accuracy":
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            ax.legend(loc="lower right", ncols=2, frameon=False)
        else:
            ax.set_ylim(bottom=0)
            ax.legend(loc="upper right", ncols=2, frameon=False)
        for extension in ("png", "pdf"):
            output = figure_dir / f"q2b_{metric}.{extension}"
            fig.savefig(output, dpi=220, facecolor="white")
            print(f"Saved {output}")
        plt.close(fig)

    fields = ["model", "selected_epoch", "validation_loss", "validation_accuracy", "elapsed_seconds"]
    with (results_dir / "training_summary.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for name, entry in results["models"].items():
            writer.writerow({"model": name, "selected_epoch": entry["selected_epoch"],
                             "validation_loss": entry["selected_validation_loss"],
                             "validation_accuracy": entry["selected_validation_accuracy"],
                             "elapsed_seconds": entry["elapsed_seconds"]})


if __name__ == "__main__":
    main()
