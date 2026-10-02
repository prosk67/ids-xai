from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch


BLUE = "#2f5d8a"
ORANGE = "#d08045"
GREEN = "#4f7f66"
INK = "#1f2933"


def _context_label(row) -> str:
    dataset = "CIC" if row.dataset == "CIC-IDS2017" else "UNSW"
    task = "Bin" if row.task == "binary" else "Multi"
    model = "RF" if row.model == "random_forest" else "XGB"
    return f"{dataset} / {task} / {model}"


def workflow_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 4.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        ("1  DATA", "CIC-IDS2017 and UNSW-NB15\nlabel audit, deduplication, fixed splits"),
        ("2  PREPROCESS", "Training-only medians and category maps\nexplicit unseen-category handling"),
        ("3  CLASSIFY", "Random Forest and XGBoost\nbinary and multiclass tasks; 5 seeds"),
        ("4  EXPLAIN", "Matched held-out flows\nTreeSHAP and LIME attribution vectors"),
        ("5  EVALUATE", "Fidelity controls, neighborhood stability,\nself-stability, latency, paired inference"),
        ("6  DECIDE", "Model-explainer deployment guidance\nand reproducible evidence package"),
    ]
    box_h = 0.115
    centers = np.linspace(0.90, 0.10, len(steps))
    colors = [BLUE, BLUE, GREEN, GREEN, ORANGE, ORANGE]
    for i, ((heading, detail), y, color) in enumerate(zip(steps, centers, colors)):
        box = FancyBboxPatch(
            (0.06, y - box_h / 2),
            0.88,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0.9,
            edgecolor=color,
            facecolor="#f8fafc",
        )
        ax.add_patch(box)
        ax.text(0.10, y + 0.025, heading, fontsize=8.1, fontweight="bold", color=color, va="center")
        ax.text(0.10, y - 0.020, detail, fontsize=6.6, color=INK, va="center")
        if i < len(steps) - 1:
            next_y = centers[i + 1]
            ax.annotate(
                "",
                xy=(0.5, next_y + box_h / 2 + 0.008),
                xytext=(0.5, y - box_h / 2 - 0.008),
                arrowprops=dict(arrowstyle="-|>", color="#667788", lw=0.9),
            )
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def paired_fidelity_figure(paired: pd.DataFrame, path: Path) -> None:
    data = paired.copy()
    data["label"] = data.apply(_context_label, axis=1)
    data = data.sort_values("mean_difference_shap_minus_lime")
    y = np.arange(len(data))
    means = data["mean_difference_shap_minus_lime"].to_numpy()
    lower = means - data["ci_low"].to_numpy()
    upper = data["ci_high"].to_numpy() - means
    colors = [BLUE if "CIC" in label else ORANGE for label in data["label"]]
    fig, ax = plt.subplots(figsize=(3.4, 3.3))
    ax.errorbar(
        means,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="#536273",
        elinewidth=1.1,
        capsize=2.2,
        zorder=1,
    )
    ax.scatter(means, y, c=colors, s=28, edgecolor="white", linewidth=0.5, zorder=2)
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_yticks(y, data["label"], fontsize=6.8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("Mean fidelity difference: TreeSHAP - LIME", fontsize=7.4)
    ax.set_title("Paired top-k deletion fidelity (95% bootstrap CI)", fontsize=8.2)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def latency_configuration_figure(xai: pd.DataFrame, path: Path) -> None:
    data = xai.copy()
    data["label"] = data.apply(_context_label, axis=1)
    order = (
        data.groupby("label")["latency_median_ms"].max().sort_values().index.tolist()
    )
    data["label"] = pd.Categorical(data["label"], categories=order, ordered=True)
    data = data.sort_values("label")
    fig, ax = plt.subplots(figsize=(3.4, 3.35))
    sns.scatterplot(
        data=data,
        x="latency_median_ms",
        y="label",
        hue="explainer",
        hue_order=["TreeSHAP", "LIME"],
        palette=[BLUE, ORANGE],
        style="explainer",
        s=46,
        ax=ax,
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Median latency per explanation (ms, log scale)", fontsize=7.2)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=6.8)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(title="", fontsize=6.5, loc="lower right")
    ax.set_title("Latency depends on both model and explainer", fontsize=8.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def class_recall_figure(per_class: pd.DataFrame, path: Path) -> None:
    data = per_class[
        per_class.dataset.eq("UNSW-NB15")
        & per_class.task.eq("multiclass")
        & per_class.seed.eq(42)
    ].copy()
    pivot = data.pivot(index="class", columns="model", values="recall")
    pivot = pivot.rename(columns={"random_forest": "RF", "xgboost": "XGBoost"})
    pivot = pivot.sort_values(["RF", "XGBoost"])
    fig, ax = plt.subplots(figsize=(3.35, 3.7))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Recall", "shrink": 0.72},
        annot_kws={"fontsize": 6.2},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=6.5, rotation=0)
    ax.tick_params(axis="x", labelsize=7, rotation=0)
    ax.set_title("UNSW-NB15 multiclass recall by attack family", fontsize=8.2)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_all(results_dir: Path) -> None:
    figures = results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paired = pd.read_csv(results_dir / "paired_comparisons.csv")
    xai = pd.read_csv(results_dir / "explanation_summary.csv")
    per_class = pd.read_csv(results_dir / "per_class_metrics.csv")
    workflow_figure(figures / "paper_workflow.png")
    paired_fidelity_figure(paired, figures / "paper_paired_fidelity.png")
    latency_configuration_figure(xai, figures / "paper_latency_configurations.png")
    class_recall_figure(per_class, figures / "paper_unsw_recall.png")
