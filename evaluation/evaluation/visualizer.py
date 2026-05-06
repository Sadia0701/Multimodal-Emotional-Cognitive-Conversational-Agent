"""
=============================================================================
visualizer.py  —  Publication-Quality Result Visualisation
=============================================================================
Generates all figures used in the thesis Results chapter:

  Fig 1  — Bar chart: response quality metrics (BLEU, ROUGE)
  Fig 2  — Bar chart: diversity metrics (DIST-1/2) + Empathy
  Fig 3  — Radar chart: normalised overall comparison
  Fig 4  — Emotion recognition: Accuracy & F1 grouped bars
  Fig 5  — Confusion matrix (proposed system)
  Fig 6  — Action distribution (cognitive layer)
  Fig 7  — Latency distribution (box plot)
  Fig 8  — Per-sample ROUGE-L: cognitive vs ablation
=============================================================================
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
from typing import Dict, List

from runner          import CONDITION_LABELS, CONDITION_COLORS
from metrics         import MetricsCalculator


# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       180,
    "savefig.bbox":      "tight",
})

_CMAP = {
    "with_cognition": "#2563EB",
    "no_cognition":   "#DC2626",
    "vanilla_gpt":    "#059669",
}

_SHORT = {
    "with_cognition": "Proposed",
    "no_cognition":   "Ablation",
    "vanilla_gpt":    "Vanilla GPT",
}


# ── Helper ────────────────────────────────────────────────────────────────────
def _bar_label(ax, bars, fmt="{:.3f}", fontsize=9):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.003,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold",
        )


def _clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


# ── Visualiser class ──────────────────────────────────────────────────────────
class ResultsVisualizer:

    def __init__(self, results: dict, output_dir: str = "evaluation_results"):
        self.results    = results
        self.output_dir = output_dir
        self.plot_dir   = os.path.join(output_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        self.calc       = MetricsCalculator()
        self.keys       = list(results.keys())

    def save(self, name: str, fig: plt.Figure):
        path = os.path.join(self.plot_dir, name)
        fig.savefig(path)
        plt.close(fig)
        print(f"  ✓ Saved: plots/{name}")

    def generate_all(self):
        print("\n  Generating plots ...")
        self.plot_response_quality()
        self.plot_diversity_empathy()
        self.plot_radar()
        self.plot_emotion()
        self.plot_confusion_matrix()
        self.plot_action_distribution()
        self.plot_latency_boxplot()
        self.plot_per_sample_rougel()
        self.plot_perplexity()          # NEW — replaces strategy_alignment
        print("  ✓ All plots generated\n")

    # ── Fig 1: Response quality ───────────────────────────────────────────────
    def plot_response_quality(self):
        metrics = ["BLEU-1", "BLEU-2", "ROUGE-1", "ROUGE-2", "ROUGE-L"]
        n       = len(metrics)
        x       = np.arange(n)
        width   = 0.25

        fig, ax = plt.subplots(figsize=(13, 6))

        for i, key in enumerate(self.keys):
            vals  = [self.results[key]["metrics"].get(m, 0) for m in metrics]
            bars  = ax.bar(x + i * width, vals, width,
                           label=_SHORT[key], color=_CMAP[key],
                           edgecolor="white", linewidth=0.6)
            _bar_label(ax, bars)

        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics)
        ax.set_ylabel("Score")
        ax.set_title("Response Quality Metrics — BLEU & ROUGE Comparison")
        ax.set_ylim(0, max(
            self.results[k]["metrics"].get(m, 0)
            for k in self.keys for m in metrics
        ) * 1.28 + 0.01)
        ax.legend()
        _clean_ax(ax)

        self.save("fig1_response_quality.png", fig)

    # ── Fig 2: Diversity + Empathy ────────────────────────────────────────────
    def plot_diversity_empathy(self):
        metrics = ["DIST-1", "DIST-2", "Empathy-Score"]
        n, width = len(metrics), 0.25
        x        = np.arange(n)

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, key in enumerate(self.keys):
            vals = [self.results[key]["metrics"].get(m, 0) for m in metrics]
            bars = ax.bar(x + i * width, vals, width,
                          label=_SHORT[key], color=_CMAP[key],
                          edgecolor="white", linewidth=0.6)
            _bar_label(ax, bars)

        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics)
        ax.set_ylabel("Score")
        ax.set_title("Diversity & Empathy Metrics Comparison\n"
                     "(Strategy-Alignment excluded — ceiling effect across all systems)")
        ax.set_ylim(0, 1.25)
        ax.legend()
        _clean_ax(ax)

        self.save("fig2_diversity_empathy.png", fig)

    # ── Fig 3: Radar chart ────────────────────────────────────────────────────
    def plot_radar(self):
        radar_metrics = [
            "BLEU-1", "ROUGE-L", "DIST-2",
            "Emotion-Acc", "Empathy-Score", "Emotion-F1-W",
        ]

        all_vals = {
            k: [self.results[k]["metrics"].get(m, 0) for m in radar_metrics]
            for k in self.keys
        }
        max_vals = [max(all_vals[k][i] for k in self.keys) + 1e-10
                    for i in range(len(radar_metrics))]
        norm_vals = {
            k: [v / max_vals[i] for i, v in enumerate(all_vals[k])]
            for k in self.keys
        }

        N      = len(radar_metrics)
        angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for key in self.keys:
            vals = norm_vals[key] + [norm_vals[key][0]]
            ax.plot(angles, vals, "o-", lw=2,
                    label=_SHORT[key], color=_CMAP[key])
            ax.fill(angles, vals, alpha=0.12, color=_CMAP[key])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_metrics, size=11)
        ax.set_ylim(0, 1)
        ax.set_title("Overall System Comparison\n(Normalised Metrics)",
                     size=14, pad=22)
        ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12))
        ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

        self.save("fig3_radar_chart.png", fig)

    # ── Fig 4: Emotion Accuracy & F1 ─────────────────────────────────────────
    def plot_emotion(self):
        metrics = ["Emotion-Acc", "Emotion-F1-W", "Emotion-F1-M"]
        n, width = len(metrics), 0.25
        x        = np.arange(n)

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, key in enumerate(self.keys):
            vals = [self.results[key]["metrics"].get(m, 0) for m in metrics]
            bars = ax.bar(x + i * width, vals, width,
                          label=_SHORT[key], color=_CMAP[key],
                          edgecolor="white", linewidth=0.6)
            _bar_label(ax, bars)

        ax.set_xticks(x + width)
        ax.set_xticklabels(["Accuracy", "F1 (Weighted)", "F1 (Macro)"])
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.2)
        ax.set_title("Emotion Recognition Performance")
        ax.legend()
        _clean_ax(ax)

        self.save("fig4_emotion_performance.png", fig)

    # ── Fig 5: Confusion matrix (proposed system) ─────────────────────────────
    def plot_confusion_matrix(self):
        if "with_cognition" not in self.results:
            return

        data  = self.results["with_cognition"]
        calc  = MetricsCalculator()
        cm, labels = calc.confusion_matrix_data(
            data["true_emotions"], data["pred_emotions"]
        )

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            cm, annot=True, fmt="d",
            xticklabels=labels, yticklabels=labels,
            cmap="Blues", linewidths=0.5, ax=ax,
        )
        ax.set_xlabel("Predicted Emotion")
        ax.set_ylabel("True Emotion")
        ax.set_title("Emotion Confusion Matrix — Proposed System\n(With Cognitive Layer)")
        plt.xticks(rotation=30, ha="right")
        plt.yticks(rotation=0)

        self.save("fig5_confusion_matrix.png", fig)

    # ── Fig 6: Action distribution ────────────────────────────────────────────
    def plot_action_distribution(self):
        if "with_cognition" not in self.results:
            return

        actions      = self.results["with_cognition"]["actions"]
        action_counts = Counter(actions)

        labels = list(action_counts.keys())
        counts = [action_counts[l] for l in labels]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(labels, counts, color="#2563EB", edgecolor="white")

        for bar, c in zip(bars, counts):
            ax.text(bar.get_width() + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    str(c), va="center", fontweight="bold")

        ax.set_xlabel("Frequency")
        ax.set_title(
            "Cognitive Layer: Action Selection Distribution\n"
            "(Mental Health / ESConv Domain)"
        )
        _clean_ax(ax)

        self.save("fig6_action_distribution.png", fig)

    # ── Fig 7: Latency box plot ────────────────────────────────────────────────
    def plot_latency_boxplot(self):
        data_boxes = [
            self.results[k]["latencies"] for k in self.keys
        ]
        short_labels = [_SHORT[k] for k in self.keys]
        colors       = [_CMAP[k]  for k in self.keys]

        fig, ax = plt.subplots(figsize=(9, 6))
        bp = ax.boxplot(data_boxes, patch_artist=True,
                        medianprops=dict(color="white", linewidth=2))

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.set_xticklabels(short_labels)
        ax.set_ylabel("Response Latency (seconds)")
        ax.set_title("Response Latency Distribution by System")
        _clean_ax(ax)

        self.save("fig7_latency_boxplot.png", fig)

    # ── Fig 8: Per-sample ROUGE-L comparison ─────────────────────────────────
    def plot_per_sample_rougel(self):
        if "with_cognition" not in self.results or "no_cognition" not in self.results:
            return

        wc = self.results["with_cognition"]["per_rouge_l"]
        nc = self.results["no_cognition"]["per_rouge_l"]
        n  = min(len(wc), len(nc))

        x    = np.arange(n)
        diff = np.array(wc[:n]) - np.array(nc[:n])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        # Top: both traces
        ax1.plot(x, wc[:n], color=_CMAP["with_cognition"],
                 label="Proposed", lw=1.5, alpha=0.85)
        ax1.plot(x, nc[:n], color=_CMAP["no_cognition"],
                 label="Ablation", lw=1.5, alpha=0.75)
        ax1.set_ylabel("ROUGE-L (F1)")
        ax1.set_title("Per-Sample ROUGE-L: Proposed vs Ablation")
        ax1.legend()
        _clean_ax(ax1)

        # Bottom: difference
        colors_diff = [_CMAP["with_cognition"] if d >= 0 else _CMAP["no_cognition"]
                       for d in diff]
        ax2.bar(x, diff, color=colors_diff, width=0.9, alpha=0.8)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Δ ROUGE-L (Proposed − Ablation)")
        ax2.set_title("Sample-Level Improvement from Cognitive Layer")
        _clean_ax(ax2)

        self.save("fig8_per_sample_rougel.png", fig)

    # ── Fig 9: Perplexity comparison ──────────────────────────────────────────
    def plot_perplexity(self):
        """
        Grouped bar chart of mean perplexity per system.
        Lower PPL = better fluency / coherence.
        Annotates which method was used (GPT-2 or N-gram LM).
        """
        keys   = list(self.keys)
        labels = [_SHORT[k] for k in keys]
        vals   = [self.results[k]["metrics"].get("Perplexity", float("nan"))
                  for k in keys]
        colors = [_CMAP[k] for k in keys]

        # Skip if all NaN
        if all(np.isnan(v) for v in vals):
            print("  ⚠ Perplexity values all NaN — skipping fig9")
            return

        # Replace NaN with 0 for plotting
        plot_vals = [v if np.isfinite(v) else 0 for v in vals]

        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.bar(labels, plot_vals, color=colors, width=0.45,
                      edgecolor="white", linewidth=0.6)

        for bar, v in zip(bars, vals):
            label_txt = f"{v:.2f}" if np.isfinite(v) else "N/A"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(plot_vals) * 0.02,
                label_txt,
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
            )

        ax.set_ylabel("Perplexity (↓ lower is better)")
        ax.set_title(
            "Response Fluency — Perplexity Comparison\n"
            "(GPT-2 / N-gram LM — lower = more fluent & coherent)"
        )
        ax.set_ylim(0, max(v for v in plot_vals if v > 0) * 1.3 + 1)

        # Annotate best (lowest PPL)
        finite_pairs = [(v, k) for v, k in zip(vals, keys) if np.isfinite(v)]
        if finite_pairs:
            best_val, best_key = min(finite_pairs)
            best_idx = keys.index(best_key)
            ax.annotate(
                "Best",
                xy=(best_idx, best_val),
                xytext=(best_idx, best_val + max(plot_vals) * 0.12),
                ha="center",
                fontsize=9,
                color=_CMAP[best_key],
                arrowprops=dict(arrowstyle="->", color=_CMAP[best_key]),
            )

        _clean_ax(ax)
        self.save("fig9_perplexity.png", fig)
