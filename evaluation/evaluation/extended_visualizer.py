"""
=============================================================================
extended_visualizer.py  —  Complete Thesis Figure Generator
=============================================================================
Generates all 20+ figures across all 4 experiments:

  EXP 1 : Original comparison     (figs 1–9,  re-uses existing visualizer)
  EXP 2 : Component ablation      (figs 10–13)
  EXP 3 : Modality ablation       (figs 14–17)
  EXP 4 : Multi-dataset           (figs 18–20)
  EXTRA : Published baseline comp  (fig 21)
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Optional

from component_ablation  import COMPONENT_CONDITIONS, COMPONENT_COLORS
from modality_ablation   import MODALITY_CONDITIONS,  MODALITY_COLORS
from published_baselines import get_esconv_baselines, get_iemocap_baselines, get_meld_baselines

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "figure.dpi":       150,
    "savefig.dpi":      180,
    "savefig.bbox":     "tight",
})

_EXP1_COLORS = {
    "with_cognition": "#2563EB",
    "no_cognition":   "#DC2626",
    "vanilla_gpt":    "#059669",
}
_EXP1_SHORT = {
    "with_cognition": "Proposed",
    "no_cognition":   "Ablation",
    "vanilla_gpt":    "Vanilla GPT",
}


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.28, linestyle="--")


def _bar_labels(ax, bars, fmt="{:.3f}", fs=8.5):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + max(h*0.02, 0.002),
                fmt.format(h), ha="center", va="bottom", fontsize=fs, fontweight="bold")


class ExtendedVisualizer:

    def __init__(
        self,
        exp1: Dict = None, exp2: Dict = None,
        exp3: Dict = None, exp4: Dict = None,
        output_dir: str = "evaluation_results",
    ):
        self.exp1 = exp1 or {}
        self.exp2 = exp2 or {}
        self.exp3 = exp3 or {}
        self.exp4 = exp4 or {}
        self.plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)

    def save(self, name: str, fig: plt.Figure):
        path = os.path.join(self.plot_dir, name)
        fig.savefig(path)
        plt.close(fig)
        print(f"  ✓ {name}")

    def generate_all(self):
        print("\n  Generating all thesis figures...")

        if self.exp1:
            self.fig_exp1_quality()
            self.fig_exp1_diversity_empathy()
            self.fig_exp1_radar()
            self.fig_exp1_emotion()

        if self.exp2:
            self.fig_exp2_component_bars()
            self.fig_exp2_component_radar()
            self.fig_exp2_module_contribution()

        if self.exp3:
            self.fig_exp3_modality_bars()
            self.fig_exp3_modality_radar()
            self.fig_exp3_modality_emotion()
            self.fig_exp3_fusion_gain()

        if self.exp4:
            self.fig_exp4_multidataset()
            self.fig_exp4_vs_published()

        self.fig_summary_heatmap()
        print(f"  ✓ All figures saved to {self.plot_dir}/\n")

    # ══════════════════════════════════════════════════════════════════════════
    # EXP 1 — Original comparison
    # ══════════════════════════════════════════════════════════════════════════

    def fig_exp1_quality(self):
        metrics = ["BLEU-1", "BLEU-2", "ROUGE-1", "ROUGE-2", "ROUGE-L"]
        keys    = list(self.exp1.keys())
        x, w    = np.arange(len(metrics)), 0.25

        fig, ax = plt.subplots(figsize=(13, 6))
        for i, k in enumerate(keys):
            vals = [self.exp1[k]["metrics"].get(m, 0) for m in metrics]
            bars = ax.bar(x + i*w, vals, w,
                          label=_EXP1_SHORT.get(k, k), color=_EXP1_COLORS.get(k, "#888"),
                          edgecolor="white", lw=0.6)
            _bar_labels(ax, bars)

        ax.set_xticks(x + w); ax.set_xticklabels(metrics)
        ax.set_ylabel("Score"); ax.set_title("EXP 1 — Response Quality: BLEU & ROUGE")
        ax.set_ylim(0, max(self.exp1[k]["metrics"].get(m,0)
                           for k in keys for m in metrics) * 1.3 + 0.01)
        ax.legend(); _clean(ax)
        self.save("fig01_exp1_quality.png", fig)

    def fig_exp1_diversity_empathy(self):
        metrics = ["DIST-1", "DIST-2", "Empathy-Score"]
        keys    = list(self.exp1.keys())
        x, w    = np.arange(len(metrics)), 0.25

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, k in enumerate(keys):
            vals = [self.exp1[k]["metrics"].get(m, 0) for m in metrics]
            bars = ax.bar(x + i*w, vals, w,
                          label=_EXP1_SHORT.get(k, k), color=_EXP1_COLORS.get(k, "#888"),
                          edgecolor="white", lw=0.6)
            _bar_labels(ax, bars)

        ax.set_xticks(x + w); ax.set_xticklabels(metrics)
        ax.set_ylabel("Score"); ax.set_title("EXP 1 — Diversity & Empathy")
        ax.set_ylim(0, 1.25); ax.legend(); _clean(ax)
        self.save("fig02_exp1_diversity_empathy.png", fig)

    def fig_exp1_radar(self):
        dims = ["BLEU-1","ROUGE-L","DIST-2","Emotion-Acc","Empathy-Score","Emotion-F1-W"]
        keys = list(self.exp1.keys())

        all_v = {k: [self.exp1[k]["metrics"].get(d, 0) for d in dims] for k in keys}
        max_v = [max(all_v[k][i] for k in keys)+1e-10 for i in range(len(dims))]
        norm  = {k: [v/max_v[i] for i,v in enumerate(all_v[k])] for k in keys}

        N      = len(dims)
        angles = [n/N*2*np.pi for n in range(N)] + [0]
        fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
        for k in keys:
            vals = norm[k] + [norm[k][0]]
            ax.plot(angles, vals, "o-", lw=2,
                    label=_EXP1_SHORT.get(k,k), color=_EXP1_COLORS.get(k,"#888"))
            ax.fill(angles, vals, alpha=0.1, color=_EXP1_COLORS.get(k,"#888"))

        ax.set_xticks(angles[:-1]); ax.set_xticklabels(dims, size=10)
        ax.set_ylim(0,1); ax.legend(loc="upper right", bbox_to_anchor=(1.32,1.12))
        ax.set_title("EXP 1 — Overall System Comparison (Normalised)", size=14, pad=22)
        self.save("fig03_exp1_radar.png", fig)

    def fig_exp1_emotion(self):
        metrics = ["Emotion-Acc","Emotion-F1-W","Emotion-F1-M"]
        keys    = list(self.exp1.keys())
        x, w    = np.arange(len(metrics)), 0.25

        fig, ax = plt.subplots(figsize=(10,6))
        for i, k in enumerate(keys):
            vals = [self.exp1[k]["metrics"].get(m, 0) for m in metrics]
            bars = ax.bar(x+i*w, vals, w,
                          label=_EXP1_SHORT.get(k,k), color=_EXP1_COLORS.get(k,"#888"),
                          edgecolor="white", lw=0.6)
            _bar_labels(ax, bars)

        ax.set_xticks(x+w); ax.set_xticklabels(["Accuracy","F1 (Weighted)","F1 (Macro)"])
        ax.set_ylabel("Score"); ax.set_ylim(0,1.2)
        ax.set_title("EXP 1 — Emotion Recognition Performance"); ax.legend(); _clean(ax)
        self.save("fig04_exp1_emotion.png", fig)

    def fig_exp2_component_bars(self):
        metrics = ["BLEU-1","ROUGE-L","Empathy-Score","Emotion-Acc"]
        keys    = list(self.exp2.keys())
        x, w    = np.arange(len(metrics)), 0.16

        fig, ax = plt.subplots(figsize=(15, 6))
        for i, k in enumerate(keys):
            vals  = [self.exp2[k]["metrics"].get(m, 0) or 0 for m in metrics]
            color = COMPONENT_COLORS.get(k, "#888")
            label = COMPONENT_CONDITIONS.get(k, k)
            bars  = ax.bar(x + i*w, vals, w, label=label, color=color,
                           edgecolor="white", lw=0.5, alpha=0.9)

        ax.set_xticks(x + w*2)
        ax.set_xticklabels(["BLEU-1","ROUGE-L","Empathy","Emotion-Acc"])
        ax.set_ylabel("Score"); ax.set_title("EXP 2 — Component Ablation: Effect of Each Cognitive Module")
        ax.legend(fontsize=9, loc="upper right"); _clean(ax)
        self.save("fig10_exp2_component_bars.png", fig)

    def fig_exp2_component_radar(self):
        dims = ["BLEU-1","ROUGE-L","DIST-2","Emotion-Acc","Empathy-Score"]
        keys = list(self.exp2.keys())

        all_v = {k: [self.exp2[k]["metrics"].get(d,0) or 0 for d in dims] for k in keys}
        max_v = [max(all_v[k][i] for k in keys)+1e-10 for i in range(len(dims))]
        norm  = {k: [v/max_v[i] for i,v in enumerate(all_v[k])] for k in keys}

        N      = len(dims)
        angles = [n/N*2*np.pi for n in range(N)] + [0]
        fig, ax = plt.subplots(figsize=(9,9), subplot_kw=dict(polar=True))

        for k in keys:
            vals  = norm[k] + [norm[k][0]]
            color = COMPONENT_COLORS.get(k,"#888")
            ax.plot(angles, vals, "o-", lw=2,
                    label=COMPONENT_CONDITIONS.get(k,k), color=color)
            ax.fill(angles, vals, alpha=0.08, color=color)

        ax.set_xticks(angles[:-1]); ax.set_xticklabels(dims, size=10)
        ax.set_ylim(0,1)
        ax.set_title("EXP 2 — Component Ablation Radar\n(Normalised, Full System = Best Expected)",
                     size=13, pad=22)
        ax.legend(loc="upper right", bbox_to_anchor=(1.38,1.12), fontsize=9)
        self.save("fig11_exp2_radar.png", fig)

    def fig_exp2_module_contribution(self):
        """Bar chart showing % drop from full system when each module removed."""
        if "full" not in self.exp2:
            return

        metrics = ["BLEU-1","ROUGE-L","Empathy-Score","Emotion-Acc"]
        full_m  = self.exp2["full"]["metrics"]
        ablation_modes = [k for k in self.exp2 if k != "full"]

        x, w = np.arange(len(ablation_modes)), 0.18
        fig, ax = plt.subplots(figsize=(12, 6))

        for i, metric in enumerate(metrics):
            drops = []
            for k in ablation_modes:
                v_full = full_m.get(metric, 0) or 0
                v_abl  = self.exp2[k]["metrics"].get(metric, 0) or 0
                drop   = ((v_full - v_abl) / (v_full + 1e-10)) * 100
                drops.append(drop)

            bars = ax.bar(x + i*w, drops, w, label=metric, edgecolor="white", lw=0.5)

        ax.set_xticks(x + w*1.5)
        ax.set_xticklabels([COMPONENT_CONDITIONS.get(k,k).replace(" ","\\n")
                            for k in ablation_modes], fontsize=9)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("Performance Drop from Full System (%)")
        ax.set_title("EXP 2 — Module Contribution: % Performance Drop When Removed")
        ax.legend(fontsize=9); _clean(ax)
        self.save("fig12_exp2_contribution.png", fig)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP 3 — Modality Ablation
    # ══════════════════════════════════════════════════════════════════════════

    def fig_exp3_modality_bars(self):
        metrics = ["BLEU-1","ROUGE-L","Emotion-Acc","Empathy-Score"]
        keys    = list(self.exp3.keys())
        x, w    = np.arange(len(metrics)), 0.13

        fig, ax = plt.subplots(figsize=(15, 6))
        for i, k in enumerate(keys):
            vals  = [self.exp3[k]["metrics"].get(m, 0) or 0 for m in metrics]
            color = MODALITY_COLORS.get(k,"#888")
            label = MODALITY_CONDITIONS.get(k,k)
            ax.bar(x + i*w, vals, w, label=label, color=color,
                   edgecolor="white", lw=0.5, alpha=0.9)

        ax.set_xticks(x + w*2.5)
        ax.set_xticklabels(["BLEU-1","ROUGE-L","Emotion-Acc","Empathy"])
        ax.set_ylabel("Score")
        ax.set_title("EXP 3 — Modality Ablation: Impact of Each Input Modality\n"
                     "(All Modalities = expected best)")
        ax.legend(fontsize=8, loc="upper right"); _clean(ax)
        self.save("fig14_exp3_modality_bars.png", fig)

    def fig_exp3_modality_radar(self):
        dims = ["BLEU-1","ROUGE-L","DIST-2","Emotion-Acc","Empathy-Score"]
        keys = list(self.exp3.keys())

        all_v = {k: [self.exp3[k]["metrics"].get(d,0) or 0 for d in dims] for k in keys}
        max_v = [max(all_v[k][i] for k in keys)+1e-10 for i in range(len(dims))]
        norm  = {k: [v/max_v[i] for i,v in enumerate(all_v[k])] for k in keys}

        N      = len(dims)
        angles = [n/N*2*np.pi for n in range(N)] + [0]
        fig, ax = plt.subplots(figsize=(9,9), subplot_kw=dict(polar=True))

        for k in keys:
            vals  = norm[k] + [norm[k][0]]
            color = MODALITY_COLORS.get(k,"#888")
            ls    = "-" if k == "all_modalities" else "--"
            ax.plot(angles, vals, "o"+ls, lw=2.5 if k=="all_modalities" else 1.5,
                    label=MODALITY_CONDITIONS.get(k,k), color=color)
            ax.fill(angles, vals, alpha=0.06, color=color)

        ax.set_xticks(angles[:-1]); ax.set_xticklabels(dims, size=10)
        ax.set_ylim(0,1)
        ax.set_title("EXP 3 — Modality Ablation Radar\n(All Modalities should cover most area)",
                     size=13, pad=22)
        ax.legend(loc="upper right", bbox_to_anchor=(1.45,1.12), fontsize=8)
        self.save("fig15_exp3_radar.png", fig)

    def fig_exp3_modality_emotion(self):
        """Emotion accuracy specifically, showing modality contribution."""
        keys  = list(self.exp3.keys())
        accs  = [self.exp3[k]["metrics"].get("Emotion-Acc", 0) or 0 for k in keys]
        lbls  = [MODALITY_CONDITIONS.get(k,k) for k in keys]
        clrs  = [MODALITY_COLORS.get(k,"#888") for k in keys]

        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.barh(lbls, accs, color=clrs, height=0.55, edgecolor="white", lw=0.5)
        for b, v in zip(bars, accs):
            ax.text(b.get_width()+0.005, b.get_y()+b.get_height()/2,
                    f"{v:.3f}", va="center", fontweight="bold", fontsize=10)

        ax.set_xlabel("Emotion Recognition Accuracy")
        ax.set_title("EXP 3 — Emotion Accuracy per Input Modality\n"
                     "(Demonstrates why multimodal fusion outperforms single modalities)")
        ax.set_xlim(0, max(accs)*1.2 + 0.01); _clean(ax)
        self.save("fig16_exp3_emotion_accuracy.png", fig)

    def fig_exp3_fusion_gain(self):
        """Shows incremental gain from adding each modality."""
        order  = ["text_only","speech_only","face_only","text_speech","text_face","all_modalities"]
        metric = "Empathy-Score"
        vals   = [self.exp3.get(k, {}).get("metrics", {}).get(metric, 0) or 0 for k in order]
        lbls   = [MODALITY_CONDITIONS.get(k,k) for k in order]
        clrs   = [MODALITY_COLORS.get(k,"#888") for k in order]

        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(lbls, vals, color=clrs, width=0.55, edgecolor="white", lw=0.6)
        _bar_labels(ax, bars)
        ax.set_ylabel("Empathy Score")
        ax.set_title("EXP 3 — Empathy Score Across Modality Conditions\n"
                     "(Expected monotonic improvement as modalities are added)")
        ax.set_ylim(0, 1.25); _clean(ax)
        plt.xticks(rotation=20, ha="right", fontsize=9)
        self.save("fig17_exp3_fusion_gain.png", fig)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP 4 — Multi-dataset
    # ══════════════════════════════════════════════════════════════════════════

    def fig_exp4_multidataset(self):
        """Grouped bars: emotion F1-W across datasets and conditions."""
        datasets = list(self.exp4.keys())
        cond_map = {
            "with_cognition": ("Proposed",     "#2563EB"),
            "no_cognition":   ("No Cognition", "#DC2626"),
            "vanilla_gpt":    ("Vanilla GPT",  "#059669"),
        }

        x  = np.arange(len(datasets))
        w  = 0.26
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, (cond_key, (label, color)) in enumerate(cond_map.items()):
            vals = [
                self.exp4.get(ds, {}).get(cond_key, {}).get("metrics", {}).get("Emotion-F1-W", 0) or 0
                for ds in datasets
            ]
            bars = ax.bar(x + i*w, vals, w, label=label, color=color,
                          edgecolor="white", lw=0.6)
            _bar_labels(ax, bars)

        ax.set_xticks(x + w)
        ax.set_xticklabels([ds.upper() for ds in datasets])
        ax.set_ylabel("Emotion F1 (Weighted)")
        ax.set_ylim(0, 1.2)
        ax.set_title("EXP 4 — Cross-Dataset Emotion Recognition\n(Generalisation across IEMOCAP & MELD)")
        ax.legend(); _clean(ax)
        self.save("fig18_exp4_multidataset.png", fig)

    def fig_exp4_vs_published(self):
        """Our system vs published baselines on IEMOCAP and MELD."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax, (ds_name, pub_baselines) in zip(
            axes,
            [("iemocap", get_iemocap_baselines()), ("meld", get_meld_baselines())]
        ):
            pub_names = [b["name"] for b in pub_baselines]
            pub_f1w   = [b.get("Emotion-F1-W") or 0 for b in pub_baselines]
            pub_years  = [b["year"] for b in pub_baselines]

            # Our system
            our_f1w = (
                self.exp4.get(ds_name, {})
                    .get("with_cognition", {})
                    .get("metrics", {})
                    .get("Emotion-F1-W", 0) or 0
            )

            all_names = pub_names + ["Proposed\n(Ours)"]
            all_vals  = pub_f1w  + [our_f1w]
            all_colors = []

            for b in pub_baselines:
                if "zero-shot" in b["name"].lower() or "GPT" in b["name"] or "LLaVA" in b["name"]:
                    all_colors.append("#94a3b8")   # grey = LLM baselines
                elif b["year"] <= 2020:
                    all_colors.append("#cbd5e1")   # lighter = older
                else:
                    all_colors.append("#60a5fa")   # blue = recent supervised
            all_colors.append("#2563EB")           # our system = solid blue

            bars = ax.bar(all_names, all_vals, color=all_colors,
                          edgecolor="white", lw=0.6, width=0.6)
            for b, v in zip(bars, all_vals):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.008,
                        f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")

            ax.set_title(f"{ds_name.upper()} — Emotion F1-W vs Published Systems")
            ax.set_ylabel("Emotion F1 (Weighted)")
            ax.set_ylim(0, 1.05)
            ax.axhline(our_f1w, color="#2563EB", lw=1.2, linestyle="--", alpha=0.5)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
            _clean(ax)

            # Legend
            patches = [
                mpatches.Patch(color="#cbd5e1", label="Supervised (older)"),
                mpatches.Patch(color="#60a5fa", label="Supervised (recent)"),
                mpatches.Patch(color="#94a3b8", label="LLM zero-shot"),
                mpatches.Patch(color="#2563EB", label="Proposed (ours)"),
            ]
            ax.legend(handles=patches, fontsize=8, loc="lower right")

        fig.suptitle("EXP 4 — Comparison with Published Baselines\n"
                     "(Cognitive Architecture vs Supervised + LLM Systems)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        self.save("fig19_exp4_vs_published.png", fig)

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY HEATMAP — all experiments in one figure
    # ══════════════════════════════════════════════════════════════════════════

    def fig_summary_heatmap(self):
        """
        Master summary heatmap: rows = all conditions across all experiments,
        columns = key metrics. Instantly shows which conditions win where.
        """
        metrics = ["BLEU-1","ROUGE-L","DIST-2","Emotion-Acc","Empathy-Score"]
        rows    = []
        row_labels = []

        for exp_name, exp_data, short_prefix in [
            ("EXP1", self.exp1, "E1"),
            ("EXP2", self.exp2, "E2"),
            ("EXP3", self.exp3, "E3"),
        ]:
            for k, v in exp_data.items():
                label = v.get("label", k)[:28]
                row   = [v["metrics"].get(m, np.nan) or np.nan for m in metrics]
                rows.append(row)
                row_labels.append(f"[{short_prefix}] {label}")

        if not rows:
            return

        mat  = np.array(rows, dtype=float)
        # Normalise each column to [0,1]; for PPL invert (lower=better)
        norm = np.zeros_like(mat)
        for j, m in enumerate(metrics):
            col     = mat[:, j]
            col_min = np.nanmin(col)
            col_max = np.nanmax(col)
            rng     = col_max - col_min + 1e-10
            norm[:, j] = (col - col_min) / rng

        fig, ax = plt.subplots(figsize=(13, max(6, len(rows) * 0.55)))
        sns.heatmap(
            norm,
            annot=np.round(mat, 3), fmt=".3f",
            xticklabels=metrics, yticklabels=row_labels,
            cmap="RdYlGn", vmin=0, vmax=1,
            linewidths=0.4, ax=ax,
            annot_kws={"size": 8},
        )
        ax.set_title("Master Summary Heatmap — All Experiments\n"
                     "(Green = best in column, Red = worst)", pad=14)
        ax.set_xlabel("Metric"); ax.set_ylabel("Condition")
        plt.xticks(rotation=25, ha="right")
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        self.save("fig20_summary_heatmap.png", fig)
