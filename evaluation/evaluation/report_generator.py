"""
=============================================================================
report_generator.py  —  Thesis Report Generator
=============================================================================
Reads evaluation_results/ and produces:

  1. LaTeX table (results chapter)
  2. Markdown summary table
  3. Qualitative sample comparison (for thesis appendix)
  4. Significance test table in LaTeX
=============================================================================
"""

import os
import json
import numpy as np
from typing import Dict, List

from runner import CONDITION_LABELS


# ── Formatting helpers ────────────────────────────────────────────────────────
def _bold(val: str, is_best: bool) -> str:
    return f"\\textbf{{{val}}}" if is_best else val


def _fmt(v, decimals=4) -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


# ── Main generator ────────────────────────────────────────────────────────────
class ReportGenerator:

    METRIC_DISPLAY = {
        "BLEU-1":              "BLEU-1",
        "BLEU-2":              "BLEU-2",
        "ROUGE-1":             "ROUGE-1",
        "ROUGE-2":             "ROUGE-2",
        "ROUGE-L":             "ROUGE-L",
        "DIST-1":              "DIST-1",
        "DIST-2":              "DIST-2",
        "Emotion-Acc":         "Emotion Accuracy",
        "Emotion-F1-W":        "Emotion F1 (W)",
        "Emotion-F1-M":        "Emotion F1 (M)",
        "Empathy-Score":       "Empathy Score",
        "Avg-Length":          "Avg. Length (tokens)",
        "Avg-Latency":         "Avg. Latency (s)",
    }

    # metrics where HIGHER is better
    HIGHER_IS_BETTER = {
        "BLEU-1", "BLEU-2", "ROUGE-1", "ROUGE-2", "ROUGE-L",
        "DIST-1", "DIST-2", "Emotion-Acc", "Emotion-F1-W", "Emotion-F1-M",
        "Empathy-Score",
    }

    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = output_dir
        self._load()

    def _load(self):
        with open(f"{self.output_dir}/detailed_results.json") as f:
            self.data = json.load(f)

        with open(f"{self.output_dir}/significance_tests.json") as f:
            self.sig = json.load(f)

    def generate_all(self):
        self.latex_main_table()
        self.markdown_summary()
        self.qualitative_examples()
        self.ablation_improvement_table()
        print("  ✓ Report files generated in evaluation_results/")

    # =========================================================================
    # LATEX MAIN RESULTS TABLE
    # =========================================================================

    def latex_main_table(self):
        metrics  = list(self.METRIC_DISPLAY.keys())
        sys_keys = list(self.data.keys())

        # Determine best value per metric (lower is better for Perplexity)
        best: Dict[str, str] = {}
        for m in metrics:
            vals = {k: self.data[k]["metrics"].get(m) for k in sys_keys}
            vals = {k: v for k, v in vals.items() if v is not None}
            if vals:
                if m in self.HIGHER_IS_BETTER:
                    best[m] = max(vals, key=vals.get)
                else:
                    best[m] = min(vals, key=vals.get)

        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\small",
            r"\caption{Evaluation Results on ESConv Mental-Health Dataset. "
            r"Bold = best score. $\dagger$ = statistically significant over baseline ($p<0.05$).}",
            r"\label{tab:main_results}",
            r"\begin{tabular}{l" + "c" * len(sys_keys) + "}",
            r"\toprule",
        ]

        # Header
        col_heads = " & ".join(
            f"\\textbf{{{CONDITION_LABELS[k][:24]}}}" for k in sys_keys
        )
        lines.append(f"\\textbf{{Metric}} & {col_heads} \\\\")
        lines.append(r"\midrule")

        # Group metrics
        groups = [
            ("Response Quality", ["BLEU-1", "BLEU-2", "ROUGE-1", "ROUGE-2", "ROUGE-L"]),
            ("Diversity",        ["DIST-1", "DIST-2"]),
            ("Emotion",          ["Emotion-Acc", "Emotion-F1-W", "Emotion-F1-M"]),
            ("Empathy",          ["Empathy-Score"]),
            ("Efficiency",       ["Avg-Length", "Avg-Latency"]),
        ]

        for group_name, group_metrics in groups:
            lines.append(f"\\multicolumn{{{1 + len(sys_keys)}}}{{l}}"
                         f"{{\\textit{{{group_name}}}}} \\\\")

            for m in group_metrics:
                if m not in self.METRIC_DISPLAY:
                    continue
                row_vals = []
                for k in sys_keys:
                    v = self.data[k]["metrics"].get(m)
                    s = _fmt(v, 2 if m in {"Avg-Length", "Avg-Latency"} else 4)
                    row_vals.append(_bold(s, best.get(m) == k))
                row = f"\\quad {self.METRIC_DISPLAY[m]} & " + " & ".join(row_vals) + r" \\"
                lines.append(row)

            lines.append(r"\midrule")

        lines[-1] = r"\bottomrule"   # replace last midrule
        lines += [
            r"\end{tabular}",
            r"\end{table}",
        ]

        path = f"{self.output_dir}/table_main_results.tex"
        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"  ✓ LaTeX table → {path}")

    # =========================================================================
    # MARKDOWN SUMMARY
    # =========================================================================

    def markdown_summary(self):
        metrics  = ["BLEU-1", "BLEU-2", "ROUGE-L",
                    "DIST-1", "DIST-2",
                    "Emotion-Acc", "Empathy-Score"]
        sys_keys = list(self.data.keys())
        sys_lbls = [CONDITION_LABELS[k] for k in sys_keys]

        lines = [
            "# Evaluation Results — Multimodal Cognitive Agent",
            "",
            "## Main Results Table",
            "",
            "| Metric | " + " | ".join(sys_lbls) + " |",
            "| --- |" + " --- |" * len(sys_keys),
        ]

        for m in metrics:
            row_vals = []
            all_v = [self.data[k]["metrics"].get(m) for k in sys_keys]
            valid = [v for v in all_v if v is not None]
            max_v = max(valid) if valid else None

            for v in all_v:
                s = _fmt(v) if v is not None else "—"
                row_vals.append(f"**{s}**" if v == max_v else s)

            lines.append(f"| {m} | " + " | ".join(row_vals) + " |")

        # Improvement section
        if "with_cognition" in self.data and "no_cognition" in self.data:
            wc = self.data["with_cognition"]["metrics"]
            nc = self.data["no_cognition"]["metrics"]

            lines += [
                "", "## Cognitive Layer Gain (Proposed vs Ablation)", "",
                "| Metric | Δ Absolute | Δ Relative |",
                "| --- | --- | --- |",
            ]

            for m in metrics:
                if m not in wc or m not in nc:
                    continue
                delta = wc[m] - nc[m]
                pct   = delta / (abs(nc[m]) + 1e-10) * 100
                sign  = "+" if delta >= 0 else ""
                lines.append(
                    f"| {m} | {sign}{delta:.4f} | {sign}{pct:.1f}% |"
                )

        path = f"{self.output_dir}/summary.md"
        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"  ✓ Markdown summary → {path}")

    # =========================================================================
    # LATEX SIGNIFICANCE TABLE
    # =========================================================================

    def qualitative_examples(self, n: int = 5):
        if "with_cognition" not in self.data:
            return

        wc_samples = self.data["with_cognition"]["samples"]
        nc_samples = (
            self.data["no_cognition"]["samples"]
            if "no_cognition" in self.data else []
        )

        nc_map = {s["idx"]: s for s in nc_samples}

        lines = [
            "# Qualitative Examples — Thesis Appendix",
            "",
            "Comparison of responses from all three systems on representative samples.",
            "",
        ]

        for i, wc in enumerate(wc_samples[:n]):
            idx = wc["idx"]
            nc  = nc_map.get(idx, {})

            lines += [
                f"---",
                f"### Example {i+1}",
                f"",
                f"**User Input:**  ",
                f"> {wc['user'][:280]}",
                f"",
                f"**Ground-Truth Reference:**  ",
                f"> {wc['reference'][:280]}",
                f"",
                f"**Proposed System (w/ Cognitive Layer):**  ",
                f"*(Emotion: {wc['pred_emotion']} | Action: {wc['action']})*  ",
                f"> {wc['generated'][:400]}",
                f"",
            ]

            if nc:
                lines += [
                    f"**Ablation (w/o Cognitive Layer):**  ",
                    f"> {nc.get('generated','—')[:400]}",
                    f"",
                ]

        path = f"{self.output_dir}/qualitative_examples.md"
        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"  ✓ Qualitative examples → {path}")

    # =========================================================================
    # ABLATION IMPROVEMENT TABLE
    # =========================================================================

    def ablation_improvement_table(self):
        if "with_cognition" not in self.data or "no_cognition" not in self.data:
            return

        wc = self.data["with_cognition"]["metrics"]
        nc = self.data["no_cognition"]["metrics"]

        all_metrics = list(wc.keys())

        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Ablation Study: Impact of Cognitive Layer "
            r"(Proposed System vs No-Cognition Baseline)}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"\textbf{Metric} & \textbf{Proposed} & \textbf{Ablation} "
            r"& \textbf{$\Delta$ (\%)} \\",
            r"\midrule",
        ]

        for m in all_metrics:
            if m in {"Avg-Length", "Avg-Latency"}:
                continue
            wv = wc.get(m)
            nv = nc.get(m)
            if wv is None or nv is None:
                continue
            delta = wv - nv
            pct   = delta / (abs(nv) + 1e-10) * 100
            sign  = "+" if delta >= 0 else ""
            arrow = r"$\uparrow$" if delta >= 0 else r"$\downarrow$"

            lines.append(
                f"{self.METRIC_DISPLAY.get(m, m)} & "
                f"\\textbf{{{_fmt(wv)}}} & "
                f"{_fmt(nv)} & "
                f"{sign}{pct:.1f}\\% {arrow} \\\\"
            )

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]

        path = f"{self.output_dir}/table_ablation.tex"
        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"  ✓ Ablation LaTeX table → {path}")
