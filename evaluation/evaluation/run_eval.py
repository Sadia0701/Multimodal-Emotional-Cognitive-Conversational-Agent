#!/usr/bin/env python3
"""
=============================================================================
run_eval.py  —  Master's Thesis Evaluation Entry Point
=============================================================================
USAGE
-----
  python run_eval.py --api-key sk-xxxx [--samples 100] [--model gpt-4o-mini]

WHAT IT DOES
------------
  1. Loads ESConv dataset (or synthetic fallback)
  2. Runs three experimental conditions:
       A. Proposed System  — with cognitive layer
       B. Ablation         — without cognitive layer
       C. Baseline         — vanilla GPT-4o (no emotional awareness)
  3. Computes all metrics:
       BLEU-1/2, ROUGE-1/2/L, DIST-1/2, BERTScore (if installed),
       Emotion Accuracy, Emotion F1 (weighted + macro),
       Empathy Score, Strategy Alignment
  5. Saves:
       evaluation_results/metrics_summary.csv
       evaluation_results/detailed_results.json
       evaluation_results/table_main_results.tex    ← paste into thesis
       evaluation_results/table_ablation.tex        ← ablation chapter
       evaluation_results/summary.md
       evaluation_results/qualitative_examples.md   ← appendix
       evaluation_results/plots/fig1_*.png ... fig9_*.png

ESTIMATED COST (gpt-4o-mini, 100 samples, 3 conditions)
---------------------------------------------------------
  ≈ 300 API calls × ~400 tokens avg ≈ 120,000 tokens ≈ $0.07 USD
=============================================================================
"""

import argparse
import sys
import os

# Make imports work from this directory
sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(
        description="Thesis Evaluation Framework — Multimodal Cognitive Agent"
    )
    parser.add_argument(
        "--api-key", required=True,
        help="OpenAI API key (starts with sk-...)"
    )
    parser.add_argument(
        "--samples", type=int, default=100,
        help="Number of evaluation samples (default: 100)"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="GPT model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.6,
        help="Seconds between API calls — increase if rate-limited (default: 0.6)"
    )
    parser.add_argument(
        "--output-dir", default="evaluation_results",
        help="Output directory (default: evaluation_results)"
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip plot generation (faster, text results only)"
    )
    parser.add_argument(
        "--skip-report", action="store_true",
        help="Skip LaTeX/Markdown report generation"
    )
    args = parser.parse_args()

    # ── Validate API key ──────────────────────────────────────────────────────
    if not args.api_key.startswith("sk-"):
        print("⚠  Warning: API key doesn't look like an OpenAI key (should start with 'sk-')")

    # ── Run evaluation ────────────────────────────────────────────────────────
    from runner import EvaluationRunner

    runner = EvaluationRunner(
        api_key     = args.api_key,
        max_samples = args.samples,
        api_delay   = args.delay,
        output_dir  = args.output_dir,
        model       = args.model,
    )

    outcome = runner.run_all()

    # ── Generate plots ────────────────────────────────────────────────────────
    if not args.skip_plots:
        try:
            from visualizer import ResultsVisualizer
            viz = ResultsVisualizer(outcome["results"], output_dir=args.output_dir)
            viz.generate_all()
        except Exception as e:
            print(f"  ⚠ Plot generation failed: {e}")

    # ── Generate reports ──────────────────────────────────────────────────────
    if not args.skip_report:
        try:
            from report_generator import ReportGenerator
            rg = ReportGenerator(output_dir=args.output_dir)
            rg.generate_all()
        except Exception as e:
            print(f"  ⚠ Report generation failed: {e}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  EVALUATION COMPLETE")
    print("═" * 65)
    print(f"  Samples evaluated : {args.samples}")
    print(f"  Output directory  : {args.output_dir}/")
    print()
    print("  Key files:")
    print("    metrics_summary.csv          → import into Excel / R")
    print("    table_main_results.tex       → paste into LaTeX Results chapter")
    print("    table_ablation.tex           → paste into Ablation Study section")
    print("    qualitative_examples.md      → thesis appendix")
    print("    plots/fig1_response_quality  → Fig in Results chapter")
    print("    plots/fig3_radar_chart       → Fig in Comparison chapter")
    print("    plots/fig5_confusion_matrix  → Fig in Emotion section")
    print()
    print("  Tip: run with --samples 200 for more statistically robust results.")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    main()
