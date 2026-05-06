"""
=============================================================================
runner.py  —  Main Evaluation Runner
=============================================================================
Orchestrates all three experimental conditions:

  Condition A : With Cognitive Layer      (proposed system)
  Condition B : Without Cognitive Layer   (ablation study)
  Condition C : Vanilla GPT-4o            (external baseline)

For each condition:
  • Iterates over ESConv / synthetic dataset
  • Collects generated responses + metadata
  • Computes all metrics
  • Runs paired statistical significance tests (A vs B, A vs C)
  • Saves results to JSON + CSV
=============================================================================
"""

import os
import json
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_loader   import ESConvLoader
from metrics       import MetricsCalculator
from cognitive_eval import CognitiveControllerEval, VanillaGPTBaseline


# ── Labels used in tables / plots ─────────────────────────────────────────────
CONDITION_LABELS = {
    "with_cognition":    "Proposed (w/ Cognitive Layer)",
    "no_cognition":      "Ablation (w/o Cognitive Layer)",
    "vanilla_gpt":       "Vanilla GPT-4o Baseline",
}

CONDITION_COLORS = {
    "with_cognition": "#2563EB",   # blue
    "no_cognition":   "#DC2626",   # red
    "vanilla_gpt":    "#059669",   # green
}


class EvaluationRunner:
    """
    End-to-end evaluation pipeline.

    Usage
    -----
    runner = EvaluationRunner(api_key="sk-...", max_samples=100)
    results = runner.run_all()
    """

    def __init__(
        self,
        api_key:      str,
        max_samples:  int   = 100,
        api_delay:    float = 0.6,    # seconds between GPT calls (rate limit)
        output_dir:   str   = "evaluation_results",
        model:        str   = "gpt-4o-mini",
    ):
        self.api_key     = api_key
        self.max_samples = max_samples
        self.api_delay   = api_delay
        self.output_dir  = output_dir
        self.model       = model

        os.makedirs(output_dir,               exist_ok=True)
        os.makedirs(f"{output_dir}/plots",    exist_ok=True)
        os.makedirs(f"{output_dir}/samples",  exist_ok=True)

        self.metrics = MetricsCalculator()
        self.results: Dict[str, dict] = {}

        self._banner()

    # =========================================================================
    # INITIALISATION
    # =========================================================================

    def _banner(self):
        print("\n" + "═" * 65)
        print("  MULTIMODAL COGNITIVE AGENT — THESIS EVALUATION FRAMEWORK")
        print("  Dataset: ESConv (Emotional Support Conversations)")
        print("═" * 65)

    def _load_data(self):
        print("\n[1/5] Loading Dataset ...")
        loader    = ESConvLoader()
        self.data = loader.load(self.max_samples)
        print(f"      → {len(self.data)} samples loaded\n")

    def _init_models(self):
        print("[2/5] Initialising Models ...")
        self.agent_cognitive = CognitiveControllerEval(
            self.api_key, use_cognition=True,  model=self.model
        )
        self.agent_ablation  = CognitiveControllerEval(
            self.api_key, use_cognition=False, model=self.model
        )
        self.agent_vanilla   = VanillaGPTBaseline(
            self.api_key, model=self.model
        )
        print("      → All models ready\n")

    # =========================================================================
    # PER-CONDITION RUN
    # =========================================================================

    def _run_condition(
        self,
        key:     str,
        label:   str,
        agent_fn,          # callable(text, emotion_label) → dict
    ) -> dict:
        print(f"[Running] {label}")
        print("─" * 55)

        generated:      List[str]   = []
        references:     List[str]   = []
        true_emotions:  List[str]   = []
        pred_emotions:  List[str]   = []
        actions:        List[str]   = []
        latencies:      List[float] = []

        for sample in tqdm(self.data, desc=f"  {label[:30]}"):
            result = agent_fn(
                sample["user_text"],
                sample["emotion_label"]
            )

            generated.append(result.get("generated", ""))
            references.append(sample["reference_response"])
            true_emotions.append(sample["emotion_normalized"])
            pred_emotions.append(result.get("emotion_predicted", "neutral"))
            actions.append(result.get("action", "respond"))
            latencies.append(result.get("latency", 0.0))

            time.sleep(self.api_delay)

        # ── Compute metrics ──────────────────────────────────────────────────
        agg_metrics = self.metrics.compute_all(
            references    = references,
            hypotheses    = generated,
            true_emotions = true_emotions,
            pred_emotions = pred_emotions,
            actions       = actions,
            latencies     = latencies,
        )

        # ── Per-sample scores (for significance tests) ────────────────────────
        per_rouge_l = self.metrics.per_sample_rouge_l(references, generated)
        per_bleu1   = self.metrics.per_sample_bleu1(references,  generated)
        per_ppl     = self.metrics.per_sample_perplexity(generated, references)

        print(f"  ✓ Done — ROUGE-L={agg_metrics['ROUGE-L']:.4f}  "
              f"DIST-2={agg_metrics['DIST-2']:.4f}  "
              f"Empathy={agg_metrics['Empathy-Score']:.4f}  "
              f"PPL={agg_metrics.get('Perplexity', float('nan')):.2f}\n")

        return {
            "key":           key,
            "label":         label,
            "metrics":       agg_metrics,
            "generated":     generated,
            "references":    references,
            "true_emotions": true_emotions,
            "pred_emotions": pred_emotions,
            "actions":       actions,
            "latencies":     latencies,
            "per_rouge_l":   per_rouge_l,
            "per_bleu1":     per_bleu1,
            "per_ppl":       per_ppl,
        }

    # =========================================================================
    # SIGNIFICANCE TESTS
    # =========================================================================

    def _significance_tests(self) -> dict:
        """
        Compare the proposed system against each baseline on ROUGE-L, BLEU-1,
        and Perplexity using both paired t-test and Wilcoxon signed-rank test.
        """
        tests = {}
        proposed = self.results["with_cognition"]

        for key in ["no_cognition", "vanilla_gpt"]:
            if key not in self.results:
                continue
            other = self.results[key]

            for metric_key, scores_a, scores_b in [
                ("ROUGE-L",    proposed["per_rouge_l"],   other["per_rouge_l"]),
                ("BLEU-1",     proposed["per_bleu1"],     other["per_bleu1"]),
                ("Perplexity", proposed["per_ppl"],       other["per_ppl"]),
            ]:
                # Skip if PPL not computed (all nan)
                if all(np.isnan(s) for s in scores_a):
                    continue

                # For perplexity: lower is better, so reverse the comparison
                # (proposed wins if its PPL is LOWER, meaning t_stat should be negative)
                t_stat, p_t = self.metrics.paired_ttest(scores_a, scores_b)
                w_stat, p_w = self.metrics.wilcoxon_test(scores_a, scores_b)

                tests[f"{key}_{metric_key}"] = {
                    "comparison":     f"Proposed vs {CONDITION_LABELS[key]}",
                    "metric":         metric_key,
                    "lower_is_better": metric_key == "Perplexity",
                    "t_stat":         t_stat,
                    "p_ttest":        p_t,
                    "significant_t":  p_t < 0.05,
                    "w_stat":         w_stat,
                    "p_wilcoxon":     p_w,
                    "significant_w":  p_w < 0.05,
                }

        return tests

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def run_all(self) -> dict:
        self._load_data()
        self._init_models()

        print("[3/5] Running Evaluation Conditions ...\n")

        # Condition A: with cognitive layer
        self.results["with_cognition"] = self._run_condition(
            key      = "with_cognition",
            label    = CONDITION_LABELS["with_cognition"],
            agent_fn = self.agent_cognitive.process,
        )

        # Condition B: without cognitive layer (ablation)
        self.results["no_cognition"] = self._run_condition(
            key      = "no_cognition",
            label    = CONDITION_LABELS["no_cognition"],
            agent_fn = self.agent_ablation.process,
        )

        # Condition C: vanilla GPT
        self.results["vanilla_gpt"] = self._run_condition(
            key      = "vanilla_gpt",
            label    = CONDITION_LABELS["vanilla_gpt"],
            agent_fn = self.agent_vanilla.process,
        )

        # ── Significance tests ────────────────────────────────────────────────
        print("[4/5] Running Statistical Significance Tests ...")
        self.sig_tests = self._significance_tests()

        # ── Save & report ─────────────────────────────────────────────────────
        print("\n[5/5] Saving Results ...")
        self._save_results()
        self._print_results_table()
        self._print_significance_table()

        return {
            "results":   self.results,
            "sig_tests": self.sig_tests,
        }

    # =========================================================================
    # SAVING
    # =========================================================================

    def _save_results(self):
        # ── Metrics summary CSV ───────────────────────────────────────────────
        rows = []
        for key, data in self.results.items():
            row = {"System": CONDITION_LABELS[key]}
            row.update(data["metrics"])
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(f"{self.output_dir}/metrics_summary.csv", index=False)

        # ── Detailed JSON ─────────────────────────────────────────────────────
        save = {}
        for key, data in self.results.items():
            save[key] = {
                "label":   CONDITION_LABELS[key],
                "metrics": data["metrics"],
                "samples": [
                    {
                        "idx":            i,
                        "user":           self.data[i]["user_text"],
                        "reference":      data["references"][i],
                        "generated":      data["generated"][i],
                        "true_emotion":   data["true_emotions"][i],
                        "pred_emotion":   data["pred_emotions"][i],
                        "action":         data["actions"][i],
                        "latency_s":      round(data["latencies"][i], 3),
                    }
                    for i in range(len(data["generated"]))
                ],
            }

        with open(f"{self.output_dir}/detailed_results.json", "w") as f:
            json.dump(save, f, indent=2)

        # ── Significance test JSON ────────────────────────────────────────────
        with open(f"{self.output_dir}/significance_tests.json", "w") as f:
            json.dump(self.sig_tests, f, indent=2)

        print(f"  ✓ Results saved to → {self.output_dir}/")

    # =========================================================================
    # CONSOLE PRINTING
    # =========================================================================

    def _print_results_table(self):
        metrics_order = [
            "BLEU-1", "BLEU-2", "ROUGE-1", "ROUGE-2", "ROUGE-L",
            "Perplexity",
            "DIST-1", "DIST-2", "Emotion-Acc", "Emotion-F1-W",
            "Empathy-Score", "Avg-Length", "Avg-Latency",
        ]

        print("\n" + "═" * 75)
        print("  EVALUATION RESULTS — FULL METRICS TABLE")
        print("═" * 75)

        rows = []
        for key, data in self.results.items():
            row = {"System": CONDITION_LABELS[key][:38]}
            for m in metrics_order:
                v = data["metrics"].get(m)
                row[m] = f"{v:.4f}" if v is not None else "N/A"
            rows.append(row)

        df = pd.DataFrame(rows).set_index("System")
        print(df.to_string())

        # ── Improvement over ablation ──────────────────────────────────────────
        if "with_cognition" in self.results and "no_cognition" in self.results:
            wc = self.results["with_cognition"]["metrics"]
            nc = self.results["no_cognition"]["metrics"]

            print("\n" + "─" * 75)
            print("  COGNITIVE LAYER GAIN  (Proposed − Ablation)")
            print("─" * 75)
            print(f"  {'Metric':<22} {'Δ Absolute':>12}  {'Δ Relative (%)':>16}")
            print("  " + "─" * 54)

            for m in ["BLEU-1", "BLEU-2", "ROUGE-L", "Perplexity",
                      "DIST-1", "DIST-2",
                      "Emotion-Acc", "Emotion-F1-W", "Empathy-Score"]:
                if m not in wc or m not in nc:
                    continue
                delta = wc[m] - nc[m]
                pct   = (delta / (abs(nc[m]) + 1e-10)) * 100.0
                sign  = "+" if delta >= 0 else ""
                print(f"  {m:<22} {sign}{delta:>12.4f}  {sign}{pct:>14.1f}%")

    def _print_significance_table(self):
        if not self.sig_tests:
            return

        print("\n" + "═" * 75)
        print("  STATISTICAL SIGNIFICANCE TESTS")
        print("  (Proposed System vs Baselines)")
        print("═" * 75)
        print(f"  {'Comparison':<38} {'Metric':<9} {'p (t-test)':>11} {'p (Wilcox)':>11} {'Sig?':>6}")
        print("  " + "─" * 72)

        for test in self.sig_tests.values():
            comp  = test["comparison"][:36]
            m     = test["metric"]
            pt    = test["p_ttest"]
            pw    = test["p_wilcoxon"]
            sig   = "✓" if test["significant_t"] or test["significant_w"] else "✗"
            print(f"  {comp:<38} {m:<9} {pt:>11.4f} {pw:>11.4f} {sig:>6}")

        print("\n  ✓ = p < 0.05 (statistically significant)")
