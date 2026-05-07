"""
=============================================================================
master_runner.py  —  Complete Thesis Evaluation Orchestrator
=============================================================================
Runs all 4 experiments in sequence:

  EXP 1  — Original 3-way comparison  (proposed / ablation / vanilla GPT)
  EXP 2  — Component ablation         (5 cognitive module conditions)
  EXP 3  — Modality ablation          (6 input modality conditions)
  EXP 4  — Multi-dataset evaluation   (ESConv + IEMOCAP + MELD)

Then generates:
  • All plots (20+ figures)
  • All LaTeX tables
  • Master comparison with published baselines
  • Complete thesis results JSON

Usage:
  python master_runner.py --api-key sk-... --samples 100 --fast
=============================================================================
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List
from tqdm import tqdm

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from data_loader       import ESConvLoader
from metrics           import MetricsCalculator
from cognitive_eval    import CognitiveControllerEval, VanillaGPTBaseline
from component_ablation import ComponentAblationController, COMPONENT_CONDITIONS, COMPONENT_COLORS
from modality_ablation  import ModalityAblationController, MODALITY_CONDITIONS, MODALITY_COLORS
from multimodal_datasets import load_all_datasets
from published_baselines import (
    get_esconv_baselines, get_iemocap_baselines, get_meld_baselines,
    format_latex_comparison_table,
)


# ── Output directories ────────────────────────────────────────────────────────
OUT       = "evaluation_results"
PLOTS_DIR = os.path.join(OUT, "plots")
for d in [OUT, PLOTS_DIR, f"{OUT}/samples"]:
    os.makedirs(d, exist_ok=True)


# ── Shared metrics calculator (singleton) ────────────────────────────────────
MC = MetricsCalculator()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str):
    print("\n" + "═" * 68)
    print(f"  {title}")
    print("═" * 68)


def run_condition(
    label:     str,
    agent_fn,               # callable(text, emotion_label) → dict
    dataset:   List[Dict],
    delay:     float = 0.6,
) -> Dict:
    """Run one evaluation condition over a dataset. Returns full results dict."""
    generated:     List[str]   = []
    references:    List[str]   = []
    true_emotions: List[str]   = []
    pred_emotions: List[str]   = []
    actions:       List[str]   = []
    latencies:     List[float] = []

    for sample in tqdm(dataset, desc=f"    {label[:38]}"):
        result = agent_fn(
            sample["user_text"],
            sample["emotion_label"],
        )
        generated.append(result.get("generated", ""))
        references.append(sample["reference_response"])
        true_emotions.append(sample["emotion_normalized"])
        pred_emotions.append(result.get("emotion_predicted", "neutral"))
        actions.append(result.get("action", "respond"))
        latencies.append(result.get("latency", 0.0))
        time.sleep(delay)

    metrics = MC.compute_all(
        references=references, hypotheses=generated,
        true_emotions=true_emotions, pred_emotions=pred_emotions,
        actions=actions, latencies=latencies,
    )

    return {
        "label":         label,
        "metrics":       metrics,
        "generated":     generated,
        "references":    references,
        "true_emotions": true_emotions,
        "pred_emotions": pred_emotions,
        "actions":       actions,
        "latencies":     latencies,
        "per_rouge_l":   MC.per_sample_rouge_l(references, generated),
        "per_bleu1":     MC.per_sample_bleu1(references,   generated),
    }


def run_emotion_only(
    label:    str,
    agent_fn,
    dataset:  List[Dict],
    delay:    float = 0.3,
) -> Dict:
    """
    Emotion-classification-only pass for IEMOCAP/MELD.
    No reference responses — just predict emotion from text.
    """
    true_emotions: List[str] = []
    pred_emotions: List[str] = []
    latencies:     List[float] = []

    for sample in tqdm(dataset, desc=f"    {label[:38]}"):
        t0     = time.time()
        result = agent_fn(sample["text"], sample["emotion"])
        true_emotions.append(sample["emotion"])
        pred_emotions.append(result.get("emotion_predicted", "neutral"))
        latencies.append(time.time() - t0)
        time.sleep(delay)

    emo_m = MC.emotion_metrics(true_emotions, pred_emotions)
    return {
        "label":         label,
        "metrics":       {**emo_m, "Avg-Latency": MC.avg_latency(latencies)},
        "true_emotions": true_emotions,
        "pred_emotions": pred_emotions,
    }


def save_json(data, filename: str):
    path = os.path.join(OUT, filename)
    # Convert numpy types for JSON serialisation
    def _clean(obj):
        if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_clean(v) for v in obj]
        if isinstance(obj, float) and np.isnan(obj): return None
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.integer,)):  return int(obj)
        return obj
    with open(path, "w") as f:
        json.dump(_clean(data), f, indent=2)
    print(f"  ✓ Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Original 3-way comparison (reuse existing or re-run)
# ══════════════════════════════════════════════════════════════════════════════

def exp1_original(api_key: str, dataset: List[Dict], delay: float, model: str) -> Dict:
    section("EXP 1 — Original Comparison  (Proposed / Ablation / Vanilla GPT)")

    agents = {
        "with_cognition": CognitiveControllerEval(api_key, use_cognition=True,  model=model),
        "no_cognition":   CognitiveControllerEval(api_key, use_cognition=False, model=model),
        "vanilla_gpt":    VanillaGPTBaseline(api_key, model=model),
    }
    labels = {
        "with_cognition": "Proposed (w/ Cognitive Layer)",
        "no_cognition":   "Ablation (w/o Cognitive Layer)",
        "vanilla_gpt":    "Vanilla GPT-4o",
    }

    results = {}
    for key, agent in agents.items():
        print(f"\n  [{key}]")
        results[key] = run_condition(labels[key], agent.process, dataset, delay)

    save_json({k: {"label": v["label"], "metrics": v["metrics"]} for k, v in results.items()},
              "exp1_original.json")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Component Ablation
# ══════════════════════════════════════════════════════════════════════════════

def exp2_component(api_key: str, dataset: List[Dict], delay: float, model: str) -> Dict:
    section("EXP 2 — Component Ablation  (5 Conditions)")

    results = {}
    for mode, label in COMPONENT_CONDITIONS.items():
        print(f"\n  [{mode}] {label}")
        agent = ComponentAblationController(api_key, mode=mode, model=model)
        results[mode] = run_condition(label, agent.process, dataset, delay)

    save_json({k: {"label": v["label"], "metrics": v["metrics"]} for k, v in results.items()},
              "exp2_component_ablation.json")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Modality Ablation
# ══════════════════════════════════════════════════════════════════════════════

def exp3_modality(api_key: str, dataset: List[Dict], delay: float, model: str) -> Dict:
    section("EXP 3 — Modality Ablation  (6 Input Conditions)")

    results = {}
    for mode, label in MODALITY_CONDITIONS.items():
        print(f"\n  [{mode}] {label}")
        agent = ModalityAblationController(api_key, mode=mode, model=model, noise_level=0.15)

        generated:     List[str]   = []
        references:    List[str]   = []
        true_emotions: List[str]   = []
        pred_emotions: List[str]   = []
        actions:       List[str]   = []
        latencies:     List[float] = []

        for sample in tqdm(dataset, desc=f"    {label[:38]}"):
            result = agent.process(
                text          = sample["user_text"],
                emotion_label = sample["emotion_label"],
                face_emotion  = sample["emotion_label"],    # ground truth face
                voice_emotion = sample["emotion_label"],    # ground truth voice
            )
            generated.append(result.get("generated", ""))
            references.append(sample["reference_response"])
            true_emotions.append(sample["emotion_normalized"])
            pred_emotions.append(result.get("emotion_predicted", "neutral"))
            actions.append(result.get("action", "respond"))
            latencies.append(result.get("latency", 0.0))
            time.sleep(delay)

        metrics = MC.compute_all(
            references=references, hypotheses=generated,
            true_emotions=true_emotions, pred_emotions=pred_emotions,
            actions=actions, latencies=latencies,
        )

        results[mode] = {
            "label": label, "metrics": metrics,
            "generated": generated, "references": references,
            "true_emotions": true_emotions, "pred_emotions": pred_emotions,
            "per_rouge_l": MC.per_sample_rouge_l(references, generated),
        }

    save_json({k: {"label": v["label"], "metrics": v["metrics"]} for k, v in results.items()},
              "exp3_modality_ablation.json")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4 — Multi-Dataset Emotion Classification
# ══════════════════════════════════════════════════════════════════════════════

def exp4_multidataset(api_key: str, max_per_ds: int, delay: float, model: str) -> Dict:
    section("EXP 4 — Multi-Dataset Evaluation  (IEMOCAP + MELD)")

    datasets    = load_all_datasets(max_per_ds)
    agent_cog   = CognitiveControllerEval(api_key, use_cognition=True,  model=model)
    agent_nocog = CognitiveControllerEval(api_key, use_cognition=False, model=model)
    agent_van   = VanillaGPTBaseline(api_key, model=model)

    results = {}
    for ds_name, ds_data in datasets.items():
        print(f"\n  Dataset: {ds_name.upper()}  ({len(ds_data)} samples)")
        results[ds_name] = {}

        for key, label, agent in [
            ("with_cognition", "Proposed",      agent_cog),
            ("no_cognition",   "No Cognition",  agent_nocog),
            ("vanilla_gpt",    "Vanilla GPT",   agent_van),
        ]:
            print(f"    [{label}]")
            results[ds_name][key] = run_emotion_only(
                label, agent.process, ds_data, delay * 0.5
            )

    save_json(
        {ds: {k: {"label": v["label"], "metrics": v["metrics"]}
               for k, v in conds.items()}
          for ds, conds in results.items()},
        "exp4_multidataset.json"
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS COMPILATION & REPORT
# ══════════════════════════════════════════════════════════════════════════════

def compile_master_csv(
    exp1: Dict, exp2: Dict, exp3: Dict, exp4: Dict
) -> pd.DataFrame:
    """Build one master CSV with all results for import into Excel / R."""
    rows = []

    for key, data in exp1.items():
        rows.append({"Experiment": "EXP1-Original", "Condition": data["label"],
                     **data["metrics"]})
    for key, data in exp2.items():
        rows.append({"Experiment": "EXP2-Component", "Condition": data["label"],
                     **data["metrics"]})
    for key, data in exp3.items():
        rows.append({"Experiment": "EXP3-Modality", "Condition": data["label"],
                     **data["metrics"]})
    for ds, conds in exp4.items():
        for key, data in conds.items():
            rows.append({"Experiment": f"EXP4-{ds.upper()}", "Condition": data["label"],
                         **data["metrics"]})

    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "master_results.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ Master CSV → {path}")
    return df


def generate_latex_tables(exp1: Dict, exp2: Dict, exp3: Dict, exp4: Dict):
    """Write all LaTeX tables used in the thesis."""
    lines_all = [
        "% ================================================================",
        "% AUTO-GENERATED — thesis_tables.tex",
        "% Copy-paste individual tables into your thesis chapters",
        "% ================================================================",
        "",
    ]

    # ── Table 1: Main comparison (EXP 1) ─────────────────────────────────────
    metrics = ["BLEU-1","BLEU-2","ROUGE-L","DIST-2",
               "Emotion-Acc","Emotion-F1-W","Empathy-Score"]
    lines_all += _latex_table(
        exp1, metrics,
        caption="Main Evaluation Results on ESConv (Proposed vs Baselines)",
        label="tab:main"
    )

    # ── Table 2: Component ablation (EXP 2) ───────────────────────────────────
    lines_all += _latex_table(
        exp2, metrics,
        caption="Component Ablation Study — Effect of Each Cognitive Module",
        label="tab:component_ablation"
    )

    # ── Table 3: Modality ablation (EXP 3) ────────────────────────────────────
    mod_metrics = ["BLEU-1","ROUGE-L","Emotion-Acc","Empathy-Score"]
    lines_all += _latex_table(
        exp3, mod_metrics,
        caption="Modality Ablation — Contribution of Each Input Modality",
        label="tab:modality_ablation"
    )

    # ── Table 4: Multi-dataset comparison (EXP 4) ────────────────────────────
    emo_metrics = ["Emotion-Acc","Emotion-F1-W","Emotion-F1-M"]
    for ds, conds in exp4.items():
        lines_all += _latex_table(
            conds, emo_metrics,
            caption=f"Emotion Recognition on {ds.upper()} — Comparison with Published Baselines",
            label=f"tab:{ds}_results"
        )

    # ── Table 5: Published baseline comparison ────────────────────────────────
    lines_all.append("\n% -- Published Baseline Comparison (fill in your actual numbers) --\n")
    our_esconv = list(exp1.values())[0]["metrics"]   # proposed system
    lines_all.append(format_latex_comparison_table(
        get_esconv_baselines(), our_esconv, "esconv",
        ["BLEU-1","ROUGE-L","DIST-2","Empathy-Score"],
    ))

    path = os.path.join(OUT, "thesis_tables.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines_all))
    print(f"  ✓ LaTeX tables → {path}")


def _latex_table(
    results: Dict, metrics: List[str], caption: str, label: str
) -> List[str]:
    """Generic LaTeX table generator."""
    keys = list(results.keys())
    n    = len(keys)

    DISPLAY = {
        "BLEU-1":"BLEU-1","BLEU-2":"BLEU-2","ROUGE-L":"ROUGE-L",
        "Perplexity":"PPL$\\downarrow$","DIST-2":"DIST-2",
        "Emotion-Acc":"Acc.","Emotion-F1-W":"F1-W","Emotion-F1-M":"F1-M",
        "Empathy-Score":"Empathy",
    }
    HIGHER_BETTER = {"BLEU-1","BLEU-2","ROUGE-L","DIST-2",
                     "Emotion-Acc","Emotion-F1-W","Emotion-F1-M","Empathy-Score"}

    def best(m):
        vals = [results[k]["metrics"].get(m) for k in keys]
        vals = [v for v in vals if v is not None and not (isinstance(v,float) and np.isnan(v))]
        if not vals: return None
        return min(vals) if m not in HIGHER_BETTER else max(vals)

    def fmt(v, m):
        if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
        s  = f"{v:.3f}"
        bv = best(m)
        return f"\\textbf{{{s}}}" if bv is not None and abs(v-bv)<1e-6 else s

    header = " & ".join(f"\\textbf{{{DISPLAY.get(m,m)}}}" for m in metrics)
    lines  = [
        "", r"\begin{table}[ht]", r"\centering", r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{l" + "c"*len(metrics) + "}",
        r"\toprule",
        f"\\textbf{{Condition}} & {header} \\\\",
        r"\midrule",
    ]

    for k in keys:
        row_m = results[k]["metrics"]
        cols  = " & ".join(fmt(row_m.get(m), m) for m in metrics)
        name  = results[k]["label"][:36]
        lines.append(f"{name} & {cols} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Master Thesis Evaluation Runner")
    parser.add_argument("--api-key",  required=True)
    parser.add_argument("--samples",  type=int,   default=100,
                        help="Samples per condition per experiment (default 100)")
    parser.add_argument("--delay",    type=float, default=0.6,
                        help="Seconds between API calls (default 0.6)")
    parser.add_argument("--model",    default="gpt-4o-mini")
    parser.add_argument("--fast",     action="store_true",
                        help="Use 30 samples — quick smoke test")
    parser.add_argument("--exp",      type=str, default="all",
                        help="Which experiments: all | 1 | 2 | 3 | 4 | 1,2,3")
    args = parser.parse_args()

    if args.fast:
        args.samples = 30
        args.delay   = 0.3

    exps_to_run = set(
        ["1","2","3","4"] if args.exp == "all"
        else args.exp.split(",")
    )

    section("MULTIMODAL COGNITIVE AGENT — COMPLETE THESIS EVALUATION")
    print(f"  Samples     : {args.samples}")
    print(f"  Model       : {args.model}")
    print(f"  Delay       : {args.delay}s")
    print(f"  Experiments : {', '.join(sorted(exps_to_run))}")

    # ── Load base dataset (ESConv / synthetic) ────────────────────────────────
    print("\nLoading ESConv dataset...")
    loader  = ESConvLoader()
    dataset = loader.load(args.samples)
    print(f"  → {len(dataset)} samples")

    # ── Run experiments ───────────────────────────────────────────────────────
    exp1 = exp2 = exp3 = exp4 = {}

    if "1" in exps_to_run:
        exp1 = exp1_original(args.api_key, dataset, args.delay, args.model)

    if "2" in exps_to_run:
        exp2 = exp2_component(args.api_key, dataset, args.delay, args.model)

    if "3" in exps_to_run:
        exp3 = exp3_modality(args.api_key, dataset, args.delay, args.model)

    if "4" in exps_to_run:
        exp4 = exp4_multidataset(
            args.api_key,
            max_per_ds = max(args.samples // 2, 50),
            delay      = args.delay,
            model      = args.model,
        )

    # ── Compile results ───────────────────────────────────────────────────────
    section("Compiling Results & Generating Outputs")

    if any([exp1, exp2, exp3, exp4]):
        compile_master_csv(exp1, exp2, exp3, exp4)

    if any([exp1, exp2, exp3, exp4]):
        generate_latex_tables(exp1, exp2, exp3, exp4)

    # ── Generate all plots ────────────────────────────────────────────────────
    try:
        from extended_visualizer import ExtendedVisualizer
        viz = ExtendedVisualizer(
            exp1=exp1, exp2=exp2, exp3=exp3, exp4=exp4,
            output_dir=OUT,
        )
        viz.generate_all()
    except Exception as e:
        print(f"  ⚠ Plot generation failed: {e}")

    section("EVALUATION COMPLETE")
    print(f"""
  Output files:
    master_results.csv          → All results in one spreadsheet
    thesis_tables.tex           → All LaTeX tables (paste into thesis)
    exp1_original.json          → EXP 1 detailed data
    exp2_component_ablation.json→ EXP 2 detailed data
    exp3_modality_ablation.json → EXP 3 detailed data
    exp4_multidataset.json      → EXP 4 detailed data
    plots/                      → All figures

  Estimated cost @ gpt-4o-mini ({args.samples} samples):
    EXP 1: ~{args.samples*3*400//1000}K tokens   ≈ ${args.samples*3*400*0.00015/1000:.3f}
    EXP 2: ~{args.samples*5*400//1000}K tokens   ≈ ${args.samples*5*400*0.00015/1000:.3f}
    EXP 3: ~{args.samples*6*200//1000}K tokens   ≈ ${args.samples*6*200*0.00015/1000:.3f}
    EXP 4: ~{args.samples*3*100//1000}K tokens   ≈ ${args.samples*3*100*0.00015/1000:.3f}
    ─────────────────────────────────────────────────────────
    Total:                              ≈ ${(args.samples*(3+5+6+3)*350)*0.00015/1000:.2f}
""")


if __name__ == "__main__":
    main()