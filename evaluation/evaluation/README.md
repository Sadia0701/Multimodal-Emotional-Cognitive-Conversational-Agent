# Thesis Evaluation Framework
## Multimodal Emotional & Cognitive Conversational Agent

---

## Overview

This evaluation framework validates the thesis contribution by running systematic
experiments across three conditions on the **ESConv** (Emotional Support Conversation)
dataset — a standard benchmark for mental-health dialogue research.

```
Liu et al. (2021). "Towards Emotional Support Dialog Systems."
ACL-Findings 2021. HuggingFace: thu-coai/esconv
```

---

## Experimental Design

| Condition | Description | Purpose |
|-----------|-------------|---------|
| **A — Proposed** | Full cognitive pipeline (EmotionalModule → ProductionRules → GPT) | Main system |
| **B — Ablation** | Direct GPT-4o, no cognitive processing (`use_cognition=False`) | Ablation study |
| **C — Baseline** | Vanilla GPT-4o with generic system prompt | External baseline |

The comparison **A vs B** is the core ablation: it proves that the cognitive layer
adds measurable value. The comparison **A vs C** situates the system against
a commercially available off-the-shelf baseline.

---

## Metrics

### Response Quality
| Metric | What it measures |
|--------|-----------------|
| BLEU-1/2 | N-gram overlap with reference responses (Papineni et al., 2002) |
| ROUGE-1/2/L | Recall-oriented overlap (Lin, 2004) |
| BERTScore (opt.) | Semantic similarity via contextual embeddings (Zhang et al., 2019) |

### Diversity
| Metric | What it measures |
|--------|-----------------|
| DIST-1 | Ratio of unique unigrams — response variety |
| DIST-2 | Ratio of unique bigrams — phrase-level variety |

### Emotion Understanding
| Metric | What it measures |
|--------|-----------------|
| Emotion Accuracy | % correct emotion classification |
| Emotion F1 (Weighted) | F1 accounting for class imbalance |
| Emotion F1 (Macro) | Equal-weight F1 across all emotion classes |

### Empathy & Alignment
| Metric | What it measures |
|--------|-----------------|
| Empathy Score | Presence of empathetic lexical markers (Sharma et al., 2020) |
| Strategy Alignment | Response alignment with cognitive action selection |

### Statistical Tests
- Paired t-test (parametric)
- Wilcoxon signed-rank test (non-parametric)
- Significance threshold: p < 0.05

---

## Installation

```bash
cd evaluation/
pip install -r requirements_eval.txt

# Download NLTK data (one time)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## Running the Evaluation

```bash
# Standard run (100 samples, recommended for thesis)
python run_eval.py --api-key sk-YOUR_KEY --samples 100

# Faster run for testing (20 samples)
python run_eval.py --api-key sk-YOUR_KEY --samples 20

# Full 200-sample run for publication-level robustness
python run_eval.py --api-key sk-YOUR_KEY --samples 200 --delay 1.0

# Text-only (no plots, useful on headless servers)
python run_eval.py --api-key sk-YOUR_KEY --samples 100 --skip-plots
```

**Estimated cost** (gpt-4o-mini, 100 samples):  
≈ 300 API calls × ~400 tokens ≈ **$0.07 USD**

---

## Output Files

```
evaluation_results/
├── metrics_summary.csv           ← Import into Excel / R / SPSS
├── detailed_results.json         ← Full per-sample results
├── significance_tests.json       ← Statistical test raw data
│
├── table_main_results.tex        ← PASTE into thesis Results chapter
├── table_ablation.tex            ← PASTE into Ablation Study section
├── table_significance.tex        ← PASTE into Statistics section
│
├── summary.md                    ← Markdown results overview
├── qualitative_examples.md       ← Paste into Appendix
│
└── plots/
    ├── fig1_response_quality.png ← BLEU & ROUGE bar chart
    ├── fig2_diversity_empathy.png← DIST & Empathy bar chart
    ├── fig3_radar_chart.png      ← Radar/spider overall comparison
    ├── fig4_emotion_performance  ← Emotion Acc & F1 grouped bars
    ├── fig5_confusion_matrix.png ← Emotion confusion matrix
    ├── fig6_action_distribution  ← Cognitive action frequency
    ├── fig7_latency_boxplot.png  ← Latency distribution
    ├── fig8_per_sample_rougel.png← Per-sample ROUGE-L comparison
    └── fig9_strategy_alignment   ← Strategy alignment bar
```

---

## Thesis Integration Guide

### Results Chapter

1. Copy `table_main_results.tex` → paste into `results.tex`
2. Copy `table_ablation.tex` → paste into ablation section
3. Copy `table_significance.tex` → paste into statistical analysis
4. Include `fig1`, `fig3`, `fig5` as main chapter figures

### Ablation Study Section

Write:
> "Table X presents the ablation study results. Removing the cognitive layer
> (Condition B) results in a ROUGE-L decrease of {Δ}% and an empathy score
> decrease of {Δ}%, confirming the cognitive layer's contribution to
> emotionally-aligned response generation (p < 0.05, paired t-test)."

### Baseline Comparison Section

Write:
> "Compared to the vanilla GPT-4o baseline (Condition C), which has no emotional
> awareness, the proposed system achieves {X}% higher empathy score and {Y}%
> improvement in emotion F1, demonstrating the benefit of the multimodal
> cognitive architecture."

---

## Architecture Diagram (for thesis)

```
Multimodal Input
  │
  ├─ Speech (Whisper STT) ─────────────────────┐
  ├─ Facial Expression (DeepFace) ─────────────┤
  └─ Text ─────────────────────────────────────┤
                                               ▼
                               ┌──────────────────────────┐
                               │   Multimodal Fusion      │
                               └──────────┬───────────────┘
                                          │
                               ┌──────────▼───────────────┐
                               │   COGNITIVE LAYER        │
                               │  ┌──────────────────┐   │
                               │  │ Working Memory    │   │
                               │  ├──────────────────┤   │
                               │  │ Emotional Module  │   │
                               │  │ (VAD dimensions)  │   │
                               │  ├──────────────────┤   │
                               │  │ Goal Module       │   │
                               │  ├──────────────────┤   │
                               │  │ Production Rules  │   │
                               │  │ (Action select.)  │   │
                               │  ├──────────────────┤   │
                               │  │ Utility Module    │   │
                               │  └──────────────────┘   │
                               └──────────┬───────────────┘
                                          │ (action, tone, style, VAD)
                               ┌──────────▼───────────────┐
                               │   GPT-4o Reasoning       │
                               └──────────┬───────────────┘
                                          │
                               ┌──────────▼───────────────┐
                               │   Piper TTS              │
                               └──────────┬───────────────┘
                                          │
                               ┌──────────▼───────────────┐
                               │   3D Avatar Response     │
                               │   (gesture + emotion +   │
                               │    speech playback)      │
                               └──────────────────────────┘
```

---

## Citation Suggestions for Thesis

For the cognitive architecture inspiration:
- Laird et al. (2012) — Soar: A Unified Theory of Cognition
- Anderson et al. (2004) — An Integrated Theory of the Mind (ACT-R)
- Russell & Barrett (1999) — Core Affect (VAD model)

For the evaluation metrics:
- Papineni et al. (2002) — BLEU
- Lin (2004) — ROUGE
- Zhang et al. (2019) — BERTScore
- Li et al. (2016) — Distinct-N (DIST)
- Sharma et al. (2020) — Empathetic dialogue evaluation

For the dataset:
- Liu et al. (2021) — ESConv (Emotional Support Conversations), ACL-Findings
