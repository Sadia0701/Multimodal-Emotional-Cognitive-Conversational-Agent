# Evaluation Results — Multimodal Cognitive Agent

## Main Results Table

| Metric | Proposed (w/ Cognitive Layer) | Ablation (w/o Cognitive Layer) | Vanilla GPT-4o Baseline |
| --- | --- | --- | --- |
| BLEU-1 | **1.0000** | 0.1049 | 0.0112 |
| BLEU-2 | **1.0000** | 0.0380 | 0.0027 |
| ROUGE-L | **1.0000** | 0.1341 | 0.1213 |
| Perplexity | 84.3824 | **203.0974** | 169.1917 |
| DIST-1 | 0.2804 | **0.3718** | 0.0333 |
| DIST-2 | 0.5887 | **0.6031** | 0.0333 |
| Emotion-Acc | **1.0000** | 0.0000 | 0.0000 |
| Empathy-Score | **0.5333** | 0.1250 | 0.2500 |

## Cognitive Layer Gain (Proposed vs Ablation)

| Metric | Δ Absolute | Δ Relative |
| --- | --- | --- |
| BLEU-1 | +0.8951 | +853.2% |
| BLEU-2 | +0.9620 | +2531.0% |
| ROUGE-L | +0.8659 | +645.5% |
| Perplexity | -118.7150 | -58.5% |
| DIST-1 | -0.0915 | -24.6% |
| DIST-2 | -0.0144 | -2.4% |
| Emotion-Acc | +1.0000 | +1000000000000.0% |
| Empathy-Score | +0.4083 | +326.7% |