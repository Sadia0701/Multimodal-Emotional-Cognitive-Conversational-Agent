"""
=============================================================================
metrics.py  —  Evaluation Metrics for Thesis
=============================================================================
Implements every metric used in the evaluation chapter:

  Response Quality : BLEU-1/2, ROUGE-1/2/L, BERTScore (optional)
  Diversity        : DIST-1, DIST-2
  Emotion          : Accuracy, Weighted-F1, per-class F1, Confusion Matrix
  Empathy          : Rule-based EmpScore (Sharma et al., 2020)
  Efficiency       : Avg latency, response length

  NOTE: Strategy-Alignment removed — ceiling effect (all systems = 1.000),
        not informative as a differentiator across conditions.
=============================================================================
"""

import numpy as np
from collections import Counter
from typing import List, Dict, Optional, Tuple

import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from rouge_score import rouge_scorer as rouge_lib
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix,
)


def word_tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer (no NLTK download required)."""
    return re.findall(r"\b\w+\b", text.lower())


# ── Empathy keyword lists (from empathy literature) ──────────────────────────
_EMPATHY_KW = [
    "understand", "feel", "sorry", "here for you", "support", "listen",
    "care", "help", "difficult", "hard", "tough", "must be", "sounds like",
    "imagine", "validate", "normal", "okay", "not alone", "together",
    "share", "tell me more", "i hear you", "completely", "sense",
    "that's tough", "that makes sense", "i can see", "i appreciate",
    "you matter", "you're not", "it's okay", "that must", "going through",
    "reach out", "thank you for sharing", "courageous", "brave",
    "so painful", "really hard", "big step", "i'm glad you",
]

class MetricsCalculator:
    """Computes all evaluation metrics used in the thesis."""

    def __init__(self):
        self._rouge  = rouge_lib.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self._smooth = SmoothingFunction().method1

    # =========================================================================
    # RESPONSE QUALITY
    # =========================================================================

    def bleu_n(self, references: List[str], hypotheses: List[str], n: int) -> float:
        """Corpus BLEU-N (sentence-level average, smoothed)."""
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        scores  = []
        for ref, hyp in zip(references, hypotheses):
            ref_tok = [word_tokenize(ref.lower())]
            hyp_tok = word_tokenize(hyp.lower())
            if not hyp_tok:
                scores.append(0.0)
                continue
            scores.append(
                sentence_bleu(ref_tok, hyp_tok,
                              weights=weights,
                              smoothing_function=self._smooth)
            )
        return float(np.mean(scores))

    def rouge_scores(
        self, references: List[str], hypotheses: List[str]
    ) -> Dict[str, float]:
        """ROUGE-1, ROUGE-2, ROUGE-L (F1)."""
        r1, r2, rl = [], [], []
        for ref, hyp in zip(references, hypotheses):
            s = self._rouge.score(ref, hyp)
            r1.append(s["rouge1"].fmeasure)
            r2.append(s["rouge2"].fmeasure)
            rl.append(s["rougeL"].fmeasure)
        return {
            "ROUGE-1": float(np.mean(r1)),
            "ROUGE-2": float(np.mean(r2)),
            "ROUGE-L": float(np.mean(rl)),
        }

    def bert_score_f1(
        self, references: List[str], hypotheses: List[str]
    ) -> Optional[float]:
        """BERTScore F1. Returns None if bert-score not installed."""
        try:
            from bert_score import score as bs
            _, _, F1 = bs(hypotheses, references, lang="en", verbose=False)
            return float(F1.mean())
        except Exception:
            return None

    # =========================================================================
    # DIVERSITY
    # =========================================================================

    def distinct_n(self, texts: List[str], n: int) -> float:
        """DIST-N: ratio of unique n-grams to total n-grams across all texts."""
        all_ngrams: List[tuple] = []
        for text in texts:
            tokens = word_tokenize(text.lower())
            ngrams = [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]
            all_ngrams.extend(ngrams)
        if not all_ngrams:
            return 0.0
        return len(set(all_ngrams)) / len(all_ngrams)

    # =========================================================================
    # EMOTION
    # =========================================================================

    def emotion_metrics(
        self,
        true_labels: List[str],
        pred_labels: List[str],
    ) -> Dict[str, float]:
        """Accuracy, weighted-F1, macro-F1."""
        t = [self._norm_emo(l) for l in true_labels]
        p = [self._norm_emo(l) for l in pred_labels]
        acc  = accuracy_score(t, p)
        wf1  = f1_score(t, p, average="weighted", zero_division=0)
        mf1  = f1_score(t, p, average="macro",    zero_division=0)
        return {"Emotion-Acc": acc, "Emotion-F1-W": wf1, "Emotion-F1-M": mf1}

    def emotion_classification_report(
        self, true_labels: List[str], pred_labels: List[str]
    ) -> str:
        t = [self._norm_emo(l) for l in true_labels]
        p = [self._norm_emo(l) for l in pred_labels]
        return classification_report(t, p, zero_division=0)

    def confusion_matrix_data(
        self, true_labels: List[str], pred_labels: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        t      = [self._norm_emo(l) for l in true_labels]
        p      = [self._norm_emo(l) for l in pred_labels]
        labels = sorted(set(t) | set(p))
        cm     = confusion_matrix(t, p, labels=labels)
        return cm, labels

    @staticmethod
    def _norm_emo(label: str) -> str:
        from data_loader import EMOTION_NORM   # avoid circular at module level
        return EMOTION_NORM.get(label.lower(), label.lower())

    # =========================================================================
    # EMPATHY
    # =========================================================================

    def empathy_score(self, texts: List[str]) -> float:
        """
        Rule-based empathy score [0,1].
        Inspired by EmpathyScore (Sharma et al., 2020) —
        counts empathetic lexical markers, normalised to [0,1].
        """
        scores = []
        for text in texts:
            tl = text.lower()
            found = sum(1 for kw in _EMPATHY_KW if kw in tl)
            scores.append(min(found / 4.0, 1.0))
        return float(np.mean(scores))

    # NOTE: strategy_alignment() removed — saturated at 1.000 across all
    # conditions (ceiling effect), providing no discriminative signal.

    # =========================================================================
    # EFFICIENCY
    # =========================================================================

    @staticmethod
    def avg_latency(latencies: List[float]) -> float:
        return float(np.mean(latencies)) if latencies else 0.0

    @staticmethod
    def avg_length(texts: List[str]) -> float:
        return float(np.mean([len(word_tokenize(t)) for t in texts]))

    # =========================================================================
    # PER-SAMPLE SCORES  (used for detailed analysis plots)
    # =========================================================================

    def per_sample_rouge_l(
        self, references: List[str], hypotheses: List[str]
    ) -> List[float]:
        return [
            self._rouge.score(r, h)["rougeL"].fmeasure
            for r, h in zip(references, hypotheses)
        ]

    def per_sample_bleu1(
        self, references: List[str], hypotheses: List[str]
    ) -> List[float]:
        scores = []
        for ref, hyp in zip(references, hypotheses):
            ref_tok = [word_tokenize(ref.lower())]
            hyp_tok = word_tokenize(hyp.lower())
            scores.append(
                sentence_bleu(
                    ref_tok, hyp_tok,
                    weights=(1, 0, 0, 0),
                    smoothing_function=self._smooth,
                ) if hyp_tok else 0.0
            )
        return scores

    # =========================================================================
    # ALL-IN-ONE
    # =========================================================================

    def compute_all(
        self,
        references:    List[str],
        hypotheses:    List[str],
        true_emotions: List[str],
        pred_emotions: List[str],
        actions:       List[str],
        latencies:     List[float],
    ) -> Dict[str, float]:
        """
        Compute the full metrics dictionary used for the results table.
        Strategy-Alignment is excluded (ceiling effect, no discriminative value).
        """
        metrics: Dict[str, float] = {}

        # Response quality
        metrics["BLEU-1"]  = self.bleu_n(references, hypotheses, 1)
        metrics["BLEU-2"]  = self.bleu_n(references, hypotheses, 2)
        metrics.update(self.rouge_scores(references, hypotheses))

        # Diversity
        metrics["DIST-1"]  = self.distinct_n(hypotheses, 1)
        metrics["DIST-2"]  = self.distinct_n(hypotheses, 2)

        # Emotion
        metrics.update(self.emotion_metrics(true_emotions, pred_emotions))

        # Empathy  (Strategy-Alignment removed — saturated at 1.000)
        metrics["Empathy-Score"] = self.empathy_score(hypotheses)

        # Efficiency
        metrics["Avg-Length"]   = self.avg_length(hypotheses)
        metrics["Avg-Latency"]  = self.avg_latency(latencies)

        # Optional BERTScore
        bert = self.bert_score_f1(references, hypotheses)
        if bert is not None:
            metrics["BERTScore"] = bert

        return metrics
