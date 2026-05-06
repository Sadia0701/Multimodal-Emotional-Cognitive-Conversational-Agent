"""
=============================================================================
metrics.py  —  Evaluation Metrics for Thesis
=============================================================================
Implements every metric used in the evaluation chapter:

  Response Quality : BLEU-1/2, ROUGE-1/2/L, BERTScore (optional)
  Fluency/Coherence: Perplexity — GPT-2 (primary) / N-gram LM (fallback)
  Diversity        : DIST-1, DIST-2
  Emotion          : Accuracy, Weighted-F1, per-class F1, Confusion Matrix
  Empathy          : Rule-based EmpScore (Sharma et al., 2020)
  Efficiency       : Avg latency, response length

  NOTE: Strategy-Alignment removed — ceiling effect (all systems = 1.000),
        not informative as a differentiator across conditions.

Statistical Tests  : paired t-test, Wilcoxon signed-rank test
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
from scipy import stats


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
    # PERPLEXITY  (Fluency / Coherence)
    # =========================================================================

    def perplexity(
        self,
        texts:      List[str],
        references: Optional[List[str]] = None,
    ) -> float:
        """
        Compute mean perplexity over a list of texts.

        Strategy
        --------
        PRIMARY   — GPT-2 Small via HuggingFace Transformers.
                    Lower PPL = more fluent / coherent response.
                    Well-established in dialogue evaluation literature
                    (Adiwardana et al., 2020; Roller et al., 2021).

        FALLBACK  — Self-contained bigram language model with
                    Laplace (add-1) smoothing, trained on the reference
                    responses passed as `references`.
                    This is equivalent in spirit: measures how well
                    the generated text fits the in-domain distribution.

        Returns
        -------
        float : mean perplexity (lower = better fluency/coherence)
        """
        ppl = self._gpt2_perplexity(texts)
        if ppl is not None:
            return ppl

        # Fallback: n-gram LM on references
        if references:
            return self._ngram_perplexity(texts, references)

        return float("nan")

    def per_sample_perplexity(
        self,
        texts:      List[str],
        references: Optional[List[str]] = None,
    ) -> List[float]:
        """Per-sample perplexity scores (for significance testing)."""
        scores = self._gpt2_perplexity_list(texts)
        if scores is not None:
            return scores
        if references:
            return self._ngram_perplexity_list(texts, references)
        return [float("nan")] * len(texts)

    # ── GPT-2 implementation ─────────────────────────────────────────────────
    def _gpt2_perplexity(self, texts: List[str]) -> Optional[float]:
        scores = self._gpt2_perplexity_list(texts)
        return float(np.mean(scores)) if scores is not None else None

    def _gpt2_perplexity_list(self, texts: List[str]) -> Optional[List[float]]:
        """
        Compute per-sample GPT-2 perplexity.
        Returns None if transformers / model not available.
        """
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            # Lazy-load — model is cached after first call
            if not hasattr(self, "_gpt2_model"):
                print("  Loading GPT-2 for perplexity scoring...")
                self._gpt2_tok   = AutoTokenizer.from_pretrained("gpt2")
                self._gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
                self._gpt2_model.eval()
                print("  GPT-2 loaded.")

            tok, model = self._gpt2_tok, self._gpt2_model
            scores = []

            with torch.no_grad():
                for text in texts:
                    if not text.strip():
                        scores.append(float("inf"))
                        continue
                    enc = tok(
                        text,
                        return_tensors  = "pt",
                        truncation      = True,
                        max_length      = 512,
                    )
                    ids = enc.input_ids
                    if ids.shape[1] < 2:
                        scores.append(float("inf"))
                        continue
                    out = model(ids, labels=ids)
                    scores.append(float(torch.exp(out.loss)))

            return scores

        except Exception as e:
            # Silently fall through to n-gram fallback
            if "connect" not in str(e).lower() and "forbidden" not in str(e).lower():
                print(f"  ⚠ GPT-2 unavailable ({type(e).__name__}), using n-gram fallback")
            return None

    # ── N-gram LM fallback ───────────────────────────────────────────────────
    def _ngram_perplexity(
        self, texts: List[str], references: List[str]
    ) -> float:
        scores = self._ngram_perplexity_list(texts, references)
        finite = [s for s in scores if np.isfinite(s)]
        return float(np.mean(finite)) if finite else float("nan")

    def _ngram_perplexity_list(
        self, texts: List[str], references: List[str]
    ) -> List[float]:
        """
        Bigram LM with Laplace smoothing trained on `references`.
        PPL = exp( -1/N * sum(log P(w_i | w_{i-1})) )
        """
        # ── Build bigram LM from references ──────────────────────────────────
        unigram: Dict[str, int] = Counter()
        bigram:  Dict[tuple, int] = Counter()

        for ref in references:
            toks = ["<s>"] + word_tokenize(ref) + ["</s>"]
            for tok in toks:
                unigram[tok] += 1
            for a, b in zip(toks, toks[1:]):
                bigram[(a, b)] += 1

        vocab_size = len(unigram)
        total_uni  = sum(unigram.values())

        def log_prob_bigram(prev: str, curr: str) -> float:
            # Laplace (add-1) smoothed bigram probability
            num = bigram.get((prev, curr), 0) + 1
            den = unigram.get(prev, 0) + vocab_size
            return np.log(num / den)

        # ── Score each hypothesis ─────────────────────────────────────────────
        scores = []
        for text in texts:
            if not text.strip():
                scores.append(float("inf"))
                continue
            toks = ["<s>"] + word_tokenize(text) + ["</s>"]
            if len(toks) < 2:
                scores.append(float("inf"))
                continue
            log_sum = sum(
                log_prob_bigram(toks[i], toks[i + 1])
                for i in range(len(toks) - 1)
            )
            n   = len(toks) - 1
            ppl = np.exp(-log_sum / n)
            scores.append(float(ppl))

        return scores

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
    # STATISTICAL SIGNIFICANCE
    # =========================================================================

    @staticmethod
    def paired_ttest(
        scores_a: List[float], scores_b: List[float]
    ) -> Tuple[float, float]:
        """Paired t-test. Returns (t_stat, p_value)."""
        t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
        return float(t_stat), float(p_val)

    @staticmethod
    def wilcoxon_test(
        scores_a: List[float], scores_b: List[float]
    ) -> Tuple[float, float]:
        """Wilcoxon signed-rank test (non-parametric). Returns (stat, p_value)."""
        try:
            stat, p_val = stats.wilcoxon(scores_a, scores_b)
            return float(stat), float(p_val)
        except Exception:
            return 0.0, 1.0

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
        Perplexity is added as the fluency/coherence metric.
        """
        metrics: Dict[str, float] = {}

        # Response quality
        metrics["BLEU-1"]  = self.bleu_n(references, hypotheses, 1)
        metrics["BLEU-2"]  = self.bleu_n(references, hypotheses, 2)
        metrics.update(self.rouge_scores(references, hypotheses))

        # Fluency / Coherence — PRIMARY metric addition
        ppl = self.perplexity(hypotheses, references)
        metrics["Perplexity"] = ppl

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
