# ==========================================================
# baseline_no_cognition.py
# Baseline system WITHOUT cognitive layer
# ==========================================================

import sys
import os
sys.path.append(os.path.abspath("../backend"))

import json
import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# ==========================================================
# SIMPLE BASELINE MODEL
# ==========================================================

class BaselineAgent:
    """
    Simple baseline:
    - No cognitive reasoning
    - No emotion modeling
    - Simple generic response generator
    """

    def generate(self, user_text):

        # Very simple template-based + generic responses
        # (acts like weak chatbot baseline)

        if "sad" in user_text.lower() or "anxious" in user_text.lower():
            return "I'm sorry to hear that. Would you like to talk more about it?"

        if "job" in user_text.lower():
            return "That sounds like a difficult situation. What do you think you can do next?"

        return "I understand. Can you tell me more about how you feel?"


agent = BaselineAgent()

# ==========================================================
# METRICS
# ==========================================================

smooth = SmoothingFunction().method1

rouge = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=True
)


def bleu_score(reference, candidate):
    try:
        return sentence_bleu(
            [reference.split()],
            candidate.split(),
            smoothing_function=smooth
        )
    except:
        return 0.0


def rouge_scores(reference, candidate):
    try:
        scores = rouge.score(reference, candidate)
        return (
            scores["rouge1"].fmeasure,
            scores["rouge2"].fmeasure,
            scores["rougeL"].fmeasure
        )
    except:
        return (0, 0, 0)


def distinct_n(text, n=1):
    tokens = text.split()
    if len(tokens) < n:
        return 0.0

    grams = set()
    for i in range(len(tokens) - n + 1):
        grams.add(tuple(tokens[i:i+n]))

    return len(grams) / len(tokens)


# ==========================================================
# LOAD ESConv
# ==========================================================

def load_esconv(path):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []

    for item in data:

        dialog = item.get("dialog", [])

        for i in range(len(dialog) - 1):

            try:
                if dialog[i]["speaker"] == "seeker" and \
                   dialog[i+1]["speaker"] == "supporter":

                    user_text = dialog[i]["content"].strip()
                    gold_reply = dialog[i+1]["content"].strip()

                    if len(user_text) > 3 and len(gold_reply) > 3:
                        samples.append({
                            "input": user_text,
                            "target": gold_reply
                        })

            except:
                continue

    print(f"✅ Extracted {len(samples)} samples")

    return samples


# ==========================================================
# MAIN EVALUATION
# ==========================================================

DATASET_PATH = "datasets/esconv.json"
OUTPUT_CSV = "results/baseline_results.csv"
NUM_SAMPLES = 200


def evaluate():

    dataset = load_esconv(DATASET_PATH)

    dataset = dataset[:NUM_SAMPLES]

    if len(dataset) == 0:
        print("❌ Dataset parsing failed")
        return

    rows = []

    print(f"Loaded {len(dataset)} samples")

    for sample in tqdm(dataset):

        user_input = sample["input"]
        gold = sample["target"]

        start = time.time()

        pred = agent.generate(user_input)

        latency = time.time() - start

        bleu = bleu_score(gold, pred)
        r1, r2, rL = rouge_scores(gold, pred)

        d1 = distinct_n(pred, 1)
        d2 = distinct_n(pred, 2)

        rows.append({
            "input": user_input,
            "reference": gold,
            "prediction": pred,
            "bleu": bleu,
            "rouge1": r1,
            "rouge2": r2,
            "rougeL": rL,
            "distinct1": d1,
            "distinct2": d2,
            "latency_sec": latency
        })

    df = pd.DataFrame(rows)

    Path("results").mkdir(exist_ok=True)

    df.to_csv(OUTPUT_CSV, index=False)

    print("\n==============================")
    print("BASELINE RESULTS")
    print("==============================")

    print("BLEU:", round(df["bleu"].mean(), 4))
    print("ROUGE-L:", round(df["rougeL"].mean(), 4))
    print("Distinct-1:", round(df["distinct1"].mean(), 4))
    print("Distinct-2:", round(df["distinct2"].mean(), 4))
    print("Latency:", round(df["latency_sec"].mean(), 3))

    print("\nSaved:", OUTPUT_CSV)


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    evaluate()