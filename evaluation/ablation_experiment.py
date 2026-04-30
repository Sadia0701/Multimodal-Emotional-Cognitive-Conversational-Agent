# ==========================================================
# ablation_experiment.py
# Compare FULL system vs BASELINE system
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
# IMPORT FULL SYSTEM
# ==========================================================

from app.cognitive.cognitive_controller import CognitiveController

full_agent = CognitiveController()

# ==========================================================
# BASELINE SYSTEM (same as before)
# ==========================================================

class BaselineAgent:

    def generate(self, user_text):

        if "sad" in user_text.lower() or "anxious" in user_text.lower():
            return "I'm sorry to hear that. Would you like to talk more about it?"

        if "job" in user_text.lower():
            return "That sounds difficult. What do you think you can do?"

        return "I understand. Can you tell me more?"

baseline_agent = BaselineAgent()

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
        return scores["rougeL"].fmeasure
    except:
        return 0.0

def distinct_n(text, n=1):
    tokens = text.split()
    if len(tokens) < n:
        return 0.0

    grams = set()
    for i in range(len(tokens) - n + 1):
        grams.add(tuple(tokens[i:i+n]))

    return len(grams) / len(tokens)

# ==========================================================
# LOAD DATASET
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
# RUN BOTH SYSTEMS
# ==========================================================

def run_full(user_text):

    start = time.time()

    result = full_agent.process_input({
        "text": user_text,
        "face_emotion": "sad",
        "voice_emotion": None
    })

    latency = time.time() - start

    return result["text"], latency

def run_baseline(user_text):

    start = time.time()

    pred = baseline_agent.generate(user_text)

    latency = time.time() - start

    return pred, latency

# ==========================================================
# MAIN EXPERIMENT
# ==========================================================

DATASET_PATH = "datasets/esconv.json"
OUTPUT_FILE = "results/ablation_results.csv"
NUM_SAMPLES = 200

def evaluate():

    dataset = load_esconv(DATASET_PATH)
    dataset = dataset[:NUM_SAMPLES]

    if len(dataset) == 0:
        print("❌ Dataset empty")
        return

    rows = []

    print(f"Running ablation on {len(dataset)} samples")

    for sample in tqdm(dataset):

        user_input = sample["input"]
        gold = sample["target"]

        # FULL SYSTEM
        full_pred, full_lat = run_full(user_input)

        # BASELINE
        base_pred, base_lat = run_baseline(user_input)

        rows.append({
            "input": user_input,
            "reference": gold,

            # FULL
            "full_pred": full_pred,
            "full_bleu": bleu_score(gold, full_pred),
            "full_rouge": rouge_scores(gold, full_pred),
            "full_d1": distinct_n(full_pred, 1),
            "full_d2": distinct_n(full_pred, 2),
            "full_latency": full_lat,

            # BASELINE
            "base_pred": base_pred,
            "base_bleu": bleu_score(gold, base_pred),
            "base_rouge": rouge_scores(gold, base_pred),
            "base_d1": distinct_n(base_pred, 1),
            "base_d2": distinct_n(base_pred, 2),
            "base_latency": base_lat,
        })

    df = pd.DataFrame(rows)

    Path("results").mkdir(exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    # ======================================================
    # SUMMARY
    # ======================================================

    print("\n==============================")
    print("ABLATION RESULTS")
    print("==============================")

    print("\n--- FULL SYSTEM ---")
    print("BLEU:", round(df["full_bleu"].mean(), 4))
    print("ROUGE:", round(df["full_rouge"].mean(), 4))
    print("Distinct-1:", round(df["full_d1"].mean(), 4))
    print("Distinct-2:", round(df["full_d2"].mean(), 4))
    print("Latency:", round(df["full_latency"].mean(), 3))

    print("\n--- BASELINE ---")
    print("BLEU:", round(df["base_bleu"].mean(), 4))
    print("ROUGE:", round(df["base_rouge"].mean(), 4))
    print("Distinct-1:", round(df["base_d1"].mean(), 4))
    print("Distinct-2:", round(df["base_d2"].mean(), 4))
    print("Latency:", round(df["base_latency"].mean(), 3))

    print("\nSaved:", OUTPUT_FILE)


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    evaluate()