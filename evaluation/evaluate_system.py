# ==========================================================
# evaluate_system.py
# Thesis Evaluation Script for ESConv Dataset
# Multimodal Cognitive Conversational Agent
# ==========================================================
import sys
import os

sys.path.append(os.path.abspath("../backend"))

import json
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Optional semantic metric
try:
    from bert_score import score as bertscore_score
    BERT_AVAILABLE = True
except:
    BERT_AVAILABLE = False

# ----------------------------------------------------------
# IMPORT YOUR SYSTEM
# ----------------------------------------------------------

from app.cognitive.cognitive_controller import CognitiveController


# ==========================================================
# CONFIG
# ==========================================================

DATASET_PATH = "datasets/esconv.json"
OUTPUT_CSV = "results/esconv_results.csv"
NUM_SAMPLES = 200   # change to full dataset later

# ==========================================================
# LOAD SYSTEM
# ==========================================================

agent = CognitiveController()

# ==========================================================
# HELPERS
# ==========================================================

smooth = SmoothingFunction().method1

rouge = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=True
)


def bleu_score(reference, candidate):
    try:
        ref = [reference.split()]
        cand = candidate.split()
        return sentence_bleu(
            ref,
            cand,
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
        return (0.0, 0.0, 0.0)


def distinct_n(text, n=1):
    tokens = text.split()

    if len(tokens) < n:
        return 0.0

    grams = set()

    for i in range(len(tokens) - n + 1):
        grams.add(tuple(tokens[i:i+n]))

    return len(grams) / max(len(tokens), 1)


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

                    # remove empty or trivial messages
                    if len(user_text) > 3 and len(gold_reply) > 3:
                        samples.append({
                            "input": user_text,
                            "target": gold_reply
                        })

            except:
                continue

    print(f"✅ Extracted {len(samples)} valid samples")

    return samples


# ==========================================================
# RUN SYSTEM
# ==========================================================

def run_agent(user_text):

    multimodal_input = {
        "text": user_text,
        "face_emotion": "sad",     # therapy simulation
        "voice_emotion": None
    }

    start = time.time()

    result = agent.process_input(multimodal_input)

    latency = time.time() - start

    reply = result["text"]

    return reply, latency, result


# ==========================================================
# MAIN EVALUATION
# ==========================================================

def evaluate():

    dataset = load_esconv(DATASET_PATH)

    dataset = dataset[:NUM_SAMPLES]

    rows = []

    refs = []
    cands = []

    print(f"Loaded {len(dataset)} samples")

    for sample in tqdm(dataset):

        user_input = sample["input"]
        gold = sample["target"]

        pred, latency, meta = run_agent(user_input)

        # Metrics
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

            "latency_sec": latency,

            "agent_emotion": meta.get("agent_emotion", ""),
            "tone": meta.get("tone", ""),
            "gesture": meta.get("gesture", "")
        })

        refs.append(gold)
        cands.append(pred)

    # ------------------------------------------------------
    # BERTScore
    # ------------------------------------------------------

    bert_f1 = None

    if BERT_AVAILABLE:
        try:
            P, R, F1 = bertscore_score(
                cands,
                refs,
                lang="en",
                verbose=True
            )

            bert_f1 = float(F1.mean())

        except:
            bert_f1 = None

    # ------------------------------------------------------
    # SAVE CSV
    # ------------------------------------------------------

    df = pd.DataFrame(rows)

    Path("results").mkdir(exist_ok=True)

    df.to_csv(OUTPUT_CSV, index=False)

    # ------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------

    print("\n==============================")
    print("THESIS EVALUATION RESULTS")
    print("==============================")

    print("Samples:", len(df))
    print("BLEU:", round(df["bleu"].mean(), 4))
    print("ROUGE-1:", round(df["rouge1"].mean(), 4))
    print("ROUGE-2:", round(df["rouge2"].mean(), 4))
    print("ROUGE-L:", round(df["rougeL"].mean(), 4))
    print("Distinct-1:", round(df["distinct1"].mean(), 4))
    print("Distinct-2:", round(df["distinct2"].mean(), 4))
    print("Latency(sec):", round(df["latency_sec"].mean(), 3))

    if bert_f1:
        print("BERTScore F1:", round(bert_f1, 4))

    print("\nSaved:", OUTPUT_CSV)


# ==========================================================
# ENTRY
# ==========================================================

if __name__ == "__main__":
    evaluate()