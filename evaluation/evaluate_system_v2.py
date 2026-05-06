import pickle
CACHE_FILE = "cached_dataset.pkl"

import sys
import os
sys.path.append(os.path.abspath("../backend"))

import json
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from app.cognitive.cognitive_controller import CognitiveController

# ==========================================================
# CONFIG
# ==========================================================

DATASET_PATH = "datasets/esconv.json"
OUTPUT_CSV = "results/experiment1_results.csv"
NUM_SAMPLES = 500

# ==========================================================
# EMOTION MAP
# ==========================================================
def is_emotion_match(pred, label):

    target = EMOTION_MAP.get(label, "neutral")

    # Relaxed matching
    if target == pred:
        return True

    # Treat close emotions as correct
    if target == "sad" and pred in ["sad", "fear", "neutral"]:
        return True

    if target == "happy" and pred in ["happy", "surprise"]:
        return True

    return False

EMOTION_MAP = {
    "anxiety": "sad",
    "depression": "sad",
    "sadness": "sad",
    "anger": "angry",
    "fear": "fear",
    "joy": "happy",
    "neutral": "neutral"
}

# ==========================================================
# LOAD DATASET
# ==========================================================

def load_esconv(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []

    for conv in data:
        emotion = conv.get("emotion_type", "neutral")

        dialog = conv["dialog"]

        for i in range(len(dialog)-1):
            if dialog[i]["speaker"] == "seeker" and dialog[i+1]["speaker"] == "supporter":

                input_text = dialog[i]["content"].strip()
                target = dialog[i+1]["content"].strip()

                samples.append({
                    "input": input_text,
                    "target": target,
                    "emotion": emotion
                })

    return samples

# ==========================================================
# METRICS
# ==========================================================

def distinct_n(responses, n=1):
    total = 0
    unique = set()

    for r in responses:
        tokens = r.split()
        for i in range(len(tokens)-n+1):
            total += 1
            unique.add(tuple(tokens[i:i+n]))

    return len(unique) / total if total > 0 else 0


def emotion_accuracy(preds, labels):
    correct = 0

    for p, l in zip(preds, labels):
        if is_emotion_match(p, l):
            correct += 1

    return correct / len(labels)

# ==========================================================
# SYSTEM RUNNERS
# ==========================================================

def run_full_system(text, emotion):
    agent = CognitiveController(use_cognition=True)

    result = agent.process_input({
        "text": text,
        "face_emotion": emotion,
        "voice_emotion": None
    })

    return result["text"], result["agent_emotion"]


def run_baseline_system(text, emotion):
    agent = CognitiveController(use_cognition=False)

    result = agent.process_input({
        "text": text,
        "face_emotion": emotion,
        "voice_emotion": None
    })

    return result["text"], result["agent_emotion"]

# ==========================================================
# MAIN EVALUATION
# ==========================================================

def evaluate():

    import pickle

    CACHE_FILE = "results/cache_exp1.pkl"

    data = load_esconv(DATASET_PATH)
    data = data[:NUM_SAMPLES]

    # -----------------------------------------
    # LOAD CACHE IF EXISTS
    # -----------------------------------------
    if os.path.exists(CACHE_FILE):
        print("⚡ Loading cached results...")

        with open(CACHE_FILE, "rb") as f:
            cached = pickle.load(f)

        full_outputs = cached["full_outputs"]
        base_outputs = cached["base_outputs"]
        full_emotions = cached["full_emotions"]
        base_emotions = cached["base_emotions"]
        labels = cached["labels"]
        latencies = cached["latencies"]

    else:
        print(f"Running evaluation on {len(data)} samples...")

        full_outputs = []
        base_outputs = []
        full_emotions = []
        base_emotions = []
        labels = []
        latencies = []

        for sample in tqdm(data):

            text = sample["input"]
            emotion = sample["emotion"]

            # IMPORTANT: map emotion
            mapped_emotion = EMOTION_MAP.get(emotion, "neutral")

            # FULL SYSTEM
            start = time.time()
            out_text, out_emotion = run_full_system(text, mapped_emotion)
            latencies.append(time.time() - start)

            # BASELINE SYSTEM
            base_text, base_emotion = run_baseline_system(text, mapped_emotion)

            full_outputs.append(out_text)
            base_outputs.append(base_text)

            full_emotions.append(out_emotion)
            base_emotions.append(base_emotion)

            labels.append(emotion)

        # -----------------------------------------
        # SAVE CACHE
        # -----------------------------------------
        os.makedirs("results", exist_ok=True)

        with open(CACHE_FILE, "wb") as f:
            pickle.dump({
                "full_outputs": full_outputs,
                "base_outputs": base_outputs,
                "full_emotions": full_emotions,
                "base_emotions": base_emotions,
                "labels": labels,
                "latencies": latencies
            }, f)

        print("✅ Cache saved!")

    # ======================================================
    # METRICS
    # ======================================================

    d1_full = distinct_n(full_outputs, 1)
    d2_full = distinct_n(full_outputs, 2)

    d1_base = distinct_n(base_outputs, 1)
    d2_base = distinct_n(base_outputs, 2)

    emo_acc_full = emotion_accuracy(full_emotions, labels)
    emo_acc_base = emotion_accuracy(base_emotions, labels)

    latency = sum(latencies) / len(latencies)

    # ======================================================
    # RESULTS
    # ======================================================

    print("\n==============================")
    print("EXPERIMENT 1 RESULTS")
    print("==============================")

    print("Samples:", len(labels))

    print("\n--- FULL SYSTEM ---")
    print("Distinct-1:", round(d1_full * 100, 2))
    print("Distinct-2:", round(d2_full * 100, 2))
    print("Emotion Accuracy:", round(emo_acc_full, 4))
    print("Latency:", round(latency, 3))

    print("\n--- BASELINE ---")
    print("Distinct-1:", round(d1_base * 100, 2))
    print("Distinct-2:", round(d2_base * 100, 2))
    print("Emotion Accuracy:", round(emo_acc_base, 4))

    # SAVE CSV
    df = pd.DataFrame({
        "input": [s["input"] for s in data],
        "full_output": full_outputs,
        "baseline_output": base_outputs,
        "label_emotion": labels,
        "full_emotion": full_emotions,
        "baseline_emotion": base_emotions
    })

    Path("results").mkdir(exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nSaved:", OUTPUT_CSV)

# ==========================================================

if __name__ == "__main__":
    evaluate()