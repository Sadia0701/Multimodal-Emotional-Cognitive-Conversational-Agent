"""
=============================================================================
data_loader.py  —  ESConv Dataset Loader
=============================================================================
Loads the ESConv (Emotional Support Conversation) dataset from HuggingFace.
Falls back to a carefully crafted synthetic mental-health dataset if the
network download fails, so evaluation always runs.

ESConv paper: Liu et al., ACL-Findings 2021
HF hub: thu-coai/esconv
=============================================================================
"""

import random
from typing import List, Dict


# ── Canonical emotion normalisation map ──────────────────────────────────────
EMOTION_NORM = {
    "anxiety":      "fear",
    "anxious":      "fear",
    "nervousness":  "fear",
    "worried":      "fear",
    "scared":       "fear",
    "fear":         "fear",
    "stress":       "fear",
    "worry":        "fear",
    "panic":        "fear",

    "depression":   "sad",
    "depressed":    "sad",
    "sadness":      "sad",
    "sad":          "sad",
    "grief":        "sad",
    "hopeless":     "sad",
    "loneliness":   "sad",
    "lonely":       "sad",
    "guilt":        "sad",
    "shame":        "sad",
    "overwhelmed":  "sad",

    "anger":        "angry",
    "angry":        "angry",
    "frustrated":   "angry",
    "frustration":  "angry",
    "irritated":    "angry",

    "happiness":    "happy",
    "happy":        "happy",
    "joy":          "happy",
    "excited":      "happy",
    "grateful":     "happy",

    "disgust":      "disgust",
    "surprise":     "surprise",
    "neutral":      "neutral",
}


# ── Synthetic fallback ────────────────────────────────────────────────────────
_SYNTHETIC = [
    # fear / anxiety cluster
    {
        "user_text": "I've been feeling really anxious about my job interview tomorrow. I can't stop overthinking everything.",
        "reference_response": "It's completely natural to feel anxious before an important interview. Take a deep breath with me — let's talk about what's worrying you most so we can work through it together.",
        "emotion_label": "anxiety", "emotion_normalized": "fear",
        "problem_type": "job crisis", "strategy": "Providing Reassurance"
    },
    {
        "user_text": "I've been having panic attacks again and I don't know what triggers them anymore.",
        "reference_response": "Panic attacks can be frightening and exhausting. I'm really glad you're talking about it. Can you tell me more about when they usually happen?",
        "emotion_label": "anxiety", "emotion_normalized": "fear",
        "problem_type": "mental health", "strategy": "Asking for Information"
    },
    {
        "user_text": "I'm scared I'm going to fail my exams and disappoint my parents. I can't sleep because of it.",
        "reference_response": "I understand that fear — the pressure you're carrying is very real, and it shows how much you care. Let's talk about what's making you feel most unprepared.",
        "emotion_label": "fear", "emotion_normalized": "fear",
        "problem_type": "academic", "strategy": "Providing Reassurance"
    },
    {
        "user_text": "I'm terrified about my health results coming back. I keep imagining the worst.",
        "reference_response": "Waiting for health news is one of the hardest things to sit with. Your feelings make complete sense. I'm here with you through this.",
        "emotion_label": "anxiety", "emotion_normalized": "fear",
        "problem_type": "health", "strategy": "Emotional Validation"
    },
    {
        "user_text": "I feel like I'm failing at everything — work, relationships, just life in general.",
        "reference_response": "That feeling of failing everywhere at once is crushing. But I want to gently push back — can you tell me what's actually been happening? I suspect you're being very hard on yourself.",
        "emotion_label": "anxiety", "emotion_normalized": "fear",
        "problem_type": "life stress", "strategy": "Reframing"
    },

    # sad / depression cluster
    {
        "user_text": "I just feel so alone and nobody understands what I'm going through. It's like nobody cares.",
        "reference_response": "I hear you, and I want you to know that your feelings are completely valid. You don't have to go through this alone — I'm here, and I care.",
        "emotion_label": "depression", "emotion_normalized": "sad",
        "problem_type": "relationship", "strategy": "Emotional Validation"
    },
    {
        "user_text": "My mom passed away last month and I still can't accept it. I keep expecting her to call.",
        "reference_response": "I'm so deeply sorry for your loss. Grief takes its own time, and it's okay to still feel her presence. She was clearly someone very precious to you.",
        "emotion_label": "grief", "emotion_normalized": "sad",
        "problem_type": "bereavement", "strategy": "Emotional Validation"
    },
    {
        "user_text": "I feel like I'm a burden to everyone around me. Maybe they'd be better off without me.",
        "reference_response": "What you're feeling right now is painful, and I want you to know you are not a burden. You matter deeply. Can we talk more about what's brought you to this place?",
        "emotion_label": "depression", "emotion_normalized": "sad",
        "problem_type": "mental health", "strategy": "Providing Emotional Support"
    },
    {
        "user_text": "I've been isolating myself from friends for months. I just don't see the point anymore.",
        "reference_response": "I'm really glad you're reaching out right now. That feeling of disconnection is incredibly hard. Can you share what started making you pull away from people?",
        "emotion_label": "depression", "emotion_normalized": "sad",
        "problem_type": "social isolation", "strategy": "Asking for Information"
    },
    {
        "user_text": "Everything is just too overwhelming — work, family, finances. I can't handle it all.",
        "reference_response": "That's a lot to carry at once, and feeling overwhelmed by it all is completely understandable. Let's take this one step at a time — what feels most urgent to you right now?",
        "emotion_label": "stress", "emotion_normalized": "sad",
        "problem_type": "life stress", "strategy": "Providing Structure"
    },
    {
        "user_text": "I cry for no reason sometimes. I don't even know why I feel this way.",
        "reference_response": "You don't need a reason for your feelings to be valid. Sometimes our emotions carry things our mind hasn't fully processed yet. Can you tell me more about when this started?",
        "emotion_label": "depression", "emotion_normalized": "sad",
        "problem_type": "mental health", "strategy": "Normalisation"
    },
    {
        "user_text": "I haven't left my house in two weeks. I just can't face people.",
        "reference_response": "Two weeks is a long time to be carrying that alone. That takes real courage to say out loud. What does 'facing people' feel like when you imagine it?",
        "emotion_label": "depression", "emotion_normalized": "sad",
        "problem_type": "agoraphobia", "strategy": "Asking for Information"
    },

    # angry / frustrated cluster
    {
        "user_text": "I'm so angry at my boss for taking credit for my work in front of everyone.",
        "reference_response": "That sounds incredibly frustrating and unfair. It's completely understandable to feel angry when your hard work isn't properly acknowledged. What would you like to do about it?",
        "emotion_label": "anger", "emotion_normalized": "angry",
        "problem_type": "workplace", "strategy": "Validation"
    },
    {
        "user_text": "My partner never listens to me. We keep having the same fight over and over.",
        "reference_response": "That cycle of repeated conflict is exhausting. Feeling unheard by someone you love is deeply painful. Can you tell me what usually starts the argument?",
        "emotion_label": "frustrated", "emotion_normalized": "angry",
        "problem_type": "relationship", "strategy": "Asking for Information"
    },
    {
        "user_text": "I'm furious at myself for missing that opportunity. I'll never get another chance like that.",
        "reference_response": "It's painful to feel you've let yourself down. That anger shows how much it mattered to you. But I want to challenge the idea that this was your only chance — can we explore that together?",
        "emotion_label": "anger", "emotion_normalized": "angry",
        "problem_type": "career", "strategy": "Reframing"
    },

    # happy / positive cluster
    {
        "user_text": "I finally got promoted after years of hard work! I can't believe it's real.",
        "reference_response": "That's absolutely wonderful news! All your dedication and perseverance have paid off. You should be incredibly proud of yourself — this is a real milestone.",
        "emotion_label": "happiness", "emotion_normalized": "happy",
        "problem_type": "career growth", "strategy": "Affirmation"
    },
    {
        "user_text": "My therapist said I've made incredible progress this year. I actually feel hopeful for the first time.",
        "reference_response": "That is such beautiful news to hear. The work you've put in to get here is real and meaningful. That feeling of hope you have — hold onto it, you've earned it.",
        "emotion_label": "happiness", "emotion_normalized": "happy",
        "problem_type": "mental health recovery", "strategy": "Affirmation"
    },
    {
        "user_text": "I finally finished my thesis after three years. I submitted it this morning!",
        "reference_response": "Congratulations — that is a massive achievement! Three years of sustained effort and dedication, and now it's done. How are you feeling right now?",
        "emotion_label": "excited", "emotion_normalized": "happy",
        "problem_type": "academic", "strategy": "Affirmation"
    },

    # mixed / nuanced
    {
        "user_text": "I don't know what's wrong with me. Some days I'm fine, other days I can barely get out of bed.",
        "reference_response": "That inconsistency can be really confusing and unsettling. You're not broken — many people experience this. Can you tell me what the hard days tend to look like?",
        "emotion_label": "depression", "emotion_normalized": "sad",
        "problem_type": "mental health", "strategy": "Normalisation"
    },
    {
        "user_text": "I've been drinking more than I should to cope with the stress. I know it's not healthy.",
        "reference_response": "I really appreciate you trusting me with that. Recognising it yourself is already an important step. What kind of stress has been pushing you to reach for it?",
        "emotion_label": "anxiety", "emotion_normalized": "fear",
        "problem_type": "substance use", "strategy": "Asking for Information"
    },
    {
        "user_text": "Nobody at school knows I'm struggling. I act like everything is fine all the time.",
        "reference_response": "Carrying that mask every day is exhausting work. I'm glad you can take it off here. What would it feel like if someone at school actually knew?",
        "emotion_label": "loneliness", "emotion_normalized": "sad",
        "problem_type": "social isolation", "strategy": "Self-Disclosure Facilitation"
    },
]


# ── Loader class ──────────────────────────────────────────────────────────────
class ESConvLoader:
    """
    Loads ESConv from HuggingFace (thu-coai/esconv).
    If unavailable, returns a hand-crafted synthetic dataset so
    the evaluation pipeline always produces valid results.
    """

    def load(self, max_samples: int = 100, split: str = "test") -> List[Dict]:
        try:
            samples = self._load_hf(max_samples, split)
            if samples:
                print(f"  ✓ Loaded {len(samples)} ESConv samples from HuggingFace")
                return samples
        except Exception as e:
            print(f"  ⚠ ESConv HuggingFace load failed: {e}")

        print("  → Using built-in synthetic mental-health dataset")
        return self._make_synthetic(max_samples)

    # ── HuggingFace path ──────────────────────────────────────────────────────
    def _load_hf(self, max_samples: int, split: str) -> List[Dict]:
        from datasets import load_dataset          # lazy import
        ds = load_dataset("thu-coai/esconv", split=split)
        samples: List[Dict] = []

        for item in ds:
            if len(samples) >= max_samples:
                break

            dialog       = item.get("dialog", [])
            top_emotion  = item.get("emotion_type", "neutral").lower()
            problem_type = item.get("problem_type", "general")

            # Extract first user→system pair from each conversation
            for i in range(len(dialog) - 1):
                u = dialog[i]
                s = dialog[i + 1]

                if u.get("speaker") != "usr" or s.get("speaker") != "sys":
                    continue

                usr_text = u.get("text", "").strip()
                ref_text = s.get("text", "").strip()
                emotion  = (u.get("emotion_type") or top_emotion).lower()

                if usr_text and ref_text and len(usr_text) > 8:
                    samples.append({
                        "user_text":          usr_text,
                        "reference_response": ref_text,
                        "emotion_label":      emotion,
                        "emotion_normalized": EMOTION_NORM.get(emotion, "neutral"),
                        "problem_type":       problem_type,
                        "situation":          item.get("situation", ""),
                        "strategy":           s.get("strategy", ""),
                    })
                    break   # one sample per conversation

        return samples

    # ── Synthetic path ────────────────────────────────────────────────────────
    def _make_synthetic(self, max_samples: int) -> List[Dict]:
        base = _SYNTHETIC.copy()
        random.shuffle(base)
        result = []
        while len(result) < max_samples:
            result.extend(base)
        return result[:max_samples]
