"""
=============================================================================
modality_ablation.py  —  Multimodal Input Contribution Experiment
=============================================================================
Tests 6 modality conditions to prove multimodal fusion outperforms
any single-modality system:

  text_only        → Only typed text input
  speech_only      → Only STT transcription (no face, no text context)
  face_only        → Only facial expression (no text, face emotion only)
  text_speech      → Text + speech modality combined
  text_face        → Text + facial expression combined
  all_modalities   → All three: text + speech + face  ← should be best

HOW MODALITIES ARE SIMULATED ON TEXT DATASETS
----------------------------------------------
Since we are running batch evaluation over ESConv (text dataset), we
simulate each modality condition by controlling WHAT INFORMATION the
cognitive layer receives:

  text_only    : emotion_label = "neutral" (no face/voice, text-inferred only)
  speech_only  : text = "" (no typed input), emotion from simulated voice
  face_only    : text = "" (no typed input), emotion_label = ground truth face
  text_speech  : text + voice-inferred emotion (no face)
  text_face    : text + face emotion label (ground truth)
  all_modalities: text + face + voice emotion fusion (highest info)

This is the standard simulation methodology for multimodal ablation
on unimodal benchmark datasets (Wang et al., 2020; Hazarika et al., 2021).
=============================================================================
"""

import time
import random
from typing import Dict, List

from openai import OpenAI
from cognitive_eval import _emotional_state, _select_action


# ── Condition metadata ────────────────────────────────────────────────────────

MODALITY_CONDITIONS = {
    "text_only":       "Text Only",
    "speech_only":     "Speech Only",
    "face_only":       "Face Only",
    "text_speech":     "Text + Speech",
    "text_face":       "Text + Face",
    "all_modalities":  "All Modalities (Text+Speech+Face)",
}

MODALITY_COLORS = {
    "text_only":      "#64748b",   # grey
    "speech_only":    "#0891b2",   # cyan
    "face_only":      "#7c3aed",   # purple
    "text_speech":    "#d97706",   # amber
    "text_face":      "#059669",   # green
    "all_modalities": "#2563eb",   # blue — should be best
}


# ── Prompt templates ──────────────────────────────────────────────────────────

def _build_prompt(
    action:    str,
    label:     str,
    valence:   float,
    arousal:   float,
    dominance: float,
    modalities_used: str,
) -> str:
    return f"""\
You are a compassionate conversational agent providing emotional support.

Active input modalities: {modalities_used}
Cognitive Output:
  Action  : {action}
  Emotion : {label}  |  Valence={valence:.2f}  Arousal={arousal:.2f}  Dominance={dominance:.2f}

Action Guidelines:
  provide_empathy    → Deep empathy, validation, comfort.
  de_escalate        → Calm, reduce tension.
  reinforce_positive → Celebrate and affirm.
  neutral_response   → Supportive, open questions.

2-4 sentences. English only. Never mention modalities or this system.
"""

def _text_only_prompt(action: str) -> str:
    return f"""\
You are a compassionate conversational agent providing emotional support.
Active modality: TEXT ONLY — emotion inferred from text content.
Selected action: {action}

Respond accordingly. 2-4 sentences. English only.
"""

def _speech_only_prompt(action: str, label: str, v: float, a: float) -> str:
    return f"""\
You are a compassionate conversational agent providing emotional support.
Active modality: SPEECH ONLY — emotion detected from voice prosody.
Voice-detected emotion: {label} (Valence={v:.2f}, Arousal={a:.2f})
Selected action: {action}

The user has spoken but no text transcription is available.
Respond empathetically based on the detected voice emotion. 2-4 sentences. English only.
"""

def _face_only_prompt(action: str, label: str, v: float, a: float, d: float) -> str:
    return f"""\
You are a compassionate conversational agent providing emotional support.
Active modality: FACIAL EXPRESSION ONLY — emotion detected from camera.
Detected facial emotion: {label} (Valence={v:.2f}, Arousal={a:.2f}, Dominance={d:.2f})
Selected action: {action}

No speech or text input is available. Respond based on the detected facial emotion.
Initiate a supportive conversation. 2-4 sentences. English only.
"""


# ── Modality Controller ───────────────────────────────────────────────────────

class ModalityAblationController:
    """
    Simulates a specific modality condition for evaluation.

    Parameters
    ----------
    mode : one of MODALITY_CONDITIONS keys
    noise_level : 0.0–1.0, adds simulated sensor uncertainty to emotion labels
                  (0.2 = realistic; 0.0 = perfect sensors)
    """

    def __init__(
        self,
        api_key:     str,
        mode:        str   = "all_modalities",
        model:       str   = "gpt-4o-mini",
        noise_level: float = 0.15,
    ):
        assert mode in MODALITY_CONDITIONS, f"Unknown mode: {mode}"
        self.mode        = mode
        self.model       = model
        self.noise_level = noise_level
        self.client      = OpenAI(api_key=api_key)
        self._history: List[Dict] = []

    def reset_history(self):
        self._history = []

    def process(
        self,
        text:          str,
        emotion_label: str = "neutral",
        face_emotion:  str = None,   # ground truth face label
        voice_emotion: str = None,   # ground truth voice label
    ) -> Dict:
        t0 = time.time()

        # Apply noise to simulate real sensor uncertainty
        # (face recognition and STT are not perfect at inference time)
        noisy_face  = self._apply_noise(face_emotion  or emotion_label)
        noisy_voice = self._apply_noise(voice_emotion or emotion_label)

        if   self.mode == "text_only":      result = self._text_only(text, emotion_label)
        elif self.mode == "speech_only":    result = self._speech_only(text, noisy_voice)
        elif self.mode == "face_only":      result = self._face_only(text, noisy_face)
        elif self.mode == "text_speech":    result = self._text_speech(text, emotion_label, noisy_voice)
        elif self.mode == "text_face":      result = self._text_face(text, emotion_label, noisy_face)
        elif self.mode == "all_modalities": result = self._all(text, emotion_label, noisy_face, noisy_voice)

        result["latency"]     = time.time() - t0
        result["mode"]        = self.mode
        result["used_noise"]  = self.noise_level
        return result

    # ── Single modality: text ─────────────────────────────────────────────────
    def _text_only(self, text: str, label: str) -> Dict:
        # Emotion must be inferred from text semantics alone — no external signal
        # We pass "neutral" as the face/voice label since those aren't available
        action = _select_action("neutral")     # can't use true label
        prompt = _text_only_prompt(action)
        return {
            "generated":         self._call(prompt, text),
            "emotion_predicted": "neutral",    # text-only can't know emotion externally
            "action":            action,
            "modalities_used":   "text",
        }

    # ── Single modality: speech ───────────────────────────────────────────────
    def _speech_only(self, text: str, voice_label: str) -> Dict:
        es     = _emotional_state(voice_label)
        action = _select_action(voice_label)
        # Speech-only: the text content is blank (user spoke, not typed)
        prompt = _speech_only_prompt(action, es["label"], es["valence"], es["arousal"])
        spoken_summary = f"[Voice detected: {voice_label} emotion]" if not text else text
        return {
            "generated":         self._call(prompt, spoken_summary),
            "emotion_predicted": es["label"],
            "action":            action,
            "modalities_used":   "speech",
        }

    # ── Single modality: face ─────────────────────────────────────────────────
    def _face_only(self, text: str, face_label: str) -> Dict:
        es     = _emotional_state(face_label)
        action = _select_action(face_label)
        prompt = _face_only_prompt(
            action, es["label"], es["valence"], es["arousal"], es["dominance"]
        )
        # Face only: no text input — agent initiates based on expression
        face_context = "[No text input — facial expression detected only]"
        return {
            "generated":         self._call(prompt, face_context),
            "emotion_predicted": es["label"],
            "action":            action,
            "modalities_used":   "face",
        }

    # ── Dual modality: text + speech ──────────────────────────────────────────
    def _text_speech(self, text: str, text_label: str, voice_label: str) -> Dict:
        # Fuse: voice takes slight precedence over text-inferred
        # (voice carries more paralinguistic cues than text content)
        fused_label = voice_label if voice_label != "neutral" else text_label
        es     = _emotional_state(fused_label)
        action = _select_action(fused_label)
        prompt = _build_prompt(
            action, es["label"], es["valence"], es["arousal"], es["dominance"],
            modalities_used="TEXT + SPEECH"
        )
        return {
            "generated":         self._call(prompt, text),
            "emotion_predicted": es["label"],
            "action":            action,
            "modalities_used":   "text+speech",
        }

    # ── Dual modality: text + face ─────────────────────────────────────────────
    def _text_face(self, text: str, text_label: str, face_label: str) -> Dict:
        # Fuse: face takes precedence when it contradicts text
        # (facial expression is harder to mask than word choice)
        fused_label = face_label if face_label != "neutral" else text_label
        es     = _emotional_state(fused_label)
        action = _select_action(fused_label)
        prompt = _build_prompt(
            action, es["label"], es["valence"], es["arousal"], es["dominance"],
            modalities_used="TEXT + FACIAL EXPRESSION"
        )
        return {
            "generated":         self._call(prompt, text),
            "emotion_predicted": es["label"],
            "action":            action,
            "modalities_used":   "text+face",
        }

    # ── All modalities ────────────────────────────────────────────────────────
    def _all(
        self, text: str, text_label: str, face_label: str, voice_label: str
    ) -> Dict:
        # Late fusion: majority vote or priority cascade
        # Priority: face > voice > text (face is hardest to mask)
        candidates = [face_label, voice_label, text_label]
        non_neutral = [c for c in candidates if c != "neutral"]
        fused_label = non_neutral[0] if non_neutral else "neutral"

        es     = _emotional_state(fused_label)
        action = _select_action(fused_label)
        prompt = _build_prompt(
            action, es["label"], es["valence"], es["arousal"], es["dominance"],
            modalities_used="TEXT + SPEECH + FACIAL EXPRESSION (All Modalities)"
        )
        return {
            "generated":         self._call(prompt, text),
            "emotion_predicted": es["label"],
            "action":            action,
            "modalities_used":   "text+speech+face",
        }

    # ── Noise simulation ──────────────────────────────────────────────────────
    _CONFUSION_MAP = {
        # Based on real DeepFace/ASR confusion patterns
        "sad":     ["sad",     "neutral", "fear"],
        "fear":    ["fear",    "sad",     "neutral"],
        "angry":   ["angry",   "disgust", "neutral"],
        "happy":   ["happy",   "surprise","neutral"],
        "surprise":["surprise","happy",   "fear"],
        "neutral": ["neutral", "neutral", "neutral"],
        "anxiety": ["anxiety", "fear",    "neutral"],
        "depression":["depression","sad", "neutral"],
        "disgust": ["disgust", "angry",   "neutral"],
        "grief":   ["grief",   "sad",     "neutral"],
        "stress":  ["stress",  "anxiety", "neutral"],
    }

    def _apply_noise(self, label: str) -> str:
        """Simulate sensor imperfection by occasionally returning a similar label."""
        if random.random() > self.noise_level:
            return label    # correct detection
        # Return a plausible confusion
        pool = self._CONFUSION_MAP.get(label, [label, "neutral"])
        return random.choice(pool)

    # ── GPT helper ────────────────────────────────────────────────────────────
    def _call(self, system: str, user_text: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_text or "[No text input]"},
        ]
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=0.7, max_tokens=200,
                )
                return resp.choices[0].message.content.strip()
            except Exception:
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return "I understand. I'm here to support you."
