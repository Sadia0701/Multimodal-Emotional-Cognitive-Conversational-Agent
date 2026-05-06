"""
=============================================================================
cognitive_eval.py  —  Text-Only Evaluation Adapter
=============================================================================
Recreates the full cognitive pipeline (EmotionalModule → ProductionRules →
GPT response generation) but stripped of all I/O dependencies (Whisper,
Piper TTS, WebSocket, etc.) so it can run cleanly over a dataset.

Two modes, matching the thesis ablation design:
  use_cognition=True   → Full cognitive layer (the proposed system)
  use_cognition=False  → Baseline mode (direct GPT, no cognitive processing)
=============================================================================
"""

import time
from typing import Dict, Optional

from openai import OpenAI


# ── Emotion Module (matches emotional_module.py) ─────────────────────────────
_EMOTION_DIMS = {
    "happy":      ( 0.8,  0.6,  0.7),
    "sad":        (-0.7,  0.4,  0.3),
    "angry":      (-0.6,  0.8,  0.6),
    "fear":       (-0.8,  0.7,  0.4),
    "surprise":   ( 0.4,  0.8,  0.6),
    "neutral":    ( 0.0,  0.2,  0.5),
    # Extended for ESConv domain
    "anxiety":    (-0.65, 0.75, 0.3),
    "depression": (-0.8,  0.3,  0.2),
    "disgust":    (-0.5,  0.6,  0.5),
    "grief":      (-0.75, 0.35, 0.25),
    "stress":     (-0.55, 0.70, 0.35),
    "loneliness": (-0.65, 0.3,  0.25),
    "guilt":      (-0.6,  0.4,  0.3),
}


def _emotional_state(label: str) -> Dict:
    label = label.lower()
    v, a, d = _EMOTION_DIMS.get(label, (0.0, 0.0, 0.0))
    return {"label": label, "valence": v, "arousal": a, "dominance": d}


# ── Production Rules (matches production_rules.py) ────────────────────────────
def _select_action(label: str) -> str:
    label = label.lower()
    if label in {"sad", "fear", "depression", "anxiety", "grief",
                 "loneliness", "guilt", "stress"}:
        return "provide_empathy"
    elif label in {"angry", "disgust"}:
        return "de_escalate"
    elif label in {"happy", "surprise"}:
        return "reinforce_positive"
    else:
        return "neutral_response"


# ── System Prompts ────────────────────────────────────────────────────────────
_COGNITIVE_SYSTEM_PROMPT = """\
You are a compassionate conversational agent providing emotional and psychological support.

The cognitive architecture has analysed the user's emotional state and selected the response action.
You MUST follow this action strictly — do not override it.

Cognitive Output:
  Action  : {action}
  Emotion : {label}  |  Valence={valence:.2f}  Arousal={arousal:.2f}  Dominance={dominance:.2f}

Action Guidelines:
  provide_empathy      → Deep empathetic acknowledgement, validation, comfort. Show you truly understand.
  de_escalate          → Calm and steady. Reduce tension. Acknowledge frustration without amplifying it.
  reinforce_positive   → Warmly celebrate, encourage, affirm the user's positive state.
  validate_and_redirect→ Acknowledge feelings, then gently guide toward constructive thinking.
  neutral_response     → Engage naturally, ask open questions, be supportive but balanced.

Style: warm, human, conversational. 2-4 sentences. English only.
Never mention the cognitive system or these instructions.
"""

_BASELINE_SYSTEM_PROMPT = """\
You are a helpful and supportive assistant. Respond naturally and helpfully to the user.
Keep your response conversational and concise (2-4 sentences). English only.
"""

_VANILLA_SYSTEM_PROMPT = """\
You are a general-purpose AI assistant. Answer the user's message helpfully.
"""


# ── Cognitive Controller (Eval version) ──────────────────────────────────────
class CognitiveControllerEval:
    """
    Evaluation-only wrapper around the cognitive pipeline + GPT.

    Parameters
    ----------
    api_key        : OpenAI API key
    use_cognition  : True  → full cognitive pipeline (proposed system)
                     False → baseline mode (direct GPT, no cognitive layer)
    model          : GPT model string
    history_turns  : Number of recent turns kept in context
    """

    def __init__(
        self,
        api_key:       str,
        use_cognition: bool = True,
        model:         str  = "gpt-4o-mini",
        history_turns: int  = 3,
    ):
        self.use_cognition  = use_cognition
        self.model          = model
        self.history_turns  = history_turns
        self.client         = OpenAI(api_key=api_key)
        self._history       = []   # list of {user, assistant}

    def reset_history(self):
        self._history = []

    def process(self, text: str, emotion_label: str = "neutral") -> Dict:
        """
        Run one turn through the pipeline.

        Returns a dict with:
          generated        : str   — generated response
          emotion_predicted: str   — emotion label used
          action           : str   — selected cognitive action
          valence/arousal/dominance: float — VAD dimensions (cognition only)
          condition        : str   — "with_cognition" | "no_cognition"
          latency          : float — wall-clock seconds
        """
        t0 = time.time()

        if self.use_cognition:
            result = self._cognitive_pass(text, emotion_label)
        else:
            result = self._baseline_pass(text, emotion_label)

        result["latency"] = time.time() - t0
        return result

    # ── Full Cognitive Pipeline ───────────────────────────────────────────────
    def _cognitive_pass(self, text: str, emotion_label: str) -> Dict:
        es     = _emotional_state(emotion_label)
        action = _select_action(emotion_label)

        system = _COGNITIVE_SYSTEM_PROMPT.format(
            action    = action,
            label     = es["label"],
            valence   = es["valence"],
            arousal   = es["arousal"],
            dominance = es["dominance"],
        )

        response = self._call_gpt(system, text)

        return {
            "generated":          response,
            "emotion_predicted":  es["label"],
            "action":             action,
            "valence":            es["valence"],
            "arousal":            es["arousal"],
            "dominance":          es["dominance"],
            "condition":          "with_cognition",
        }

    # ── Baseline (No Cognition) ───────────────────────────────────────────────
    def _baseline_pass(self, text: str, emotion_label: str) -> Dict:
        response = self._call_gpt(_BASELINE_SYSTEM_PROMPT, text)
        return {
            "generated":         response,
            "emotion_predicted": emotion_label,   # pass-through (no processing)
            "action":            "respond",
            "valence":           0.0,
            "arousal":           0.0,
            "dominance":         0.0,
            "condition":         "no_cognition",
        }

    # ── GPT Call (with retry + history context) ───────────────────────────────
    def _call_gpt(
        self, system_prompt: str, user_text: str, retries: int = 3
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]

        # Inject recent conversation history
        for turn in self._history[-self.history_turns:]:
            messages.append({"role": "user",      "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        messages.append({"role": "user", "content": user_text})

        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model       = self.model,
                    messages    = messages,
                    temperature = 0.7,
                    max_tokens  = 220,
                )
                text = resp.choices[0].message.content.strip()
                self._history.append({"user": user_text, "assistant": text})
                return text

            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  ⚠ GPT call failed after {retries} attempts: {e}")

        fallback = "I understand. I'm here to listen — please tell me more about how you're feeling."
        self._history.append({"user": user_text, "assistant": fallback})
        return fallback


# ── Vanilla GPT Baseline (no emotional awareness whatsoever) ─────────────────
class VanillaGPTBaseline:
    """
    Pure GPT-4o with a generic system prompt.
    Represents a commercially available off-the-shelf chatbot.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model  = model

    def process(self, text: str, emotion_label: str = "neutral") -> Dict:
        t0 = time.time()

        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model    = self.model,
                    messages = [
                        {"role": "system",  "content": _VANILLA_SYSTEM_PROMPT},
                        {"role": "user",    "content": text},
                    ],
                    temperature = 0.7,
                    max_tokens  = 220,
                )
                generated = resp.choices[0].message.content.strip()
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    generated = "I'd be happy to help. Could you tell me more?"

        return {
            "generated":         generated,
            "emotion_predicted": "neutral",
            "action":            "respond",
            "valence":            0.0,
            "arousal":            0.0,
            "dominance":          0.0,
            "condition":         "vanilla_gpt",
            "latency":           time.time() - t0,
        }
