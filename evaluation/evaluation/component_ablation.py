"""
=============================================================================
component_ablation.py  —  Fine-grained Cognitive Layer Ablation
=============================================================================
Tests 5 conditions by removing one cognitive module at a time:

  full              → complete proposed system (baseline for this ablation)
  no_emotional      → EmotionalModule bypassed (VAD always neutral/zero)
  no_production     → ProductionRuleEngine bypassed (action always neutral)
  no_memory         → WorkingMemory bypassed (no conversational history)
  no_cognitive      → entire cognitive layer removed (GPT-only baseline)

This proves which module contributes the most to performance improvement,
and answers the committee question: "which part of your cognitive layer
actually matters?"

Academic framing: Component ablation study (Stuartetal., 2021 naming convention)
=============================================================================
"""

import time
from typing import Dict, List, Optional
from openai import OpenAI
from cognitive_eval import _emotional_state, _select_action


# ── System prompt templates ───────────────────────────────────────────────────

_FULL_COGNITIVE_PROMPT = """\
You are a compassionate conversational agent providing emotional and psychological support.
The cognitive architecture has analysed the user's emotional state and selected the action.
You MUST follow this action strictly.

Cognitive Output:
  Action  : {action}
  Emotion : {label}  |  Valence={valence:.2f}  Arousal={arousal:.2f}  Dominance={dominance:.2f}

Action Guidelines:
  provide_empathy    → Deep empathetic acknowledgement, validation, comfort.
  de_escalate        → Calm, steady. Reduce tension without amplifying it.
  reinforce_positive → Warmly celebrate and affirm the user's positive state.
  neutral_response   → Natural, supportive, ask open questions.

2-4 sentences. English only. Never mention this system.
"""

_NO_EMOTIONAL_PROMPT = """\
You are a compassionate conversational agent providing emotional and psychological support.
The system has selected the action: {action}

Respond according to this action. 2-4 sentences. English only.
NOTE: No emotional state dimensions are available — respond based on action label only.
"""

_NO_PRODUCTION_PROMPT = """\
You are a compassionate conversational agent providing emotional and psychological support.

The user's emotional state has been analysed:
  Emotion : {label}  |  Valence={valence:.2f}  Arousal={arousal:.2f}  Dominance={dominance:.2f}

Respond naturally and supportively based on this emotional understanding.
No specific action has been selected — use your judgement.
2-4 sentences. English only.
"""

_NO_MEMORY_PROMPT = """\
You are a compassionate conversational agent providing emotional and psychological support.

Cognitive Output:
  Action  : {action}
  Emotion : {label}  |  Valence={valence:.2f}  Arousal={arousal:.2f}  Dominance={dominance:.2f}

Respond to the CURRENT message only — do not assume any prior context.
2-4 sentences. English only.
"""

_NO_COGNITIVE_PROMPT = """\
You are a helpful and supportive assistant. Respond naturally to the user.
Keep your response conversational and concise (2-4 sentences). English only.
"""


# ── Ablation condition metadata ───────────────────────────────────────────────

COMPONENT_CONDITIONS = {
    "full":          "Full Cognitive System",
    "no_emotional":  "No Emotional Module",
    "no_production": "No Production Rules",
    "no_memory":     "No Working Memory",
    "no_cognitive":  "No Cognitive Layer",
}

COMPONENT_COLORS = {
    "full":          "#2563EB",   # blue   — the proposed system
    "no_emotional":  "#7C3AED",   # purple
    "no_production": "#D97706",   # amber
    "no_memory":     "#059669",   # green
    "no_cognitive":  "#DC2626",   # red    — worst expected
}


# ── Controller ────────────────────────────────────────────────────────────────

class ComponentAblationController:
    """
    Runs one ablation condition.

    Parameters
    ----------
    mode : one of {"full","no_emotional","no_production","no_memory","no_cognitive"}
    """

    def __init__(
        self,
        api_key:       str,
        mode:          str  = "full",
        model:         str  = "gpt-4o-mini",
        history_turns: int  = 3,
    ):
        assert mode in COMPONENT_CONDITIONS, f"Unknown mode: {mode}"
        self.mode          = mode
        self.model         = model
        self.history_turns = history_turns
        self.client        = OpenAI(api_key=api_key)
        self._history: List[Dict] = []

    def reset_history(self):
        self._history = []

    def process(self, text: str, emotion_label: str = "neutral") -> Dict:
        t0 = time.time()

        if   self.mode == "full":          result = self._full(text, emotion_label)
        elif self.mode == "no_emotional":  result = self._no_emotional(text, emotion_label)
        elif self.mode == "no_production": result = self._no_production(text, emotion_label)
        elif self.mode == "no_memory":     result = self._no_memory(text, emotion_label)
        elif self.mode == "no_cognitive":  result = self._no_cognitive(text)

        result["latency"] = time.time() - t0
        result["mode"]    = self.mode
        return result

    # ── Full system ───────────────────────────────────────────────────────────
    def _full(self, text: str, label: str) -> Dict:
        es     = _emotional_state(label)
        action = _select_action(label)
        prompt = _FULL_COGNITIVE_PROMPT.format(
            action=action, label=es["label"],
            valence=es["valence"], arousal=es["arousal"], dominance=es["dominance"]
        )
        return {
            "generated":         self._call(prompt, text, use_history=True),
            "emotion_predicted": es["label"],
            "action":            action,
            "valence":           es["valence"],
            "arousal":           es["arousal"],
            "dominance":         es["dominance"],
        }

    # ── No Emotional Module ───────────────────────────────────────────────────
    # EmotionalModule is bypassed: emotion label passes through but VAD dims
    # are all zero. The action is still selected from the label, but the LLM
    # receives no VAD information — tests the value of dimensional emotion coding.
    def _no_emotional(self, text: str, label: str) -> Dict:
        action = _select_action(label)    # rules still fire
        prompt = _NO_EMOTIONAL_PROMPT.format(action=action)
        return {
            "generated":         self._call(prompt, text, use_history=True),
            "emotion_predicted": label,
            "action":            action,
            "valence":           0.0, "arousal": 0.0, "dominance": 0.0,
        }

    # ── No Production Rules ───────────────────────────────────────────────────
    # Production rules bypassed: action always "neutral_response".
    # VAD dimensions are computed but action selection is removed.
    # Tests value of rule-based action routing.
    def _no_production(self, text: str, label: str) -> Dict:
        es     = _emotional_state(label)
        action = "neutral_response"       # fixed — no rule engine
        prompt = _NO_PRODUCTION_PROMPT.format(
            label=es["label"],
            valence=es["valence"], arousal=es["arousal"], dominance=es["dominance"]
        )
        return {
            "generated":         self._call(prompt, text, use_history=True),
            "emotion_predicted": es["label"],
            "action":            action,
            "valence":           es["valence"],
            "arousal":           es["arousal"],
            "dominance":         es["dominance"],
        }

    # ── No Working Memory ─────────────────────────────────────────────────────
    # WorkingMemory bypassed: full cognitive processing happens BUT no
    # conversation history is injected into the GPT context.
    # Tests value of multi-turn conversational memory.
    def _no_memory(self, text: str, label: str) -> Dict:
        es     = _emotional_state(label)
        action = _select_action(label)
        prompt = _NO_MEMORY_PROMPT.format(
            action=action, label=es["label"],
            valence=es["valence"], arousal=es["arousal"], dominance=es["dominance"]
        )
        return {
            "generated":         self._call(prompt, text, use_history=False),  # ← no history
            "emotion_predicted": es["label"],
            "action":            action,
            "valence":           es["valence"],
            "arousal":           es["arousal"],
            "dominance":         es["dominance"],
        }

    # ── No Cognitive Layer ────────────────────────────────────────────────────
    # Entire cognitive layer removed. Plain GPT with supportive prompt.
    def _no_cognitive(self, text: str) -> Dict:
        return {
            "generated":         self._call(_NO_COGNITIVE_PROMPT, text, use_history=True),
            "emotion_predicted": "neutral",
            "action":            "respond",
            "valence":           0.0, "arousal": 0.0, "dominance": 0.0,
        }

    # ── GPT helper ────────────────────────────────────────────────────────────
    def _call(self, system: str, user_text: str, use_history: bool = True) -> str:
        messages = [{"role": "system", "content": system}]

        if use_history:
            for turn in self._history[-self.history_turns:]:
                messages.append({"role": "user",      "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["assistant"]})

        messages.append({"role": "user", "content": user_text})

        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=0.7, max_tokens=200,
                )
                out = resp.choices[0].message.content.strip()
                self._history.append({"user": user_text, "assistant": out})
                return out
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)

        fallback = "I understand. I'm here to support you."
        self._history.append({"user": user_text, "assistant": fallback})
        return fallback
