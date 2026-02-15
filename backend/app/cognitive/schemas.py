from pydantic import BaseModel
from typing import List, Dict, Any


class EmotionalState(BaseModel):
    valence: float
    arousal: float
    dominance: float
    label: str


class WorkingMemoryState(BaseModel):
    current_input: Dict[str, Any]
    recent_history: List[Dict[str, Any]]
    attention_focus: str | None


class CognitiveState(BaseModel):
    working_memory: WorkingMemoryState
    emotional_state: EmotionalState
    goals: List[str]
    selected_action: str | None
