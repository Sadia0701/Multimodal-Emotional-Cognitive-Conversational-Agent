from typing import Dict, Any, List


class WorkingMemory:
    def __init__(self, max_history: int = 10):
        self.current_input: Dict[str, Any] = {}
        self.recent_history: List[Dict[str, Any]] = []
        self.attention_focus: str | None = None
        self.max_history = max_history

    def update_input(self, input_data: Dict[str, Any]):
        self.current_input = input_data

    def add_to_history(self, item: Dict[str, Any]):
        self.recent_history.append(item)
        if len(self.recent_history) > self.max_history:
            self.recent_history.pop(0)

    def set_attention(self, focus: str):
        self.attention_focus = focus
