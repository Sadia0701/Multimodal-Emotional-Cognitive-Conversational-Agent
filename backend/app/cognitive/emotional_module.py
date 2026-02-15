class EmotionalModule:
    def __init__(self):
        self.valence = 0.0
        self.arousal = 0.0
        self.dominance = 0.0
        self.label = "neutral"

    def update_from_label(self, label: str):
        self.label = label
        self._map_label_to_dimensions()

    def _map_label_to_dimensions(self):
        mapping = {
            "happy": (0.8, 0.6, 0.7),
            "sad": (-0.7, 0.4, 0.3),
            "angry": (-0.6, 0.8, 0.6),
            "fear": (-0.8, 0.7, 0.4),
            "surprise": (0.4, 0.8, 0.6),
            "neutral": (0.0, 0.2, 0.5)
        }

        self.valence, self.arousal, self.dominance = mapping.get(
            self.label,
            (0.0, 0.0, 0.0)
        )

    def get_state(self):
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "label": self.label
        }
#ee