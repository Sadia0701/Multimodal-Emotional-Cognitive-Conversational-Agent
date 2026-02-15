class ProductionRuleEngine:
    def select_action(self, emotional_label: str):
        if emotional_label in ["sad", "fear"]:
            return "provide_empathy"
        elif emotional_label == "angry":
            return "de_escalate"
        elif emotional_label == "happy":
            return "reinforce_positive"
        else:
            return "neutral_response"
