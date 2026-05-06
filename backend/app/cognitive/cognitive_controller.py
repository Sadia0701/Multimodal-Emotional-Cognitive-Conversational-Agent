from .working_memory import WorkingMemory
from .emotional_module import EmotionalModule
from .goal_module import GoalModule
from .production_rules import ProductionRuleEngine
from .utility_module import UtilityModule
from app.services.gpt_service import GPTService


class CognitiveController:

    def __init__(self, use_cognition=True):
        self.use_cognition = use_cognition

        self.working_memory = WorkingMemory()
        self.emotion = EmotionalModule()
        self.goals = GoalModule()
        self.rules = ProductionRuleEngine()
        self.utility = UtilityModule()
        self.gpt = GPTService()

    # -----------------------------------------------------
    def process_input(self, multimodal_input: dict):

        # -----------------------------
        # BASELINE MODE (NO COGNITION)
        # -----------------------------
        if not self.use_cognition:

            text = multimodal_input.get("text", "")
            face_emotion = multimodal_input.get("face_emotion", "neutral")

            emotional_state = {
                "label": face_emotion,   # USE INPUT EMOTION
                "valence": 0.0,
                "arousal": 0.0,
                "dominance": 0.0
            }

            generated_text = self.gpt.generate_response(
                text=text,
                emotional_state=emotional_state,
                selected_action="respond"
            )

            return {
                "text": generated_text,
                "agent_emotion": face_emotion,  # IMPORTANT
                "tone": "neutral",
                "facial_expression": "neutral",
                "gesture": "idle",
                "speaking_speed": 1.0
            }

        # -----------------------------
        # FULL COGNITIVE PIPELINE
        # -----------------------------

        perception = multimodal_input

        self.working_memory.update_input(perception)
        self.working_memory.add_to_history(perception)

        face_emotion = perception.get("face_emotion", "neutral")
        self.emotion.update_from_label(face_emotion)

        emotional_state = self.emotion.get_state()

        self.goals.add_goal("assist_user")
        primary_goal = self.goals.get_primary_goal()

        selected_action = self.rules.select_action(
            emotional_state["label"]
        )

        utility_score = self.utility.compute_utility(
            emotional_state["valence"]
        )

        generated_text = self.gpt.generate_response(
            text=perception.get("text", ""),
            emotional_state=emotional_state,
            selected_action=selected_action
        )

        agent_emotion = emotional_state["label"]

        tone = self._map_tone(agent_emotion)
        facial_expression = self._map_expression(agent_emotion)
        gesture = self._map_gesture(agent_emotion)
        speaking_speed = self._map_speaking_speed(agent_emotion)

        return {
            "text": generated_text,
            "agent_emotion": agent_emotion,
            "tone": tone,
            "facial_expression": facial_expression,
            "gesture": gesture,
            "speaking_speed": speaking_speed,
            "cognitive_metadata": {
                "emotional_state": emotional_state,
                "selected_action": selected_action,
                "utility_score": utility_score,
                "primary_goal": primary_goal,
                "history_length": len(self.working_memory.recent_history)
            }
        }

    # -----------------------------------------------------

    def _map_tone(self, emotion):
        mapping = {
            "happy": "warm",
            "sad": "gentle",
            "angry": "firm",
            "neutral": "calm",
            "fear": "soft",
            "surprise": "animated"
        }
        return mapping.get(emotion, "calm")

    def _map_expression(self, emotion):
        mapping = {
            "happy": "smile",
            "sad": "soft_sad",
            "angry": "serious",
            "neutral": "neutral",
            "fear": "concerned",
            "surprise": "raised_eyebrows"
        }
        return mapping.get(emotion, "neutral")

    def _map_gesture(self, emotion):
        mapping = {
            "happy": "slight_nod",
            "sad": "slow_head_tilt",
            "angry": "firm_posture",
            "neutral": "idle",
            "fear": "small_recoil",
            "surprise": "lean_forward"
        }
        return mapping.get(emotion, "idle")

    def _map_speaking_speed(self, emotion):
        mapping = {
            "happy": 1.1,
            "sad": 0.85,
            "angry": 1.2,
            "neutral": 1.0,
            "fear": 0.9,
            "surprise": 1.15
        }
        return mapping.get(emotion, 1.0)