from .working_memory import WorkingMemory
from .emotional_module import EmotionalModule
from .goal_module import GoalModule
from .production_rules import ProductionRuleEngine
from .utility_module import UtilityModule
from app.services.gpt_service import GPTService



class CognitiveController:
    def __init__(self):
        self.working_memory = WorkingMemory()
        self.emotion = EmotionalModule()
        self.goals = GoalModule()
        self.rules = ProductionRuleEngine()
        self.utility = UtilityModule()
        self.gpt = GPTService()


    def process_input(self, multimodal_input: dict):

        # 1. Perception
        perception = multimodal_input

        # 2. Update Working Memory
        self.working_memory.update_input(perception)
        self.working_memory.add_to_history(perception)

        # 3. Emotional Appraisal
        face_emotion = perception.get("face_emotion", "neutral")
        self.emotion.update_from_label(face_emotion)

        emotional_state = self.emotion.get_state()

        # 4. Goal Handling
        self.goals.add_goal("assist_user")
        primary_goal = self.goals.get_primary_goal()

        # 5. Decision (Production Rules)
        selected_action = self.rules.select_action(emotional_state["label"])

        # 6. Utility Evaluation
        utility_score = self.utility.compute_utility(
            emotional_state["valence"]
        )

        # 7. Structured Cognitive Output
        generated_response = self.gpt.generate_response(
        text=perception["text"],
        emotional_state=emotional_state,
        selected_action=selected_action
     )  

        return {
          "perception": perception,
          "emotional_state": emotional_state,
          "working_memory": {
               "attention_focus": self.working_memory.attention_focus,
               "history_length": len(self.working_memory.recent_history)
            },
           "goal_state": {
           "primary_goal": primary_goal
           },
           "decision": {
           "selected_action": selected_action,
           "utility_score": utility_score
          },
          "generated_response": generated_response
           }
