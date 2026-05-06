import sys
import os
sys.path.append(os.path.abspath("../backend"))

from app.cognitive.cognitive_controller import CognitiveController
from app.services.gpt_service import GPTService

# ---------------------------------------
# TEST INPUT
# ---------------------------------------

TEST_INPUT = "I feel very anxious about my job and I don't know what to do."

# ---------------------------------------
# 1. LLM ONLY (NO COGNITION)
# ---------------------------------------

def run_llm_only(text):
    gpt = GPTService()

    response = gpt.generate_response(
        text=text,
        emotional_state={
            "label": "neutral",
            "valence": 0,
            "arousal": 0,
            "dominance": 0
        },
        selected_action="respond"
    )

    return response


# ---------------------------------------
# 2. DEFAULT COGNITIVE SYSTEM
# ---------------------------------------

def run_cognitive_default(text):

    agent = CognitiveController(use_cognition=True)

    result = agent.process_input({
        "text": text,
        "face_emotion": "neutral"
    })

    return result


# ---------------------------------------
# 3. MODIFIED RULE SYSTEM
# ---------------------------------------

class ModifiedRuleAgent(CognitiveController):

    def __init__(self):
        super().__init__(use_cognition=True)

    def process_input(self, multimodal_input):

        # Run normal pipeline first
        perception = multimodal_input

        self.working_memory.update_input(perception)
        self.working_memory.add_to_history(perception)

        # Force emotion detection from text (simple)
        text = perception.get("text", "")
        self.emotion.infer_from_text(text)
        emotional_state = self.emotion.get_state()

        # CHANGE BEHAVIOR HERE 👇
        selected_action = "problem_solving"   # instead of empathy

        generated_text = self.gpt.generate_response(
            text=text,
            emotional_state=emotional_state,
            selected_action=selected_action
        )

        return {
            "text": generated_text,
            "agent_emotion": emotional_state["label"],
            "selected_action": selected_action
        }


def run_cognitive_modified(text):

    agent = ModifiedRuleAgent()

    result = agent.process_input({
        "text": text,
        "face_emotion": "neutral"
    })

    return result


# ---------------------------------------
# MAIN
# ---------------------------------------

def main():

    print("\n==============================")
    print("CONTROLLABILITY EXPERIMENT")
    print("==============================\n")

    print("INPUT:")
    print(TEST_INPUT)

    print("\n--- LLM ONLY ---")
    print(run_llm_only(TEST_INPUT))

    print("\n--- COGNITIVE (DEFAULT) ---")
    default = run_cognitive_default(TEST_INPUT)
    print("Action:", default["cognitive_metadata"]["selected_action"])
    print("Response:", default["text"])

    print("\n--- COGNITIVE (MODIFIED RULE) ---")
    modified = run_cognitive_modified(TEST_INPUT)
    print("Action:", modified["selected_action"])
    print("Response:", modified["text"])


if __name__ == "__main__":
    main()