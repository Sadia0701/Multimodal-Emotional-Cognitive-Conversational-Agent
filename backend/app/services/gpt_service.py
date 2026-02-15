from openai import OpenAI
from app.config import settings


class GPTService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def generate_response(self, text, emotional_state, selected_action):

        system_prompt = f"""
You are a conversational agent.

The cognitive system has selected the action: {selected_action}.

Emotional state of user:
Label: {emotional_state['label']}
Valence: {emotional_state['valence']}
Arousal: {emotional_state['arousal']}
Dominance: {emotional_state['dominance']}

You must generate a response strictly aligned with the selected action.
Do not override the action.
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content
