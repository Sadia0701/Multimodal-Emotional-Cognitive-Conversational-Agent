
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Multimodal Cognitive Agent")

# Allow frontend to communicate later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Multimodal Cognitive Agent is running"}

from app.cognitive.cognitive_controller import CognitiveController
from pydantic import BaseModel

controller = CognitiveController()


class MultimodalInput(BaseModel):
    text: str
    face_emotion: str = "neutral"
    voice_emotion: str | None = None


@app.post("/process")
def process_input(data: MultimodalInput):
    result = controller.process_input(data.dict())
    return result
