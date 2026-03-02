from TTS.api import TTS
import torch
import uuid
import os


class TTSService:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load research-grade VITS model
        self.tts = TTS(
            model_name="tts_models/en/ljspeech/vits",
            progress_bar=False
        ).to(self.device)

    def synthesize(self, text: str, speaking_speed: float = 1.0):

        output_path = f"temp_{uuid.uuid4()}.wav"

        self.tts.tts_to_file(
            text=text,
            file_path=output_path,
            speed=speaking_speed
        )
        #instead os sending file path, convert to base64
        #return output_path
        import base64

        with open(output_path, "rb") as f:
            audio_bytes = f.read()

        os.remove(output_path)

        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return audio_base64