from faster_whisper import WhisperModel
import numpy as np
import tempfile
import soundfile as sf


class STTService:

    def __init__(self):
        self.model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )

    def transcribe_audio(self, audio_bytes: bytes) -> str:

        # Save temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()

            segments, _ = self.model.transcribe(tmp.name)

            text = ""
            for segment in segments:
                text += segment.text + " "

        return text.strip()
