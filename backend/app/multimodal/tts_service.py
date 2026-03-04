'''
#Code for using CoquiTTS, which is not working right now as for t i need to downgrade python from 3.12 to 3.11 
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

'''

# Piper TTS

from piper import PiperVoice
import base64
import io
import wave


class TTSService:

    def __init__(self):

        print("Loading Piper TTS model...")

        self.voice = PiperVoice.load(
            "models/en_US-lessac-medium.onnx",
            config_path="models/en_US-lessac-medium.onnx.json"
        )

        print("Piper TTS ready.")


    def synthesize(self, text: str, speaking_speed: float = 1.0, tone: str = "neutral"):

        import io
        import wave
        import base64
        import numpy as np

        print("TTS generating...")
        print("Speed:", speaking_speed)
        print("Tone:", tone)

        audio_stream = self.voice.synthesize(text)

        audio_bytes = b''

        for chunk in audio_stream:

            if chunk.audio_int16_bytes is not None:
                audio_bytes += chunk.audio_int16_bytes

            elif chunk.audio_int16_array is not None:
                audio_bytes += chunk.audio_int16_array.tobytes()

            elif chunk.audio_float_array is not None:
                int16_audio = (chunk.audio_float_array * 32767).astype(np.int16)
                audio_bytes += int16_audio.tobytes()

        print("Audio bytes length:", len(audio_bytes))

        if len(audio_bytes) == 0:
            print("⚠ Piper produced empty audio")
            return ""

        buffer = io.BytesIO()

        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(audio_bytes)

        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return encoded