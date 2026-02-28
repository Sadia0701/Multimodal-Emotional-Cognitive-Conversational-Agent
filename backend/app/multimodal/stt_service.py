'''version1
from faster_whisper import WhisperModel


class STTService:

    def __init__(self):

        print("Loading Whisper medium on GPU...")

        self.model = WhisperModel(
            "medium",
            device="cuda",
            compute_type="float16"
        )

        print("Whisper ready.")

    def transcribe_audio(self, audio_path):

        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True
        )

        text = ""
        for segment in segments:
            text += segment.text + " "

        return text.strip()

'''
'''
#Version 2
# app/multimodal/stt_service.py

from faster_whisper import WhisperModel
import subprocess
import tempfile
import os


class STTService:

    def __init__(self):

        print("Loading Whisper medium on GPU...")

        self.model = WhisperModel(
            "medium",
            device="cuda",
            compute_type="float16"
        )

        print("Whisper ready.")

    def convert_to_wav(self, input_path):

        output_path = input_path.replace(".webm", ".wav")

        command = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            output_path
        ]

        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return output_path

    def transcribe_audio(self, audio_path):

        # Convert to WAV first
        wav_path = self.convert_to_wav(audio_path)

        segments, info = self.model.transcribe(
            wav_path,
            beam_size=5,
            vad_filter=True
        )

        text = ""
        for segment in segments:
            text += segment.text + " "

        os.remove(wav_path)

        return text.strip()

'''

#Version 3
# app/multimodal/stt_service.py

from faster_whisper import WhisperModel
import numpy as np


class STTService:

    def __init__(self):

        print("Loading Whisper base/small on CPU...")

        self.model = WhisperModel(
            "base", #lighter model for cpu
            device="cpu",
            compute_type="int8"
        )

'''Whisper when using GPU
        print("Loading Whisper medium on GPU...")

        self.model = WhisperModel(
            "medium",
            device="cuda",
            compute_type="float16"
        )
'''
        print("Whisper ready.")

    def transcribe_array(self, audio_array):

        segments, info = self.model.transcribe(
            audio_array,
            beam_size=5,
            vad_filter=True
        )

        text = ""
        for segment in segments:
            text += segment.text + " "

        return text.strip()