'''Version 1
from app.cognitive.cognitive_controller import CognitiveController
from app.multimodal.fusion_engine import MultimodalFusion
from app.multimodal.stt_service import STTService

import asyncio


class StreamingPipeline:

    def __init__(self):
        self.controller = CognitiveController()
        self.fusion = MultimodalFusion()
        self.stt = STTService()
        self.audio_buffer = b''

    async def process_stream(self, data: dict):

        # TEXT INPUT (existing)
        if data.get("type") == "text":
            fused_input = self.fusion.fuse(data)
            return await self._run_cognitive(fused_input)

        # AUDIO INPUT (new)
        elif data.get("type") == "audio_chunk":

            chunk = bytes.fromhex(data["data"])
            self.audio_buffer += chunk

            # If buffer > 2 seconds (approx 32000 bytes at 16kHz mono)
            if len(self.audio_buffer) > 32000:

                transcription = self.stt.transcribe_audio(self.audio_buffer)
                self.audio_buffer = b''

                fused_input = self.fusion.fuse({
                    "text": transcription,
                    "face_emotion": data.get("face_emotion", "neutral")
                })

                return await self._run_cognitive(fused_input)

            return {"status": "buffering_audio"}

    async def _run_cognitive(self, fused_input):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.controller.process_input,
            fused_input
        )
        return result

    import tempfile
import os


async def process_audio_bytes(self, audio_bytes: bytes):

    # Save clean audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # GPU Transcription
    transcription = self.stt.transcribe_audio(tmp_path)

    # Remove temp file
    os.remove(tmp_path)

    if not transcription:
        return {"status": "no_speech_detected"}

    # Build structured perception input
    data = {
        "text": transcription,
        "face_emotion": "neutral",
        "voice_emotion": None
    }

    return await self.process_stream(data)    
'''

'''Version 2
from app.cognitive.cognitive_controller import CognitiveController
from app.multimodal.fusion_engine import MultimodalFusion
from app.multimodal.stt_service import STTService

import asyncio
import tempfile
import os


class StreamingPipeline:

    def __init__(self):
        self.controller = CognitiveController()
        self.fusion = MultimodalFusion()
        self.stt = STTService()
        self.audio_buffer = b''

    async def process_stream(self, data: dict):

        # TEXT INPUT
        if data.get("type") == "text":
            fused_input = self.fusion.fuse(data)
            return await self._run_cognitive(fused_input)

        # AUDIO CHUNK INPUT (real-time streaming mode)
        elif data.get("type") == "audio_chunk":

            chunk = bytes.fromhex(data["data"])
            self.audio_buffer += chunk

            if len(self.audio_buffer) > 32000:

                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                    tmp.write(self.audio_buffer)
                    tmp_path = tmp.name

                transcription = self.stt.transcribe_audio(tmp_path)
                os.remove(tmp_path)
                self.audio_buffer = b''

                fused_input = self.fusion.fuse({
                    "text": transcription,
                    "face_emotion": data.get("face_emotion", "neutral")
                })

                return await self._run_cognitive(fused_input)

            return {"status": "buffering_audio"}

    async def process_audio_bytes(self, audio_bytes: bytes):

        # Save clean audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # GPU Transcription
        transcription = self.stt.transcribe_audio(tmp_path)

        # Remove temp file
        os.remove(tmp_path)

        if not transcription:
            return {"status": "no_speech_detected"}

        # Build structured perception input
        data = {
            "type": "text",  # IMPORTANT
            "text": transcription,
            "face_emotion": "neutral",
            "voice_emotion": None
        }

        return await self.process_stream(data)

    async def _run_cognitive(self, fused_input):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.controller.process_input,
            fused_input
        )
        return result

'''
#Version 3
# app/multimodal/streaming_pipeline.py

from app.cognitive.cognitive_controller import CognitiveController
from app.multimodal.fusion_engine import MultimodalFusion
from app.multimodal.stt_service import STTService

import asyncio
import tempfile
import os
import time


class StreamingPipeline:

    def __init__(self):
        self.controller = CognitiveController()
        self.fusion = MultimodalFusion()
        self.stt = STTService()

        # VAD + Streaming state
        import numpy as np
        self.audio_buffer = np.array([], dtype=np.float32)
        self.silence_counter = 0
        self.speech_detected = False

   #    self.audio_buffer = b''
    #   self.last_audio_time = time.time()


    async def process_stream(self, data: dict):

        # -------------------------
        # TEXT INPUT (already supported)
        # -------------------------
        if data.get("type") == "text":
            fused_input = self.fusion.fuse(data)
            return await self._run_cognitive(fused_input)

        # -------------------------
        # AUDIO STREAMING INPUT
        # -------------------------
        elif data.get("type") == "audio_chunk":

            import numpy as np

            chunk = np.array(data["data"], dtype=np.float32)

            # Initialize state variables
            if not hasattr(self, "audio_buffer"):
                self.audio_buffer = np.array([], dtype=np.float32)
                self.silence_counter = 0
                self.speech_detected = False

            # Simple energy-based VAD
            energy = np.mean(np.abs(chunk))

            SILENCE_THRESHOLD = 0.01
            SILENCE_CHUNKS_TO_END = 8  # ~8 chunks ≈ ~1 sec silence

            if energy > SILENCE_THRESHOLD:
                self.speech_detected = True
                self.silence_counter = 0
                self.audio_buffer = np.concatenate((self.audio_buffer, chunk))
                return {"status": "listening"}

            else:
                if self.speech_detected:
                    self.silence_counter += 1

                    if self.silence_counter >= SILENCE_CHUNKS_TO_END:

                        # End of speech detected
                        transcription = self.stt.transcribe_array(self.audio_buffer)

                        # Reset state
                        self.audio_buffer = np.array([], dtype=np.float32)
                        self.silence_counter = 0
                        self.speech_detected = False

                        if not transcription.strip():
                            return {"status": "silence"}

                        perception_input = {
                            "type": "text",
                            "text": transcription,
                            "face_emotion": data.get("face_emotion", "neutral"),
                            "voice_emotion": None
                        }

                        fused_input = self.fusion.fuse(perception_input)

                        return await self._run_cognitive(fused_input)

                return {"status": "silence"}

    async def _run_cognitive(self, fused_input):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.controller.process_input,
            fused_input
        )
        return result