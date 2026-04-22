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

                try:
                  transcription = self.stt.transcribe_audio(self.audio_buffer)
                except Exception as e:
                  print("STT ERROR:", e)
                  self.audio_buffer = b''
                  return {"status": "audio_error"}

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
from app.multimodal.facial_emotion_service import FacialEmotionService
from app.multimodal.tts_service import TTSService

import asyncio
import numpy as np
import time


class StreamingPipeline:

    def __init__(self):

        # Core modules
        self.controller = CognitiveController()
        self.fusion = MultimodalFusion()
        self.stt = STTService()
        self.face_service = FacialEmotionService()
        self.tts_service = TTSService()

        # -------------------------
        # AUDIO STREAMING STATE
        # -------------------------
        self.audio_buffer = np.array([], dtype=np.float32)

        self.speech_detected = False
        self.silence_counter = 0

        self.last_chunk_time = time.time()

        # Voice Activity Detection
        self.SILENCE_THRESHOLD = 0.008
        self.SILENCE_CHUNKS_TO_END = 4

        # Memory safety
        self.MAX_AUDIO_SECONDS = 8
        self.MAX_SAMPLES = 16000 * self.MAX_AUDIO_SECONDS

        # Prevent overlapping STT
        self.processing_audio = False

        # -------------------------
        # VIDEO / FACE STATE
        # -------------------------
        self.current_face_emotion = "neutral"
        self.last_face_time = 0

        # -------------------------
        # PERFORMANCE
        # -------------------------
        self.loop = asyncio.get_event_loop()

    # =====================================================
    # MAIN ROUTER
    # =====================================================

    async def process_stream(self, data: dict):

        msg_type = data.get("type", "")

        # -------------------------
        # TEXT INPUT
        # -------------------------
        if msg_type == "text":
            return await self.handle_text(data)

        # -------------------------
        # VIDEO FRAME
        # -------------------------
        elif msg_type == "video_frame":
            return await self.handle_video(data)

        # -------------------------
        # AUDIO STREAM
        # -------------------------
        elif msg_type == "audio_chunk":
            return await self.handle_audio(data)

        return {"status": "unknown"}

    # =====================================================
    # TEXT
    # =====================================================

    async def handle_text(self, data):

        perception_input = {
            "type": "text",
            "text": data.get("text", ""),
            "face_emotion": self.current_face_emotion,
            "voice_emotion": None
        }

        fused = self.fusion.fuse(perception_input)

        return await self.run_cognitive(fused)

    # =====================================================
    # VIDEO / FACE
    # =====================================================

    async def handle_video(self, data):

        try:
            # throttle face processing
            now = time.time()

            if now - self.last_face_time < 1.0:
                return {"status": "face_wait"}

            self.last_face_time = now

            frame = data.get("data", None)

            if frame:
                emotion = self.face_service.detect_emotion(frame)

                if emotion:
                    self.current_face_emotion = emotion

            return {
                "status": "face_updated",
                "face_emotion": self.current_face_emotion
            }

        except Exception as e:
            print("Face error:", e)
            return {"status": "face_error"}

    # =====================================================
    # AUDIO
    # =====================================================

    async def handle_audio(self, data):

        try:
            chunk = np.array(
                data.get("data", []),
                dtype=np.float32
            )

            if len(chunk) == 0:
                return {"status": "empty"}

            energy = np.mean(np.abs(chunk))

            # -------------------------
            # SPEECH DETECTED
            # -------------------------
            if energy > self.SILENCE_THRESHOLD:

                self.speech_detected = True
                self.silence_counter = 0

                self.audio_buffer = np.concatenate(
                    (self.audio_buffer, chunk)
                )

                # memory safe buffer
                if len(self.audio_buffer) > self.MAX_SAMPLES:
                    self.audio_buffer = self.audio_buffer[
                        -self.MAX_SAMPLES:
                    ]

                return {"status": "listening"}

            # -------------------------
            # SILENCE AFTER SPEECH
            # -------------------------
            else:

                if self.speech_detected:

                    self.silence_counter += 1

                    if self.silence_counter >= self.SILENCE_CHUNKS_TO_END:

                        if self.processing_audio:
                            return {"status": "busy"}

                        self.processing_audio = True

                        result = await self.process_completed_speech()

                        self.processing_audio = False

                        return result

            return {"status": "silence"}

        except Exception as e:
            print("Audio error:", e)
            self.reset_audio_state()
            return {"status": "audio_error"}

    # =====================================================
    # FINISHED SPEECH -> STT -> COGNITION
    # =====================================================

    async def process_completed_speech(self):

        try:

            if len(self.audio_buffer) < 2000:
                self.reset_audio_state()
                return {"status": "too_short"}

            audio_copy = self.audio_buffer.copy()

            self.reset_audio_state()

            # Run Whisper off main thread
            transcription = await self.loop.run_in_executor(
                None,
                self.stt.transcribe_array,
                audio_copy
            )

            if not transcription.strip():
                return {"status": "no_text"}

            perception_input = {
                "type": "text",
                "text": transcription,
                "face_emotion": self.current_face_emotion,
                "voice_emotion": None
            }

            fused = self.fusion.fuse(perception_input)

            return await self.run_cognitive(fused)

        except Exception as e:

            print("STT error:", e)

            self.reset_audio_state()

            return {"status": "stt_error"}

    # =====================================================
    # COGNITION + TTS
    # =====================================================

    async def run_cognitive(self, fused_input):

        try:

            result = await self.loop.run_in_executor(
                None,
                self.controller.process_input,
                fused_input
            )

            # -------------------------
            # TTS
            # -------------------------
            audio_file = await self.loop.run_in_executor(
                None,
                self.tts_service.synthesize,
                result["text"],
                result.get("speaking_speed", 1.0),
                result.get("tone", "neutral")
            )

            result["audio_file"] = audio_file

            print("FINAL RESPONSE:")
            print(result["text"])

            return result

        except Exception as e:

            print("Cognitive error:", e)

            return {
                "text": "I apologize, something went wrong.",
                "agent_emotion": "neutral",
                "tone": "calm",
                "gesture": "idle",
                "speaking_speed": 1.0,
                "audio_file": ""
            }

    # =====================================================
    # RESET AUDIO STATE
    # =====================================================

    def reset_audio_state(self):

        self.audio_buffer = np.array([], dtype=np.float32)
        self.speech_detected = False
        self.silence_counter = 0