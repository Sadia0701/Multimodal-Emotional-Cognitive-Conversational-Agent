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
        self.SILENCE_THRESHOLD = 0.003
        self.SILENCE_CHUNKS_TO_END = 2

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
'''
# app/multimodal/streaming_pipeline.py
# Version 4 — Production-ready pipeline
#
# KEY FIXES over v3:
#   1. handle_video runs DeepFace in run_in_executor — was blocking the
#      entire asyncio event loop (0.5-3 s), preventing audio chunks from
#      being received and destroying VAD silence detection.
#   2. Speech timeout (SPEECH_TIMEOUT_SEC): if the user speaks for longer
#      than N seconds without a clear silence, force STT completion so
#      speech is never lost due to AGC / noisy environments.
#   3. SILENCE_THRESHOLD raised to 0.008 (was 0.003) — more robust to
#      background noise and browser AGC keeping a residual signal level.
#   4. processing_audio lock is now always released in a try/finally block
#      so a crashed STT call can never permanently lock the pipeline.

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

        # ── Core modules ──────────────────────────────────────────────────
        self.controller   = CognitiveController()
        self.fusion       = MultimodalFusion()
        self.stt          = STTService()
        self.face_service = FacialEmotionService()
        self.tts_service  = TTSService()

        # ── Audio streaming state ─────────────────────────────────────────
        self.audio_buffer  = np.array([], dtype=np.float32)
        self.speech_detected   = False
        self.silence_counter   = 0
        self.speech_start_time = 0.0

        # ── VAD parameters ────────────────────────────────────────────────
        # Raised from 0.003 → 0.008 to be robust against AGC / background
        # noise keeping the signal slightly above zero between words.
        self.SILENCE_THRESHOLD     = 0.008
        self.SILENCE_CHUNKS_TO_END = 3     # ≈ 0.77 s of silence at 4096/16kHz

        # Force STT after this many seconds even without a clear silence
        # (handles AGC environments where silence never hits the threshold)
        self.SPEECH_TIMEOUT_SEC = 8.0

        # ── Memory safety ─────────────────────────────────────────────────
        self.MAX_AUDIO_SECONDS = 10
        self.MAX_SAMPLES       = 16000 * self.MAX_AUDIO_SECONDS

        # ── Concurrency guard ─────────────────────────────────────────────
        self.processing_audio = False

        # ── Video / face state ────────────────────────────────────────────
        self.current_face_emotion = "neutral"
        self.last_face_time       = 0.0

        # ── Event loop reference (set at first use via get_running_loop) ──
        self._loop = None

    @property
    def loop(self):
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.get_event_loop()
        return self._loop

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN ROUTER
    # ═══════════════════════════════════════════════════════════════════════

    async def process_stream(self, data: dict):
        msg_type = data.get("type", "")

        if   msg_type == "text":        return await self.handle_text(data)
        elif msg_type == "video_frame": return await self.handle_video(data)
        elif msg_type == "audio_chunk": return await self.handle_audio(data)

        return {"status": "unknown_type"}

    # ═══════════════════════════════════════════════════════════════════════
    # TEXT
    # ═══════════════════════════════════════════════════════════════════════

    async def handle_text(self, data):
        perception = {
            "type":          "text",
            "text":          data.get("text", ""),
            "face_emotion":  data.get("face_emotion", self.current_face_emotion),
            "voice_emotion": None,
        }
        return await self.run_cognitive(self.fusion.fuse(perception))

    # ═══════════════════════════════════════════════════════════════════════
    # VIDEO / FACE  —  *** CRITICAL FIX: run_in_executor ***
    # ═══════════════════════════════════════════════════════════════════════

    async def handle_video(self, data):
        try:
            now = time.time()

            # Server-side throttle: max one DeepFace call per 1.2 s
            if now - self.last_face_time < 1.2:
                return {"status": "face_wait"}

            self.last_face_time = now

            frame = data.get("data")
            if not frame:
                return {"status": "face_wait"}

            # ─────────────────────────────────────────────────────────────
            # FIX: DeepFace.analyze() is synchronous and CPU-intensive.
            # Calling it directly in an async function blocks the event
            # loop (and therefore the WebSocket receive loop) for up to
            # 3 seconds, which destroys VAD silence detection.
            # run_in_executor offloads it to a thread pool so the event
            # loop stays free to receive audio chunks while DeepFace runs.
            # ─────────────────────────────────────────────────────────────
            emotion = await self.loop.run_in_executor(
                None,
                self.face_service.detect_emotion,
                frame,
            )

            if emotion:
                self.current_face_emotion = emotion

            return {
                "status":       "face_updated",
                "face_emotion": self.current_face_emotion,
            }

        except Exception as e:
            print(f"[Face] Error: {e}")
            return {"status": "face_error"}

    # ═══════════════════════════════════════════════════════════════════════
    # AUDIO  —  VAD + buffer + silence/timeout triggers
    # ═══════════════════════════════════════════════════════════════════════

    async def handle_audio(self, data):
        try:
            chunk = np.array(data.get("data", []), dtype=np.float32)

            if len(chunk) == 0:
                return {"status": "empty"}

            energy = float(np.mean(np.abs(chunk)))

            # ── Speech energy detected ───────────────────────────────────
            if energy > self.SILENCE_THRESHOLD:

                if not self.speech_detected:
                    # First speech chunk — start the timeout clock
                    self.speech_start_time = time.time()

                self.speech_detected = True
                self.silence_counter = 0

                # Accumulate (memory-safe circular buffer)
                self.audio_buffer = np.concatenate((self.audio_buffer, chunk))
                if len(self.audio_buffer) > self.MAX_SAMPLES:
                    self.audio_buffer = self.audio_buffer[-self.MAX_SAMPLES:]

                # ── Speech timeout: force completion on long utterances ──
                # Prevents the pipeline from never completing when AGC /
                # background noise prevents clean silence detection.
                elapsed = time.time() - self.speech_start_time
                if elapsed >= self.SPEECH_TIMEOUT_SEC:
                    print(f"[VAD] Speech timeout after {elapsed:.1f}s — forcing STT")
                    return await self._trigger_stt()

                return {"status": "listening"}

            # ── Silence after speech ─────────────────────────────────────
            else:
                if self.speech_detected:
                    self.silence_counter += 1

                    if self.silence_counter >= self.SILENCE_CHUNKS_TO_END:
                        return await self._trigger_stt()

            return {"status": "silence"}

        except Exception as e:
            print(f"[Audio] Error: {e}")
            self.reset_audio_state()
            return {"status": "audio_error"}

    async def _trigger_stt(self):
        """Fire STT on the accumulated audio buffer (with concurrency guard)."""
        if self.processing_audio:
            return {"status": "busy"}

        self.processing_audio = True
        try:
            return await self.process_completed_speech()
        finally:
            # Always release the lock — even if process_completed_speech raises
            self.processing_audio = False

    # ═══════════════════════════════════════════════════════════════════════
    # COMPLETED SPEECH → STT → COGNITION
    # ═══════════════════════════════════════════════════════════════════════

    async def process_completed_speech(self):
        try:
            if len(self.audio_buffer) < 3200:   # < 0.2 s — too short
                self.reset_audio_state()
                return {"status": "too_short"}

            audio_copy = self.audio_buffer.copy()
            self.reset_audio_state()

            # Whisper runs in thread pool (already non-blocking)
            transcription = await self.loop.run_in_executor(
                None,
                self.stt.transcribe_array,
                audio_copy,
            )

            if not transcription or not transcription.strip():
                print("[STT] Empty transcription — no speech detected in audio")
                return {"status": "no_text"}

            print(f"[STT] Transcribed: '{transcription.strip()}'")

            perception = {
                "type":          "text",
                "text":          transcription.strip(),
                "face_emotion":  self.current_face_emotion,
                "voice_emotion": None,
            }

            return await self.run_cognitive(self.fusion.fuse(perception))

        except Exception as e:
            print(f"[STT] Error: {e}")
            self.reset_audio_state()
            return {"status": "stt_error", "detail": str(e)}

    # ═══════════════════════════════════════════════════════════════════════
    # COGNITION + TTS
    # ═══════════════════════════════════════════════════════════════════════

    async def run_cognitive(self, fused_input):
        try:
            result = await self.loop.run_in_executor(
                None,
                self.controller.process_input,
                fused_input,
            )

            audio_b64 = await self.loop.run_in_executor(
                None,
                self.tts_service.synthesize,
                result["text"],
                result.get("speaking_speed", 1.0),
                result.get("tone", "neutral"),
            )

            result["audio_file"] = audio_b64

            print(f"[Agent] Emotion={result.get('agent_emotion')} "
                  f"Action={result.get('cognitive_metadata', {}).get('selected_action', '?')}")
            print(f"[Agent] Response: {result['text'][:80]}...")

            return result

        except Exception as e:
            print(f"[Cognitive] Error: {e}")
            return {
                "text":          "I'm sorry, something went wrong on my end. Please try again.",
                "agent_emotion": "neutral",
                "tone":          "calm",
                "gesture":       "idle",
                "speaking_speed": 1.0,
                "audio_file":    "",
            }

    # ═══════════════════════════════════════════════════════════════════════
    # RESET
    # ═══════════════════════════════════════════════════════════════════════

    def reset_audio_state(self):
        self.audio_buffer      = np.array([], dtype=np.float32)
        self.speech_detected   = False
        self.silence_counter   = 0
        self.speech_start_time = 0.0        
        