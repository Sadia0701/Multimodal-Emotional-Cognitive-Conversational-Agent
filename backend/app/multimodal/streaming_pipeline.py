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
