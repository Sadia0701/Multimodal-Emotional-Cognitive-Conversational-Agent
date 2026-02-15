class MultimodalFusion:

    def fuse(self, data: dict):

        # Expected data format:
        # {
        #   "text": "...",
        #   "face_emotion": "...",
        #   "voice_emotion": "...",
        # }

        return {
            "text": data.get("text", ""),
            "face_emotion": data.get("face_emotion", "neutral"),
            "voice_emotion": data.get("voice_emotion", None)
        }
