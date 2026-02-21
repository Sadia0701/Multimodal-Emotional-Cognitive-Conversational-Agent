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