from deepface import DeepFace
import base64
import cv2
import numpy as np


class FacialEmotionService:

    def detect_emotion(self, base64_image):

        img_data = base64.b64decode(base64_image.split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False
            )

            return result[0]["dominant_emotion"]

        except:
            return "neutral"