import cv2
import numpy as np
from tensorflow.keras.models import load_model

class EmotionDetector:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def detect_emotion(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (48, 48))
        reshaped_image = np.reshape(resized_image, (1, 48, 48, 1)) / 255.0
        prediction = self.model.predict(reshaped_image)
        emotion_label = self.emotion_labels[np.argmax(prediction)]
        return emotion_label
