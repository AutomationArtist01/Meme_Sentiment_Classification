import json
import numpy as np
from PIL import Image
import tensorflow as tf

from config import MODEL_DIR, IMAGE_SIZE

class MemeSentimentPredictor:
    def __init__(self):
        model_path = MODEL_DIR / "meme_sentiment.keras"
        labels_path = MODEL_DIR / "class_names.json"

        if not model_path.exists():
            raise FileNotFoundError("Train the model first: python src/train.py")

        self.model = tf.keras.models.load_model(model_path)
        self.classes = json.loads(labels_path.read_text(encoding="utf-8"))

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
        array = np.asarray(image, dtype=np.float32)
        array = np.expand_dims(array, axis=0)

        probabilities = self.model.predict(array, verbose=0)[0]
        index = int(np.argmax(probabilities))

        return {
            "sentiment": self.classes[index],
            "confidence": round(float(probabilities[index]) * 100, 2),
            "probabilities": {
                label: round(float(probabilities[i]) * 100, 2)
                for i, label in enumerate(self.classes)
            }
        }
