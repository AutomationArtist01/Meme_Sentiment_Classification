import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from config import DATA_DIR, MODEL_DIR, IMAGE_SIZE, BATCH_SIZE, SEED, CLASS_NAMES

model = tf.keras.models.load_model(MODEL_DIR / "meme_sentiment.keras")

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=CLASS_NAMES
)

y_true, y_pred = [], []

for images, labels in test_ds:
    probabilities = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(probabilities, axis=1))
    y_true.extend(labels.numpy())

print(classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    zero_division=0
))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
disp.plot()
plt.title("Meme Sentiment Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
print("Saved confusion_matrix.png")
