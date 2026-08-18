from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path

from config import UPLOAD_DIR
from src.predict import MemeSentimentPredictor

app = Flask(__name__)
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = MemeSentimentPredictor()
    return predictor

@app.route("/")
def index():
    return render_template("index.html")

@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Please upload a meme image."}), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    filename = secure_filename(file.filename)
    path = UPLOAD_DIR / filename
    file.save(path)

    try:
        result = get_predictor().predict(str(path))
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
