# Meme Sentiment Classification

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-MobileNetV2-D00000?logo=keras&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-Memotion%207k-8b5cf6)

A deep learning project that looks at a **meme image** and predicts whether its sentiment is
**Positive**, **Neutral** or **Negative** — trained with transfer learning on **MobileNetV2**
and served through a clean **Flask web app** where you simply drag-and-drop a meme.

---

## Table of Contents

- [About the Project](#about-the-project)
- [Demo](#demo)
- [How It Works](#how-it-works)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Dataset — Where to Get It](#dataset--where-to-get-it)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Documentation](#documentation)
- [Limitations & Future Work](#limitations--future-work)
- [Tech Stack](#tech-stack)
- [References](#references)

---

## About the Project

Memes are one of the most shared forms of content on social media. For media & entertainment
companies, brands and content moderators, knowing whether the memes about a topic are positive,
neutral or negative is valuable feedback — but reading every meme by hand is impossible at scale.

This project builds an automatic classifier for that job:

- **Input:** any meme image (JPG / PNG / GIF)
- **Output:** predicted sentiment + confidence + probability for each of the 3 classes
- **Approach:** MobileNetV2 CNN (pre-trained on ImageNet) adapted to memes with **transfer learning**
- **Data:** the **Memotion 7k** dataset (~7,000 human-labelled memes)
- **Deployment:** Flask web application with a drag-and-drop UI

Very little research exists on meme sentiment specifically — the area only received a proper
benchmark in 2020 with the Memotion task — which is what makes it an interesting problem.

---

## Demo

Run the app (see [How to Run](#how-to-run)), open **http://127.0.0.1:5001**, drop a meme, click
**Predict Sentiment**:

```
Prediction : POSITIVE   (85.38% confidence)
Positive   ████████████████████  85.38%
Neutral    ██                    10.42%
Negative   █                      4.20%
```

The UI shows the uploaded image preview, the verdict in colour, and animated probability bars.

---

## How It Works

```
Meme image
   → Resize to 224×224 & normalise (0–1)
   → Data augmentation (flip / rotate / zoom — training only)
   → MobileNetV2 backbone (pre-trained feature extractor)
   → Global Average Pooling
   → Dense layers + Softmax
   → Positive / Neutral / Negative
```

**Training happens in two phases:**

| Phase | What is trained | Learning rate | Epochs |
|-------|-----------------|---------------|--------|
| 1 — Feature extraction | Only the new dense head (backbone frozen) | 1e-4 | up to 10 |
| 2 — Fine-tuning | Head + last 30 layers of MobileNetV2 | 1e-5 | up to 10 |

Three Keras callbacks manage the run automatically:

- **ModelCheckpoint** — saves the weights of the best epoch (highest validation accuracy)
- **EarlyStopping** (patience 5) — stops when validation loss stops improving and restores the best weights
- **ReduceLROnPlateau** — lowers the learning rate when progress stalls

---

## Model Architecture

| # | Layer | Purpose |
|---|-------|---------|
| 1 | `MobileNetV2` (154 layers, ImageNet weights, top removed) | Feature extractor — transfer learning |
| 2 | `GlobalAveragePooling2D` | 7×7×1280 feature maps → 1280-length vector |
| 3 | `BatchNormalization` | Stable, faster training |
| 4 | `Dense(256, ReLU)` | Hidden layer — learns meme-sentiment patterns |
| 5 | `Dropout(0.4)` | Reduces overfitting |
| 6 | `Dense(3, Softmax)` | Output probabilities for the 3 classes |

Loss: categorical cross-entropy · Optimizer: Adam · Batch size: 32 · Image size: 224×224

---

## Results

| Metric | Value |
|--------|-------|
| Validation accuracy | **~54–57 %** (1,398 images) |
| Random-chance baseline (3 classes) | 33 % |
| Positive class F1-score | ≈ 0.69 |

The positive class is recognised well; neutral and negative memes are frequently mistaken for
positive. This is in line with published image-only results on Memotion. The two main reasons
are the **class imbalance** of the dataset (≈ 6 positive memes for every negative one) and the
fact that a meme's sentiment is often carried by its **caption text**, which an image-only model
cannot read. See `confusion_matrix.png` and `PROJECT_REPORT.pdf` for the full analysis.

---

## Project Structure

```
Meme_Sentiment_Classification/
│
├── app.py                          # Flask web application (port 5001)
├── config.py                       # Paths, image size, batch size, class names
├── requirements.txt                # Python dependencies
│
├── src/
│   ├── train.py                    # Builds the model and runs 2-phase training
│   ├── evaluate.py                 # Precision / recall / F1 + confusion matrix
│   └── predict.py                  # Loads the saved model, predicts one image
│
├── templates/index.html            # Web UI (drag-and-drop, results)
├── static/style.css                # Web UI styling
│
├── models/
│   ├── class_names.json            # Class order: ["negative", "neutral", "positive"]
│   └── meme_sentiment.keras        # Trained best weights  (created by train.py — not in git)
│
├── data/memes/                     # Dataset, sorted by class  (not in git — see below)
│   ├── positive/
│   ├── neutral/
│   └── negative/
│
├── uploads/                        # Temporary uploads from the web app
├── confusion_matrix.png            # Output of evaluate.py
│
├── README.md                       # This file
├── PROJECT_REPORT.pdf              # Full project report (IEEE references)
├── PROJECT_REPORT.md               # Editable source of the report
├── MODEL_MANUAL.pdf                # Beginner-to-pro explanation of the model
└── Meme_Sentiment_Presentation.pptx# Project presentation slides
```

> **Not included in this repository** (see `.gitignore`): the dataset (`data/memes/`, ~700 MB),
> the trained model file (`models/meme_sentiment.keras`, 25 MB) and the virtual environment.
> Follow the steps below to get them.

---

## Dataset — Where to Get It

This project uses the **Memotion 7k** dataset (SemEval-2020 Task 8), which is **not
redistributed here**. Download it yourself and follow its licence terms:

- **Kaggle:** https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k
- Files you get: `images/` (~7,000 memes) and `labels.csv`

The `labels.csv` has an `overall_sentiment` column with 5 values. This project merges them into
3 classes and puts each image into a class folder:

| `overall_sentiment` in labels.csv | Folder | Images |
|-----------------------------------|--------|--------|
| `very_positive`, `positive` | `data/memes/positive/` | 4,158 |
| `neutral` | `data/memes/neutral/` | 2,201 |
| `negative`, `very_negative` | `data/memes/negative/` | 631 |

You can sort them with a short script like this (run from the project root, after unzipping the
Kaggle archive somewhere):

```python
import pandas as pd, shutil
from pathlib import Path

src = Path("/path/to/memotion_dataset_7k")          # <-- change this
df  = pd.read_csv(src / "labels.csv")
mapping = {"very_positive": "positive", "positive": "positive", "neutral": "neutral",
           "negative": "negative", "very_negative": "negative"}

for _, row in df.iterrows():
    cls = mapping.get(str(row["overall_sentiment"]).strip())
    img = src / "images" / str(row["image_name"]).strip()
    if cls and img.exists():
        dst = Path("data/memes") / cls
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst / img.name)
```

---

## Installation & Setup

Requires **Python 3.10 – 3.12** (TensorFlow does not yet support newer versions).

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Meme_Sentiment_Classification.git
cd Meme_Sentiment_Classification

# 2. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Get the dataset (see section above) into data/memes/<class>/
```

---

## How to Run

All commands are run from the project root with the virtual environment activated.

**Train the model** (creates `models/meme_sentiment.keras`; ~15 min on a laptop CPU):
```bash
PYTHONPATH=. python src/train.py
```

**Evaluate** (prints precision / recall / F1 and saves `confusion_matrix.png`):
```bash
PYTHONPATH=. python src/evaluate.py
```

**Run the web app:**
```bash
python app.py
```
Then open **http://127.0.0.1:5001**, drop a meme and click **Predict Sentiment**.

> `PYTHONPATH=.` is needed because the scripts import `config.py` from the project root.
> Port 5001 is used because macOS reserves port 5000 for AirPlay.

---

## Documentation

| File | What it contains |
|------|------------------|
| `PROJECT_REPORT.pdf` | Abstract, introduction, literature survey, dataset, methodology, results, conclusion, IEEE references |
| `MODEL_MANUAL.pdf` | Beginner-to-pro guide: every layer, activation functions, how training works, how the best weights are chosen, how to improve accuracy |
| `Meme_Sentiment_Presentation.pptx` | 13-slide project presentation |

---

## Limitations & Future Work

- **Class imbalance** — the dataset is ~59 % positive, 31 % neutral, 9 % negative, so the model
  leans towards "positive". *Next step:* class-weighted training.
- **The joke is in the caption** — an image-only model cannot read the text that often carries
  the sentiment. *Next step:* OCR the caption (already available as `text_ocr` in `labels.csv`)
  and fuse a text model (LSTM / BERT) with the CNN — the most promising route to ~70 %.
- Fine-tune more backbone layers with a low learning rate, or try a stronger backbone
  (EfficientNet).

---

## Tech Stack

Python · TensorFlow / Keras · MobileNetV2 · NumPy · Pandas · scikit-learn · Matplotlib · Seaborn · Pillow · Flask

---

## References

1. C. Sharma *et al.*, "SemEval-2020 Task 8: Memotion Analysis — The Visuo-Lingual Metaphor!," *Proc. SemEval-2020*, 2020.
2. M. Sandler *et al.*, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," *Proc. IEEE/CVF CVPR*, 2018.
3. J. Deng *et al.*, "ImageNet: A Large-Scale Hierarchical Image Database," *Proc. IEEE CVPR*, 2009.
4. S. J. Pan and Q. Yang, "A Survey on Transfer Learning," *IEEE Trans. Knowledge and Data Engineering*, 2010.
5. D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," *Proc. ICLR*, 2015.

---

*Media & Entertainment mini project — Meme Sentiment Classification using Deep Learning.*
