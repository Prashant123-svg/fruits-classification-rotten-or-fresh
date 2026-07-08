import os
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
from src.preprocess import preprocess_image

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / ".." / "models" / "latest_model.keras"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).resolve()

_model = None
def get_model():
  global _model
  if _model is None:
    _model = tf.keras.models.load_model(str(MODEL_PATH))
  return _model
   
def predict(image:Image.Image)->dict:
  model = get_model()
  x = preprocess_image(image)
  y = np.array(model.predict(x))

  if y.ndim == 2 and y.shape[1] == 1:
    score = float(y[0][0])
    label = "rotten" if score > 0.5 else "fresh"
    confidence = score if label == "rotten" else 1.0 - score
    pretty_label = "Rotten" if label == "rotten" else "Fresh"
    return {
      "Label": label,
      "confidence": float(confidence),
      "raw-score": score,
      "prediction": pretty_label,
      "prediction_score": float(confidence),
    }

  probs = y[0].astype(float)
  idx = int(np.argmax(probs))
  return {
    "Label": str(idx),
    "confidence": float(probs[idx]),
    "raw-score": probs.tolist(),
    "prediction": str(idx),
    "prediction_score": float(probs[idx]),
  }