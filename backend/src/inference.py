import os
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf

from backend.src.preprocess import preprocess_image

# Try multiple model paths for compatibility
_DEFAULT_MODEL_PATHS = [
    "/app/models/Fruit_classification_model.h5",
    "/app/models/fruit_model_clean.keras",
    "/app/models/Fruit_classification_model.keras",
    "/app/models/latest_model.keras",
]

MODEL_PATH_ENV = os.getenv("MODEL_PATH")
if MODEL_PATH_ENV:
    MODEL_PATH = Path(MODEL_PATH_ENV).resolve()
else:
    MODEL_PATH = Path(_DEFAULT_MODEL_PATHS[0]).resolve()

_model = None


def _build_legacy_sequential_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3), name="input_layer_3"),
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding="valid", activation="relu", name="conv2d_3"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", name="max_pooling2d_3"),
            tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding="valid", activation="relu", name="conv2d_4"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", name="max_pooling2d_4"),
            tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding="valid", activation="relu", name="conv2d_5"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", name="max_pooling2d_5"),
            tf.keras.layers.Flatten(name="flatten_1"),
            tf.keras.layers.Dense(128, activation="relu", name="dense_2"),
            tf.keras.layers.Dropout(0.5, name="dropout"),
            tf.keras.layers.Dense(1, activation="sigmoid", name="dense_3"),
        ]
    )
    model.build((None, 224, 224, 3))
    return model


def get_model():
    global _model
    if _model is None:
        # Try the configured path first, then fallback paths
        paths_to_try = [MODEL_PATH] + [Path(p).resolve() for p in _DEFAULT_MODEL_PATHS if Path(p).resolve() != MODEL_PATH]

        last_error = None
        for model_path in paths_to_try:
            if model_path.exists():
                try:
                    print(f"Loading model from {model_path}...")
                    if model_path.suffix.lower() in {".h5", ".hdf5"}:
                        _model = _build_legacy_sequential_model()
                        _model.load_weights(str(model_path))
                    else:
                        _model = tf.keras.models.load_model(str(model_path), compile=False)

                    print("✓ Model loaded successfully")
                    return _model
                except Exception as e:
                    print(f"Failed to load from {model_path}: {e}")
                    last_error = e
                    continue

        raise FileNotFoundError(
            f"Model file not found or could not be loaded. Tried: {[str(p) for p in paths_to_try if p.exists()]} "
            f"(existing paths). Last error: {last_error}"
        )
    return _model


def predict(image: Image.Image) -> dict:
    model = get_model()
    x = preprocess_image(image)
    y = np.array(model.predict(x, verbose=0))

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
