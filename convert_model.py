import tensorflow as tf
from pathlib import Path

# Load the old model
model_path = Path("models/Fruit_classification_model.keras")
print(f"Loading model from {model_path}...")

try:
    # Try loading with TF 2.8.0
    model = tf.keras.models.load_model(str(model_path), compile=False)
    print("✓ Model loaded successfully")
    
    # Resave it
    model.save(str(model_path))
    print(f"✓ Model resaved to {model_path}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("Try running this script inside the Docker container instead:")
    print("  docker run --rm -v $(pwd):/app tensorflow/tensorflow:2.8.0 python /app/convert_model.py")
