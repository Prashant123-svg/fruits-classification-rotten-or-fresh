#!/usr/bin/env python3
"""Convert .keras model to .h5 format for compatibility"""
import tensorflow as tf
import sys

keras_path = "/app/models/Fruit_classification_model.keras"
h5_path = "/app/models/Fruit_classification_model.h5"

print(f"Converting {keras_path} to {h5_path}...")

try:
    # Load with newer TF that can read the file
    model = tf.keras.models.load_model(keras_path)
    print("✓ Model loaded")
    
    # Save as H5
    model.save(h5_path)
    print(f"✓ Model saved to {h5_path}")
    
except Exception as e:
    print(f"✗ Conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
