#!/usr/bin/env python3
"""Convert .keras to clean .keras format compatible with TF 2.8"""
import tensorflow as tf
import os
import shutil

keras_path = "/app/models/Fruit_classification_model.keras"
output_path = "/app/models/fruit_model_clean.keras"

print(f"Loading {keras_path}...")

try:
    # Load with latest TF
    model = tf.keras.models.load_model(keras_path)
    print("✓ Model loaded")
    
    # Save as clean .keras
    model.save(output_path)
    print(f"✓ Saved to {output_path}")
    
    # Verify it loads
    model2 = tf.keras.models.load_model(output_path)
    print("✓ Verification: model reloaded successfully")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
