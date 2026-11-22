import os
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.h5')

print(f"Model path: {MODEL_PATH}")
print(f"Model exists: {os.path.exists(MODEL_PATH)}")

try:
    model = load_model(MODEL_PATH)
    print("✓ Model loaded successfully")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    
    # Test with dummy input
    dummy_input = np.random.randn(1, 66, 200, 3)
    prediction = model.predict(dummy_input, verbose=0)
    print(f"✓ Model prediction works: {prediction}")
    print(f"  Steering angle: {float(prediction)}")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")

# Test throttle calculation
print("\n--- Testing Throttle Calculation ---")
speed_limit = 10
for speed in [0, 2, 5, 10, 15]:
    throttle = float(np.clip(1.0 - speed / speed_limit, 0.0, 1.0))
    print(f"Speed: {speed:2} → Throttle: {throttle:.2f}")
