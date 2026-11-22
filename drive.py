import base64
import os
from io import BytesIO

import cv2
import eventlet
import eventlet.wsgi
import numpy as np
import socketio
from flask import Flask
from PIL import Image
from tensorflow.keras.models import load_model

# -------------------------------
# SETUP
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.h5')

sio = socketio.Server()
app = Flask(__name__)

print('Loading model from:', MODEL_PATH)
model = load_model(MODEL_PATH, compile=False)

steering_smoother = 0.0
alpha = 0.7  # Strong smoothing to prevent jerks
offset_smoother = 0.0
offset_alpha = 0.8

# Last steering value to detect if user stopped
last_steering = 0.0

# -------------------------------
# IMAGE PREPROCESS
# -------------------------------
def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    return img / 255.0

# -------------------------------
# LANE OFFSET (SIMPLE & ROBUST)
# -------------------------------
def lane_offset(img):
    try:
        h, w = img.shape[:2]
        crop = img[int(h * 0.7):]  # Look at bottom of image
        
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        
        # Simple edge detection - look for road edges (bright/dark transitions)
        edges = cv2.Canny(gray, 100, 200)
        
        # Get columns with most edge pixels (likely road boundaries)
        edge_cols = np.sum(edges, axis=0)
        
        # Find peaks in edge columns (left and right boundaries)
        threshold = np.mean(edge_cols) + np.std(edge_cols)
        edge_positions = np.where(edge_cols > threshold)[0]
        
        if len(edge_positions) > 0:
            # Road center is middle of detected edges
            road_center = int(np.mean(edge_positions))
            img_center = w // 2
            
            # Calculate offset (-1 to +1)
            offset = (road_center - img_center) / img_center
            
            # Only return if offset is small enough (avoid noise)
            if abs(offset) < 0.5:
                return offset
        
        return None
    except:
        return None

# -------------------------------
# TELEMETRY
# -------------------------------
@sio.on('telemetry')
def telemetry(sid, data):
    global steering_smoother, offset_smoother, last_steering

    if data is None:
        return

    speed = float(data.get('speed', 0.0))

    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)

    # MODEL PREDICTION (Trust the model!)
    processed = img_preprocess(image)
    processed = np.array([processed])
    prediction = model.predict(processed, verbose=0)
    steering = float(prediction[0][0])  # Output is already [-1, 1] from tanh
    
    # Scale to driving range - INCREASED for better curve handling
    steering = steering * 0.25

    # HEAVY smoothing to prevent zigzag and jerky movements
    steering_smoother = alpha * steering_smoother + (1 - alpha) * steering
    steering = steering_smoother

    # VERY WEAK lane correction - only for fine-tuning, not for main control
    offset = lane_offset(image)

    if offset is not None:
        # Heavy smoothing on offset
        offset_smoother = offset_alpha * offset_smoother + (1 - offset_alpha) * offset
        
        # STRONGER correction for curves - if offset is large, we're in a curve!
        if abs(offset_smoother) > 0.06:  # Curve detected
            # More aggressive correction for curves
            correction = -offset_smoother * 0.25
            steering = steering * 0.8 + correction * 0.2
        elif abs(offset_smoother) > 0.02:  # Slight drift
            # Gentle correction for small drifts
            correction = -offset_smoother * 0.15
            steering = steering + correction * 0.05
    
    # LIMIT to safe steering range
    steering = float(np.clip(steering, -0.25, 0.25))

    # SPEED CONTROL
    throttle = 0.12

    if abs(steering) > 0.15:
        throttle = 0.08
    elif abs(steering) > 0.1:
        throttle = 0.1
    
    if speed < 1:
        throttle = 0.15
    
    throttle = float(np.clip(throttle, 0.05, 0.15))

    print("Steering:", round(steering, 3),
          "Throttle:", round(throttle, 3),
          "Speed:", round(speed, 2),
          "Offset:", round(offset, 3) if offset is not None else "None")

    send_control(steering, throttle)

# -------------------------------
# CONNECT
# -------------------------------
@sio.on('connect')
def connect(sid, environ):
    print('Connected to simulator')
    send_control(0, 0.2)

# -------------------------------
# SEND CONTROL
# -------------------------------
def send_control(steering_angle, throttle):
    sio.emit(
        'steer',
        data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        }
    )

# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == '__main__':
    app = socketio.WSGIApp(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)