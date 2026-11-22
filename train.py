import os
import sys
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Sequential

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

column_names = [
    'center','left','right','steering','throttle','brake','speed'
]

# -------------------------------
# IMAGE PREPROCESS
# -------------------------------
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    return img

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return preprocess_image(img)

# -------------------------------
# LOAD DATA
# -------------------------------
data = pd.read_csv(
    os.path.join(BASE_DIR,'data','driving_log.csv'),
    names=column_names
)

data.columns = data.columns.str.strip()
data['steering'] = pd.to_numeric(data['steering'], errors='coerce')
data = data.dropna(subset=['steering'])

# -------------------------------
# 🔥 BALANCE DATA (IMPORTANT FIX)
# -------------------------------
# separate straight and turning
straight = data[abs(data['steering']) <= 0.03]
turns = data[abs(data['steering']) > 0.03]

# reduce straight driving (keep only 20%)
straight = straight.sample(frac=0.2, random_state=42)

# combine again
data = pd.concat([straight, turns])

print("Balanced dataset size:", len(data))

# -------------------------------
# LOAD IMAGES
# -------------------------------
images = []
angles = []

camera_offsets = {
    'center': 0.0,
    'left': 0.18,
    'right': -0.18
}

for i, row in data.iterrows():

    if i % 500 == 0:
        print("Loading image:", i)

    base = float(row['steering'])

    for cam in ['center','left','right']:
        img_path = os.path.basename(row[cam])
        full_path = os.path.join(BASE_DIR,"data","IMG",img_path)

        img = load_image(full_path)
        if img is None:
            continue

        steering = base + camera_offsets[cam]
        steering = np.clip(steering, -1.0, 1.0)

        images.append(img)
        angles.append(steering)

        # flip augmentation
        images.append(cv2.flip(img,1))
        angles.append(-steering)

print("Total images:", len(images))

# -------------------------------
# NUMPY
# -------------------------------
X = np.array(images, dtype=np.float32) / 255.0
y = np.array(angles, dtype=np.float32)

# -------------------------------
# SPLIT
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# MODEL
# -------------------------------
model = Sequential([
    Input(shape=(66,200,3)),
    Conv2D(24,(5,5),strides=(2,2),activation='relu'),
    Conv2D(36,(5,5),strides=(2,2),activation='relu'),
    Conv2D(48,(5,5),strides=(2,2),activation='relu'),
    Conv2D(64,(3,3),activation='relu'),
    Conv2D(64,(3,3),activation='relu'),
    Flatten(),
    Dense(100,activation='relu'),
    Dense(50,activation='relu'),
    Dense(10,activation='relu'),
    Dense(1, activation='tanh')  # 🔥 CRITICAL: tanh constrains output to [-1, 1]
])

model.compile(loss='mean_squared_error', optimizer='adam')

# -------------------------------
# TRAIN
# -------------------------------
print("🚀 Training started...")

model.fit(
    X_train, y_train,
    validation_data=(X_val,y_val),
    epochs=8,
    batch_size=32,
    verbose=1
)

# -------------------------------
# SAVE
# -------------------------------
os.makedirs("models", exist_ok=True)
model.save("models/model.h5")

print("✅ Training complete!")