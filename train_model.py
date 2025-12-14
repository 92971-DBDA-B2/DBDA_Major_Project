# train_model.py
import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint   # EarlyStopping removed
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from collections import Counter
import argparse

# Config
IMG_SIZE = 100
DATA_DIR = "known_faces"
MODEL_PATH = "face_model.keras"
LABELS_PATH = "labels.npy"
EPOCHS = 50
BATCH_SIZE = 8
VALIDATION_SPLIT = 0.2
MIN_SAMPLES_PER_CLASS = 3  # small safeguard

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", help="Force retrain even if model exists")
args = parser.parse_args()

# Ensure data directory exists
if not os.path.exists(DATA_DIR):
    raise SystemExit(f"Data directory '{DATA_DIR}' not found. Create known_faces/ with subfolders first.")

# Load + sort categories for stable mapping
CATEGORIES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
if len(CATEGORIES) == 0:
    raise SystemExit("No person folders found in known_faces/. Add subfolders for each person.")

print("Detected categories:", CATEGORIES)

# If model exists and not forcing, skip training
if os.path.exists(MODEL_PATH) and not args.force:
    print(f"Model {MODEL_PATH} already exists. Use --force to retrain.")
    np.save(LABELS_PATH, CATEGORIES)
    print("Labels saved/updated.")
    raise SystemExit("Exiting without retraining.")

# Helper: detect face; fallback to whole image if no face found
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def load_dataset():
    X = []
    y = []
    for label, category in enumerate(CATEGORIES):
        folder = os.path.join(DATA_DIR, category)
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(files) == 0:
            print(f"Warning: no images in {folder}")
            continue
        for fname in files:
            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                x, y1, w, h = faces[0]
                face_img = gray[y1:y1+h, x:x+w]
            else:
                face_img = gray
            try:
                face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
            except Exception as e:
                continue
            X.append(face_img)
            y.append(label)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y

print("📥 Loading and preprocessing images...")
X, y = load_dataset()
print("Total samples found:", len(X))
if len(X) == 0:
    raise SystemExit("No usable face images found after preprocessing. Check known_faces/ images.")

# Check samples per class
counts = Counter(y)
for idx, c in enumerate(CATEGORIES):
    print(f"  {c}: {counts.get(idx,0)} samples")
    if counts.get(idx,0) < MIN_SAMPLES_PER_CLASS:
        print(f"  ⚠️ Warning: '{c}' has fewer than {MIN_SAMPLES_PER_CLASS} usable samples (model may be unstable).")

# Normalize + reshape
X = X / 255.0
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_cat = to_categorical(y, num_classes=len(CATEGORIES))

# Stratified split
try:
    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=VALIDATION_SPLIT, random_state=42, stratify=y)
except Exception:
    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=VALIDATION_SPLIT, random_state=42)

print("Train samples:", len(X_train), " Val samples:", len(X_val))

# Compute class weights
from sklearn.utils.class_weight import compute_class_weight
y_integers = np.argmax(y_cat, axis=1)
try:
    class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
    class_weights = {i: w for i, w in enumerate(class_weights)}
    print("Class weights:", class_weights)
except Exception:
    class_weights = None

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.08,
    height_shift_range=0.08,
    shear_range=0.05
)
datagen.fit(X_train)

# Build CNN
def build_model(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

print("🧠 Building model...")
model = build_model(len(CATEGORIES))

# ---------------------------
# EARLY STOPPING REMOVED
# ---------------------------
callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
]

print("📈 Training...")
steps = max(1, len(X_train) // BATCH_SIZE)
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    steps_per_epoch=steps,
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Save labels
np.save(LABELS_PATH, CATEGORIES)
print(f"✅ Labels saved to {LABELS_PATH}")

# Save model if checkpoint didn't fire
if not os.path.exists(MODEL_PATH):
    model.save(MODEL_PATH)

print(f"✅ Model trained & saved as {MODEL_PATH}")
