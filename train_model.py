import os
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import normalize

DATA_DIR = "known_faces_processed"
IMG_SIZE = 100
EMB_SIZE = 128

X, names = [], []

for person in sorted(os.listdir(DATA_DIR)):
    person_dir = os.path.join(DATA_DIR, person)
    if not os.path.isdir(person_dir):
        continue

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = img.astype("float32") / 255.0
        img = img.reshape(IMG_SIZE, IMG_SIZE, 1)

        X.append(img)
        names.append(person)

X = np.array(X)

# Embedding CNN
inp = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
x = Conv2D(32, 3, activation="relu")(inp)
x = MaxPooling2D()(x)
x = Conv2D(64, 3, activation="relu")(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
emb = Dense(EMB_SIZE)(x)

model = Model(inp, emb)
model.compile(optimizer=Adam(0.001), loss="mse")

# Self-supervised embedding stabilization
model.fit(X, model.predict(X), epochs=10, batch_size=16)

embeddings = model.predict(X)
embeddings = normalize(embeddings)

os.makedirs("embeddings", exist_ok=True)
np.save("embeddings/embeddings.npy", embeddings)
np.save("embeddings/names.npy", np.array(names))

model.save("face_embedding_model.keras")
print("Embedding model trained and saved.")
