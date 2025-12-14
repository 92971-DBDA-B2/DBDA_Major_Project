import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

IMG_SIZE = 100
THRESHOLD = 0.5

model = load_model("face_embedding_model.keras")
embeddings = np.load("embeddings/embeddings.npy")
names = np.load("embeddings/names.npy")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.title("Face Recognition (Embedding Based)")

if st.button("Start Camera"):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = face.astype("float32") / 255.0
            face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)

            emb = model.predict(face)
            emb = emb / np.linalg.norm(emb)

            sims = cosine_similarity(emb, embeddings)
            idx = np.argmax(sims)
            score = sims[0][idx]

            if score > THRESHOLD:
                name = names[idx]
            else:
                name = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{name} ({score:.2f})",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
