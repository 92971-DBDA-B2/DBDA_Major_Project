# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model

# CONFIG
MODEL_PATH = "face_model.keras"
LABELS_PATH = "labels.npy"
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.xlsx"
IMG_SIZE = 100
CONF_THRESHOLD = 0.85  # higher threshold to reduce false positives

# Load model and labels
model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    st.warning("Model not found. Please run train_model.py first to create face_model.keras")

if os.path.exists(LABELS_PATH):
    CATEGORIES = list(np.load(LABELS_PATH, allow_pickle=True))
else:
    # fallback: sorted directory listing (less safe)
    CATEGORIES = sorted([d for d in os.listdir(KNOWN_FACES_DIR) if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Helper functions
def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_excel(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    # avoid duplicates for same person and same date
    exists = ((df["Name"] == name) & (df["Date"] == date)).any()
    if exists:
        print(f"Attendance already exists for {name} today.")
        return False

    new_row = pd.DataFrame({"Name": [name], "Date": [date], "Time": [time]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_excel(ATTENDANCE_FILE, index=False)
    print(f"📝 Attendance marked for {name}")
    return True

def preprocess_face(face_img):
    # input: grayscale face image or color image
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_img = face_img.astype("float32") / 255.0
    face_img = face_img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return face_img

# Streamlit UI
st.set_page_config(page_title="Face Attendance CNN", page_icon="🧠", layout="centered")
st.title("🧠 CNN-based Face Recognition Attendance System")

menu = st.sidebar.radio("Menu", ["🏠 Home", "📸 Mark Attendance", "📄 View Attendance"])

if menu == "🏠 Home":
    st.write("""
    ### Welcome!
    - If you already captured faces into `known_faces/<name>/` folders, run `train_model.py` once.
    - This app loads `face_model.keras` and `labels.npy` to mark attendance.
    - If you retrain the model, re-run the Streamlit app to pick up the new model/labels.
    """)

elif menu == "📸 Mark Attendance":
    if model is None:
        st.warning("⚠️ No trained model found. Please run train_model.py first.")
    else:
        st.subheader("🧠 Start Recognition")
        if st.button("Start Camera"):
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Cannot access camera.")
            else:
                st.info("Press Q in the window to stop.")
                # Keep track of which names we've already marked in this session (and check file for today's marks too)
                marked_session = set()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    for (x, y, w, h) in faces:
                        face_img = gray[y:y+h, x:x+w]
                        face_input = preprocess_face(face_img)
                        preds = model.predict(face_input, verbose=0)
                        label_index = int(np.argmax(preds))
                        confidence = float(np.max(preds))

                        if confidence >= CONF_THRESHOLD:
                            name = CATEGORIES[label_index] if label_index < len(CATEGORIES) else "Unknown"
                        else:
                            name = "Unknown"

                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, f"{name} ({confidence*100:.1f}%)", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                        if name != "Unknown" and name not in marked_session:
                            # also check attendance file to avoid duplicate marking today
                            marked = mark_attendance(name)
                            if marked:
                                marked_session.add(name)

                    cv2.imshow("Face Attendance (press q to quit)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()

elif menu == "📄 View Attendance":
    st.subheader("📋 Attendance Records")
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_excel(ATTENDANCE_FILE)
        st.dataframe(df)
        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode('utf-8'),
                           "attendance.csv", "text/csv")
    else:
        st.info("No attendance records found.")
