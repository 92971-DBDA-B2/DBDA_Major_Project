import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
from datetime import datetime
from tensorflow.keras.models import load_model

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 224
CONF_THRESHOLD = 0.5
IMAGE_PATH = "What-is-Facial-Recognition.webp"

GREEN = (0, 200, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

st.set_page_config(
    page_title="Face Attendance System",
    page_icon="📸",
    layout="wide"
)

# -----------------------------
# LOAD MODEL & LABELS
# -----------------------------
model = load_model("face_recognition_mobilenetv2.h5")

with open("class_indices.json") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# HERO SECTION
# -----------------------------
st.markdown(
    """
    <div style="text-align:center; padding:30px">
        <h1 style="font-size:48px">📸 Face Attendance System</h1>
        <p style="font-size:20px; color:gray">
            AI-powered attendance using MobileNetV2
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(IMAGE_PATH, use_container_width=True)

st.divider()

# -----------------------------
# SIDEBAR
# -----------------------------
menu = st.sidebar.radio(
    "Choose Section",
    ["🏫 Mark Attendance", "📥 Download Attendance"]
)

# -----------------------------
# SESSION STATE
# -----------------------------
if "attendance" not in st.session_state:
    st.session_state.attendance = []

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    for row in st.session_state.attendance:
        if row["Name"] == name and row["Date"] == date:
            return

    st.session_state.attendance.append({
        "Name": name,
        "Date": date,
        "Time": time
    })

# -----------------------------
# MARK ATTENDANCE
# -----------------------------
if menu == "🏫 Mark Attendance":

    if st.button("▶️ Start Camera"):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                face = face.astype("float32") / 255.0
                face = np.expand_dims(face, axis=0)

                preds = model.predict(face, verbose=0)
                idx = np.argmax(preds)
                confidence = preds[0][idx]

                if confidence > CONF_THRESHOLD:
                    name = labels[idx]
                    color = GREEN
                    mark_attendance(name)
                    label = f"{name}  {confidence:.2f}"
                else:
                    name = "UNKNOWN"
                    color = RED
                    label = f"UNKNOWN  {confidence:.2f}"

                # Face box
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                # Text background
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(
                    frame,
                    (x, y - th - 15),
                    (x + tw + 10, y),
                    color,
                    -1
                )

                # Text
                cv2.putText(
                    frame,
                    label,
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    WHITE,
                    2,
                    cv2.LINE_AA
                )

            cv2.imshow("Face Attendance (Press Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

# -----------------------------
# DOWNLOAD ATTENDANCE
# -----------------------------
elif menu == "📥 Download Attendance":

    if len(st.session_state.attendance) == 0:
        st.warning("No attendance recorded.")
    else:
        df = pd.DataFrame(st.session_state.attendance)
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "⬇️ Download CSV",
            df.to_csv(index=False),
            "attendance.csv"
        )

st.divider()

# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    """
    <div style="text-align:center; color:gray; padding:20px">
        <b>Face Recognition Attendance System</b><br>
        Built with Streamlit • OpenCV • MobileNetV2
    </div>
    """,
    unsafe_allow_html=True
)
