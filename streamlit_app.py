import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
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

st.set_page_config(page_title="Face Attendance System", layout="centered")
st.title("🧠 Face Recognition Attendance System")

# Session attendance storage
if "attendance" not in st.session_state:
    st.session_state.attendance = []

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Avoid duplicates in same session
    for row in st.session_state.attendance:
        if row["Name"] == name and row["Date"] == date:
            return False

    st.session_state.attendance.append({
        "Name": name,
        "Date": date,
        "Time": time
    })
    return True


menu = st.sidebar.radio("Menu", ["Mark Attendance", "Download Attendance"])

# -------------------------------
# MARK ATTENDANCE
# -------------------------------
if menu == "Mark Attendance":

    if st.button("Start Camera"):
        cap = cv2.VideoCapture(0)
        marked_today = set()

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

                emb = model.predict(face, verbose=0)
                emb = emb / np.linalg.norm(emb)

                sims = cosine_similarity(emb, embeddings)
                idx = np.argmax(sims)
                score = sims[0][idx]

                if score > THRESHOLD:
                    name = names[idx]
                    if name not in marked_today:
                        if mark_attendance(name):
                            marked_today.add(name)
                else:
                    name = "Unknown"

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(
                    frame,
                    f"{name} ({score:.2f})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )

            cv2.imshow("Attendance Camera (Press Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

# -------------------------------
# DOWNLOAD ATTENDANCE
# -------------------------------
elif menu == "Download Attendance":

    if len(st.session_state.attendance) == 0:
        st.warning("No attendance recorded in this session.")
    else:
        df = pd.DataFrame(st.session_state.attendance)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        excel_name = f"attendance_{timestamp}.xlsx"
        csv_name = f"attendance_{timestamp}.csv"

        st.subheader("📄 Attendance Preview")
        st.dataframe(df)

        # CSV download
        st.download_button(
            label="⬇️ Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=csv_name,
            mime="text/csv"
        )

        # Excel download
        st.download_button(
            label="⬇️ Download Excel",
            data=df.to_excel(excel_name, index=False),
            file_name=excel_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
