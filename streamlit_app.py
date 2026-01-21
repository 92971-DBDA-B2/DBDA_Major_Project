import streamlit as st
import cv2
import numpy as np
import pandas as pd
import io
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 100
THRESHOLD = 0.5
IMAGE_PATH = "What-is-Facial-Recognition.webp"

st.set_page_config(
    page_title="Face Attendance System",
    page_icon="📸",
    layout="wide"
)

# -----------------------------
# LOAD MODEL & DATA
# -----------------------------
model = load_model("face_embedding_model.keras")
embeddings = np.load("embeddings/embeddings.npy")
names = np.load("embeddings/names.npy")

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
            AI-powered attendance using face recognition and deep learning
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Centered Image
img_col1, img_col2, img_col3 = st.columns([1, 2, 1])
with img_col2:
    st.image(IMAGE_PATH, use_container_width=True)

st.divider()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📂 Navigation")
menu = st.sidebar.radio(
    "Choose Section",
    ["🏫 Mark Attendance", "📥 Download Attendance"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ System Overview")
st.sidebar.write("✔ Embedding Model Loaded")
st.sidebar.write(f"✔ Total Registered Faces: {len(names)}")

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
            return False

    st.session_state.attendance.append({
        "Name": name,
        "Date": date,
        "Time": time
    })
    return True

# -----------------------------
# MARK ATTENDANCE
# -----------------------------
if menu == "🏫 Mark Attendance":

    left, right = st.columns([3, 2])

    with left:
        st.subheader("🎥 Live Face Recognition")
        st.write("Click **Start Camera** to mark attendance")

        if st.button("▶️ Start Camera"):
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

    with right:
        st.subheader("📊 Session Summary")
        st.metric("Registered Faces", len(names))
        st.metric("Attendance Marked", len(st.session_state.attendance))

        st.markdown(
            """
            ### 🧠 How It Works
            - Face detection using OpenCV  
            - Deep model generates embeddings  
            - Cosine similarity for matching  
            - Attendance stored per session  
            """
        )

# -----------------------------
# DOWNLOAD ATTENDANCE
# -----------------------------
elif menu == "📥 Download Attendance":

    st.subheader("📄 Attendance Records")

    if len(st.session_state.attendance) == 0:
        st.warning("No attendance recorded in this session.")
    else:
        df = pd.DataFrame(st.session_state.attendance)
        st.dataframe(df, use_container_width=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        excel_name = f"attendance_{timestamp}.xlsx"
        csv_name = f"attendance_{timestamp}.csv"

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="⬇️ Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=csv_name,
                mime="text/csv"
            )

        with col2:
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine="openpyxl")
            excel_buffer.seek(0)

            st.download_button(
                label="⬇️ Download Excel",
                data=excel_buffer,
                file_name=excel_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

st.divider()

# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    """
    <div style="text-align:center; color:gray; padding:20px">
        <b>Face Recognition Attendance System</b><br>
        Built with Streamlit • OpenCV • Deep Learning
    </div>
    """,
    unsafe_allow_html=True
)
