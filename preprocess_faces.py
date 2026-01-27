import cv2
import os

RAW_DIR = "known_faces_raw"
PROC_DIR = "known_faces_processed"
IMG_SIZE = 224

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

os.makedirs(PROC_DIR, exist_ok=True)

for person in os.listdir(RAW_DIR):
    raw_person_dir = os.path.join(RAW_DIR, person)
    proc_person_dir = os.path.join(PROC_DIR, person)

    if not os.path.isdir(raw_person_dir):
        continue

    os.makedirs(proc_person_dir, exist_ok=True)

    for img_name in os.listdir(raw_person_dir):
        img_path = os.path.join(raw_person_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            continue

        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        cv2.imwrite(os.path.join(proc_person_dir, img_name), face)

    print(f"Processed faces for {person}")

print("Preprocessing completed.")
