import cv2
import os

RAW_DIR = "known_faces_raw"
MAX_IMAGES = 50

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

name = input("Enter person name: ").strip()
if not name:
    print("Invalid name")
    exit()

person_dir = os.path.join(RAW_DIR, name)
os.makedirs(person_dir, exist_ok=True)

count = len(os.listdir(person_dir))
if count >= MAX_IMAGES:
    print("Already reached 50 images.")
    exit()

cap = cv2.VideoCapture(0)
print("Press SPACE to capture | ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Capture RAW Faces", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break

    elif key == 32 and len(faces) > 0:
        if count >= MAX_IMAGES:
            print("Reached max limit (50).")
            break

        count += 1
        cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), frame)
        print(f"Saved RAW image {count}/50")

cap.release()
cv2.destroyAllWindows()