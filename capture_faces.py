# capture_faces.py
import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

name = input("Enter your name: ").strip()
if not name:
    print("❌ Please enter a valid name")
    exit()

path = os.path.join("known_faces", name)
os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = len(os.listdir(path))
print("📸 Press SPACE to capture | ESC to stop (captures cropped faces if detected)")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(display, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Capture - Press SPACE", display)
    key = cv2.waitKey(1)
    if key % 256 == 27:  # ESC
        break
    elif key % 256 == 32:  # SPACE
        count += 1
        filename = f"{path}/{count}.jpg"
        # crop if face found, else save full frame
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]
            cv2.imwrite(filename, face)
        else:
            cv2.imwrite(filename, frame)
        print(f"✅ Saved: {filename}")
        if count >= 50:
            print("✅ Captured 50 images. Exiting...")
            break

cap.release()
cv2.destroyAllWindows()
