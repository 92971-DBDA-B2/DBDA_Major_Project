import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

PROC_DIR = "known_faces_processed"
AUG_DIR = "augmented_data"
IMG_SIZE = 224
AUG_PER_IMAGE = 5

datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

os.makedirs(AUG_DIR, exist_ok=True)

for person in os.listdir(PROC_DIR):
    person_dir = os.path.join(PROC_DIR, person)
    aug_person_dir = os.path.join(AUG_DIR, person)

    if not os.path.isdir(person_dir):
        continue

    os.makedirs(aug_person_dir, exist_ok=True)

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.reshape((1,) + img.shape)

        i = 0
        for batch in datagen.flow(img, batch_size=1):
            aug_img = batch[0].astype("uint8")
            aug_name = f"aug_{i}_{img_name}"
            cv2.imwrite(os.path.join(aug_person_dir, aug_name), aug_img)
            i += 1
            if i >= AUG_PER_IMAGE:
                break

    print(f"Augmented images created for {person}")

print("Augmentation completed.")
