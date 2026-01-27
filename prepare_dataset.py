import os
import shutil

PROC_DIR = "known_faces_processed"
AUG_DIR = "augmented_data"
DATASET_DIR = "dataset"

os.makedirs(DATASET_DIR, exist_ok=True)

def copy_images(src_dir):
    for person in os.listdir(src_dir):
        src_person = os.path.join(src_dir, person)
        dst_person = os.path.join(DATASET_DIR, person)

        if not os.path.isdir(src_person):
            continue

        os.makedirs(dst_person, exist_ok=True)

        for img in os.listdir(src_person):
            shutil.copy(
                os.path.join(src_person, img),
                os.path.join(dst_person, img)
            )

copy_images(PROC_DIR)
copy_images(AUG_DIR)

print("Final dataset prepared successfully.")
