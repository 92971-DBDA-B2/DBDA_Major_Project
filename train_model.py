import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import json

DATASET_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
MODEL_PATH = "face_recognition_mobilenetv2.h5"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

num_classes = train_gen.num_classes

with open("class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# 🔓 Fine-tune last 30 layers
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

model.save(MODEL_PATH)
print("Model saved:", MODEL_PATH)
