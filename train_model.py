
import tensorflow as tf
import json
from tensorflow.keras import layers, models

# Dataset location
train_path = "dataset/train"
val_path = "dataset/validation"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15

# Load training dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["fresh", "rotten"]
)

# Load validation dataset
val_data = tf.keras.utils.image_dataset_from_directory(
    val_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["fresh", "rotten"]
)

# Save class names
with open("classes.json", "w") as f:
    json.dump(train_data.class_names, f)

# Improve data performance
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(AUTOTUNE)
val_data = val_data.prefetch(AUTOTUNE)

# CNN Model
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1./255),

    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Save model
model.save("fruit_model.keras")

print("CNN Model Training Completed!")
print("Model saved successfully!")
