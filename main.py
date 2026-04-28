import os

DATASET_PATH = "bloodcells_subset"

print("Folders found:")
print(os.listdir(DATASET_PATH))

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_data.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE

# Data augmentation to help with overfitting and improve generalization
# Applied only to train dataset
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.3),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.2),
    layers.RandomTranslation(0.1, 0.1),
], name="data_augmentation")

train_data = train_data.cache().shuffle(5000).map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
).prefetch(buffer_size=AUTOTUNE)

val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)

weight_decay = 1e-3

# Model has convolutional layers with L2 regularization and batch normalization
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1./255),

    layers.Conv2D(
        16, (3, 3), activation="relu",
        kernel_regularizer=regularizers.l2(weight_decay)
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(
        32, (3, 3), activation="relu",
        kernel_regularizer=regularizers.l2(weight_decay)
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(
        64, (3, 3), activation="relu",
        kernel_regularizer=regularizers.l2(weight_decay)
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),

    layers.GlobalAveragePooling2D(),
    layers.Dense(
        64, activation="relu",
        kernel_regularizer=regularizers.l2(weight_decay)
    ),
    layers.Dropout(0.6),

    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50
)

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

val_loss, val_accuracy = model.evaluate(val_data)
print("Validation Accuracy:", val_accuracy)

y_true = []
y_pred = []

for images, labels in val_data:
    predictions = model.predict(images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

model.save("blood_cell_cnn_model.keras")
print("Model saved as blood_cell_cnn_model.keras")