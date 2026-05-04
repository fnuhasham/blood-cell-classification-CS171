import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

DATASET_PATH = "bloodcells_dataset"
IMG_SIZE = (160, 160)
BATCH_SIZE = 16
SEED = 42

print("Folders found:")
print(os.listdir(DATASET_PATH))

# Split Data into 80/10/10
train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

temp_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

temp_batches = tf.data.experimental.cardinality(temp_data).numpy()
val_data = temp_data.take(temp_batches // 2)
test_data = temp_data.skip(temp_batches // 2)

class_names = train_data.class_names
print("Classes:", class_names)

# Use class weights from training labels
train_labels = np.concatenate([y.numpy() for _, y in train_data])
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# Cache and Prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)
test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)

# Data augmentation to help with overfitting and improve generalization
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1),
], name="data_augmentation")

weight_decay = 1e-4

# Model has convolutional layers with L2 regularization and batch normalization
model = models.Sequential([
    layers.Input(shape=(160, 160, 3)),
    data_augmentation,
    layers.Rescaling(1./255),

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

    layers.Conv2D(
        128, (3, 3), activation="relu",
        kernel_regularizer=regularizers.l2(weight_decay)
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(
        256, (3, 3), activation="relu",
        kernel_regularizer=regularizers.l2(weight_decay)
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),

    layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(weight_decay)),
    layers.BatchNormalization(),
    layers.Dropout(0.6),

    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stopping],
    class_weight=class_weight_dict
)

# Show Accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

test_loss, test_accuracy = model.evaluate(test_data)
print("Test Accuracy:", test_accuracy)

y_true = []
y_pred = []

for images, labels in test_data:
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