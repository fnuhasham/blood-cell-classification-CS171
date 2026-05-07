import os
import gc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

tf.keras.backend.clear_session()
gc.collect()

DATASET_PATH = "bloodcells_dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

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

total_size = temp_data.cardinality().numpy()
val_data = temp_data.take(total_size // 2)
test_data = temp_data.skip(total_size // 2)

class_names = train_data.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.shuffle(500).prefetch(AUTOTUNE)
val_data = val_data.prefetch(AUTOTUNE)
test_data = test_data.prefetch(AUTOTUNE)

# ---- Data Augmentation ----
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
])

# ---- Base Model ----
base_model = ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# ---- Model ----
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    data_augmentation,
    layers.Lambda(lambda x: tf.keras.applications.resnet50.preprocess_input(x)),

    base_model,

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# ---- Initial Training ----
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[early_stopping]
)

# ---- Fine-tuning ----
print("Starting fine-tuning...")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
    callbacks=[early_stopping]
)

# ---- Evaluate ----
test_loss, test_accuracy = model.evaluate(test_data)
print("Test Accuracy:", test_accuracy)

y_true = []
y_pred = []

for images, labels in test_data:
    preds = model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(pred_labels)

# ---- Confusion Matrix ----
print("Confusion Matrix:")
ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    display_labels=class_names
)

plt.xticks(rotation=45, ha="right")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ---- Sample Predictions ----
def to_displayable(img):
    img = img - img.min()
    img = img / img.max()
    return img

correct_imgs, correct_true = [], []
incorrect_imgs, incorrect_true, incorrect_pred = [], [], []

for images, labels in test_data:
    preds = model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    for i in range(len(labels)):
        if pred_labels[i] == labels[i] and len(correct_imgs) < 4:
            correct_imgs.append(images[i].numpy())
            correct_true.append(labels[i].numpy())
        elif pred_labels[i] != labels[i] and len(incorrect_imgs) < 4:
            incorrect_imgs.append(images[i].numpy())
            incorrect_true.append(labels[i].numpy())
            incorrect_pred.append(pred_labels[i])

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("Test Predictions", fontsize=14)

for col in range(4):
    axes[0, col].imshow(to_displayable(correct_imgs[col]))
    axes[0, col].set_title(f"✓ {class_names[correct_true[col]]}", color="green", fontsize=9)
    axes[0, col].axis("off")

    axes[1, col].imshow(to_displayable(incorrect_imgs[col]))
    axes[1, col].set_title(
        f"✗ pred: {class_names[incorrect_pred[col]]}\ntrue: {class_names[incorrect_true[col]]}",
        color="red",
        fontsize=9
    )
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("Correct")
axes[1, 0].set_ylabel("Incorrect")

plt.tight_layout()
plt.show()

# ---- Save Model ----
model.save("/content/drive/MyDrive/blood_cell_resnet50_model.keras")