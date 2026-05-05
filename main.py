import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
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

total_size = temp_data.cardinality().numpy()
val_data = temp_data.take(total_size // 2)
test_data = temp_data.skip(total_size // 2)
print("Validation Images: ", val_data.cardinality().numpy() * BATCH_SIZE)
print("Test Images: ", test_data.cardinality().numpy() * BATCH_SIZE)

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

# Show Loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()

test_loss, test_accuracy = model.evaluate(test_data)
print("Test Accuracy:", test_accuracy)

y_true = []
y_pred = []

correct_imgs, correct_true = [], []
incorrect_imgs, incorrect_true, incorrect_pred = [], [], []

for images, labels in test_data:
    predictions = model.predict(images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = labels.numpy()

    y_true.extend(true_labels)
    y_pred.extend(predicted_labels)

    for i in range(len(labels.numpy())):
        if predicted_labels[i] == true_labels[i] and len(correct_imgs) < 4:
            correct_imgs.append(images[i].numpy())
            correct_true.append(true_labels[i])
        elif predicted_labels[i] != true_labels[i] and len(incorrect_imgs) < 4:
            incorrect_imgs.append(images[i].numpy())
            incorrect_true.append(true_labels[i])
            incorrect_pred.append(predicted_labels[i])

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))

print("\nConfusion Matrix:")
disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names)
disp.plot(cmap='Blues')
plt.xticks(rotation = 45, ha='right')
plt.tight_layout()
plt.show()

def to_displayable(img):
    img = img - img.min()  # shift so min is 0
    img = img / img.max()  # scale so max is 1
    return img

# Print Sample Predictions
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle('Test Predictions', fontsize=14)

for col in range(4):
    axes[0, col].imshow(to_displayable(correct_imgs[col]))
    axes[0, col].set_title(f'✓ {class_names[correct_true[col]]}', color='green', fontsize=9)
    axes[0, col].axis('off')

    axes[1, col].imshow(to_displayable(incorrect_imgs[col]))
    axes[1, col].set_title(
        f'✗ pred: {class_names[incorrect_pred[col]]}\ntrue:  {class_names[incorrect_true[col]]}',
        color='red', fontsize=9)
    axes[1, col].axis('off')

axes[0, 0].set_ylabel('Correct', fontsize=11)
axes[1, 0].set_ylabel('Incorrect', fontsize=11)

plt.tight_layout()
plt.savefig('test_examples.png', dpi=150, bbox_inches='tight')
plt.show()

model.save("blood_cell_cnn_model.keras")
print("Model saved as blood_cell_cnn_model.keras")