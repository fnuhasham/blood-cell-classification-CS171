import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = "bloodcells_subset"
IMG_SIZE = (160, 160)
SEED = 42
IMG_PATH = "ERB_2429.jpg"

# Get class names from the same dataset used in training
train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=16
)
class_names = train_data.class_names
print("Class names:", class_names)  # verify the order

# Load Model
model = tf.keras.models.load_model("blood_cell_cnn_model.keras")

# Load Image
img = tf.keras.utils.load_img(IMG_PATH, target_size=(160, 160))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, axis=0)

# Show Image
img_display = tf.keras.utils.load_img(IMG_PATH, target_size=IMG_SIZE)

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]
confidence = np.max(predictions) * 100

plt.imshow(img_display)
plt.axis("off")
plt.title(f"Predicted: {class_names[predicted_class]} ({confidence:.2f}%)")
plt.show()

print(f"Predicted: {class_names[predicted_class]} ({confidence:.2f}%)")