import os
import shutil
import random

random.seed(42)

SOURCE_DIR = "bloodcells_dataset"
TARGET_DIR = "bloodcells_subset"

IMAGES_PER_CLASS = 1000

os.makedirs(TARGET_DIR, exist_ok=True)

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue
    images = os.listdir(class_path)

    # ensure reproducibility
    images = sorted(images)
    random.shuffle(images)

    n_samples = len(images)
    selected_images = images[:n_samples]

    # create output folder
    split_folder = os.path.join(TARGET_DIR, class_name)
    os.makedirs(split_folder, exist_ok=True)

    # copy selected images
    for img in selected_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(split_folder, img)
        shutil.copy2(src, dst)

print("Subset dataset (1000 per class) created successfully!")