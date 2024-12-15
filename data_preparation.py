import os
import json
import cv2
from pathlib import Path
from tqdm import tqdm
import random
import shutil

images_dir = "images/10k/train"  # Folder containing all images
labels_file = "lane_train.json"  # Path to the single JSON file with all labels
output_dir = "yolo_dataset"  # Folder to store YOLO dataset

# YOLO class ID for "crosswalk"
CROSSWALK_CLASS_ID = 0

# Create output folders
os.makedirs(f"{output_dir}/images/train", exist_ok=True)
os.makedirs(f"{output_dir}/images/val", exist_ok=True)
os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
os.makedirs(f"{output_dir}/labels/val", exist_ok=True)


# Convert polygon to bounding box
def polygon_to_bbox(vertices, image_width, image_height):
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    x_min, x_max = max(0, min(x_coords)), min(image_width, max(x_coords))
    y_min, y_max = max(0, min(y_coords)), min(image_height, max(y_coords))
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    # Normalize values to [0, 1]
    return x_center / image_width, y_center / image_height, width / image_width, height / image_height

with open(labels_file, "r") as f:
    labels_data = json.load(f)

labels_by_image = {}
for item in labels_data:
    if "labels" in item:
        labels_by_image[item["name"]] = item["labels"]

# Parse images and process labels
image_files = list(Path(images_dir).glob("*.jpg"))

random.shuffle(image_files)
split_ratio = 0.8
train_files = image_files[:int(len(image_files) * split_ratio)]
val_files = image_files[int(len(image_files) * split_ratio):]

for image_path in tqdm(train_files + val_files):
    image_name = image_path.name

    # Skip if no labels are found for the image
    if image_name not in labels_by_image:
        print(f"Skipping: No labels for {image_name}")
        continue

    # Load image to get dimensions
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Skipping: Unable to load image {image_name}")
        continue
    height, width, _ = image.shape

    # Get the labels for this image
    labels = labels_by_image[image_name]

    # Create YOLO label file
    yolo_labels = []
    for label in labels:
        if label["category"] == "crosswalk":  # Process only crosswalks
            for poly in label["poly2d"]:
                bbox = polygon_to_bbox(poly["vertices"], width, height)
                yolo_labels.append(f"{CROSSWALK_CLASS_ID} {' '.join(map(str, bbox))}")

    # Skip images without crosswalks
    if not yolo_labels:
        print(f"Skipping: No crosswalk labels for {image_name}")
        continue

    # Determine output subfolder
    subfolder = "train" if image_path in train_files else "val"
    label_file = Path(output_dir) / f"labels/{subfolder}/{image_name.replace('.jpg', '.txt')}"
    image_output_path = Path(output_dir) / f"images/{subfolder}/{image_name}"

    # Write YOLO label file
    with open(label_file, "w") as f:
        f.write("\n".join(yolo_labels))

    # Copy image to corresponding folder


    shutil.copy(image_path, image_output_path)

# Create YOLO configuration files
with open(f"{output_dir}/data.yaml", "w") as f:
    f.write(f"""
train: {output_dir}/images/train
val: {output_dir}/images/val

nc: 1  # Number of classes
names: ['crosswalk']  # Class names
""")

print("Dataset preparation completed.")
