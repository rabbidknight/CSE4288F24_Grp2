import os
import json
import cv2
from pathlib import Path
import random
import shutil

images_path = "images/100k/train"  # All images
labels_file = "lane_train.json"  # Labels
yolo_dataset_path = "yolo_dataset"  # YOLO dataset

CROSSWALK_CLASS_ID = 0  # Class ID of the crosswalk

os.makedirs(f"{yolo_dataset_path}/images/train", exist_ok=True)
os.makedirs(f"{yolo_dataset_path}/images/val", exist_ok=True)
os.makedirs(f"{yolo_dataset_path}/labels/train", exist_ok=True)
os.makedirs(f"{yolo_dataset_path}/labels/val", exist_ok=True)


# Convert the vertices into YOLO format. Basically, find the x center, y center, width and height, then normalize
def polygon_to_bbox(vertices, image_width, image_height):
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    x_min, x_max = max(0, min(x_coords)), min(image_width, max(x_coords))
    y_min, y_max = max(0, min(y_coords)), min(image_height, max(y_coords))
    width = x_max - x_min  # Width is calculated with x vertices
    height = y_max - y_min  # Height is calculated with y vertices
    x_center = x_min + width / 2  # Then, centers are calculated for both
    y_center = y_min + height / 2
    return x_center / image_width, y_center / image_height, width / image_width, height / image_height  # Normalize


with open(labels_file, "r") as f:  # Retrieve the labels json file
    labels_data = json.load(f)

labels_by_image = {}
for item in labels_data:
    if "labels" in item:
        labels_by_image[item["name"]] = item["labels"]

image_files = list(Path(images_path).glob("*.jpg"))  # Retrieve all the images from the given directory

random.shuffle(image_files)  # Shuffle the images for fairness
split_ratio = 0.8  # Divide into 80% training and 20% validation for YOLO training and evaluation processes
train_files = image_files[:int(len(image_files) * split_ratio)]
val_files = image_files[int(len(image_files) * split_ratio):]

for image_path in train_files + val_files:  # All the images with crosswalk is here
    image_name = image_path.name

    if image_name not in labels_by_image:  # Skip if no labels are found for the image
        continue

    image = cv2.imread(str(image_path))
    if image is None:
        continue
    height, width, _ = image.shape

    labels = labels_by_image[image_name]

    yolo_labels = []
    for label in labels:
        if label["category"] == "crosswalk":  # Process images with only crosswalk as category in labeling
            for poly in label["poly2d"]:
                bbox = polygon_to_bbox(poly["vertices"], width, height)
                yolo_labels.append(f"{CROSSWALK_CLASS_ID} {' '.join(map(str, bbox))}")

    if not yolo_labels:  # Skip images without crosswalks, because no object to detect in those images
        continue

    subfolder = "train" if image_path in train_files else "val"
    label_file = Path(yolo_dataset_path) / f"labels/{subfolder}/{image_name.replace('.jpg', '.txt')}"
    image_output_path = Path(yolo_dataset_path) / f"images/{subfolder}/{image_name}"

    with open(label_file, "w") as f:
        f.write("\n".join(yolo_labels))

    shutil.copy(image_path, image_output_path)

# Create YOLO configuration files
with open(f"{yolo_dataset_path}/data.yaml", "w") as f:
    f.write(f"""
            train: {yolo_dataset_path}/images/train
            val: {yolo_dataset_path}/images/val
            nc: 1  # Number of classes
            names: ['crosswalk']  # Class names
            """)

print("YOLO dataset is ready.")
