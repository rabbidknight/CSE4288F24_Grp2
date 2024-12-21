import os
import json
import cv2
from pathlib import Path
import random

images_dir = "images/100k/train"  # Training images
labels_file = "lane_train.json"  # Label file
output_dir = "yolo_dataset"  # Prepared dataset for YOLO

CROSSWALK_CLASS_ID = 0  # Class ID for crosswalk = 0 (has no significance)

os.makedirs(f"{output_dir}/images/train", exist_ok=True)
os.makedirs(f"{output_dir}/images/val", exist_ok=True)
os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
os.makedirs(f"{output_dir}/labels/val", exist_ok=True)


def polygon_to_bbox(vertices, image_width, image_height):  # Convert 2D polygons into YOLO boundaries
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    x_min, x_max = max(0, min(x_coords)), min(image_width, max(x_coords))
    y_min, y_max = max(0, min(y_coords)), min(image_height, max(y_coords))
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    # YOLO LABEL FORMAT = <x center>, <y center> <width> <height>, ALL NORMALIZED!
    return x_center / image_width, y_center / image_height, width / image_width, height / image_height


with open(labels_file, "r") as f:  # Load the labels file
    labels_data = json.load(f)

labels_by_image = {}
for item in labels_data:
    if "labels" in item:
        labels_by_image[item["name"]] = item["labels"]  # Retrieve each label of the image and create dictionary

# Parse images and process labels = For setting the set as training and validation randomly
image_files = list(Path(images_dir).glob("*.jpg"))
random.shuffle(image_files)
split_ratio = 0.8
train_files = image_files[:int(len(image_files) * split_ratio)]
val_files = image_files[int(len(image_files) * split_ratio):]

# --------------------------------------------------------
for image_path in train_files + val_files:
    image_name = image_path.name

    if image_name not in labels_by_image:
        print(f"Skipping: No labels for {image_name}")
        continue  # Skipped for unlabeled images

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

    if not yolo_labels:
        print(f"Skipping: No crosswalk labels for {image_name}")
        continue  # Skipped for images with no crosswalk

    # Determine output subfolder
    subfolder = "train" if image_path in train_files else "val"
    label_file = Path(output_dir) / f"labels/{subfolder}/{image_name.replace('.jpg', '.txt')}"
    image_output_path = Path(output_dir) / f"images/{subfolder}/{image_name}"

    # Write YOLO label file
    with open(label_file, "w") as f:
        f.write("\n".join(yolo_labels))

    # Copy image to corresponding folder using basic file operations
    with open(image_path, "rb") as src_file:
        with open(image_output_path, "wb") as dest_file:
            dest_file.write(src_file.read())
# --------------------------------------------------------

with open(f"{output_dir}/data.yaml", "w") as f:  # Create the data.yaml file for YOLO-acceptability
    f.write(f"""
        train: {output_dir}/images/train
        val: {output_dir}/images/val

        nc: 1  # Number of classes
        names: ['crosswalk']  # Class names
        """)

print("Dataset preparation completed.")
