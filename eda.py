import os
from pathlib import Path
from PIL import Image

# Paths
labels_dir = "yolo_dataset/labels/train"
val_labels_dir = "yolo_dataset/labels/val"
images_dir = "yolo_dataset/images/train"
val_images_dir = "yolo_dataset/images/val"

def count_crosswalks(label_folder):
    total_annotations = 0
    crosswalks_per_image = []
    file_names = []

    for label_file in Path(label_folder).glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()
            crosswalk_count = sum(1 for line in lines if line.strip())  # Count lines with valid entries
            crosswalks_per_image.append(crosswalk_count)
            file_names.append(label_file.stem)  # Save file name without extension
            total_annotations += crosswalk_count

    return total_annotations, crosswalks_per_image, file_names

def get_image_sizes(image_folder):
    image_sizes = []
    for image_file in Path(image_folder).glob("*.jpg"):  # Assuming images are in JPG format
        with Image.open(image_file) as img:
            width, height = img.size
            image_sizes.append((width, height))
    return image_sizes

def calculate_image_stats(image_sizes):
    min_width = min(image_sizes, key=lambda x: x[0])[0]
    min_height = min(image_sizes, key=lambda x: x[1])[1]
    max_width = max(image_sizes, key=lambda x: x[0])[0]
    max_height = max(image_sizes, key=lambda x: x[1])[1]
    avg_width = sum(size[0] for size in image_sizes) / len(image_sizes)
    avg_height = sum(size[1] for size in image_sizes) / len(image_sizes)
    return (min_width, min_height), (max_width, max_height), (avg_width, avg_height)

train_total, train_crosswalks_per_image, train_file_names = count_crosswalks(labels_dir)
val_total, val_crosswalks_per_image, val_file_names = count_crosswalks(val_labels_dir)

total_annotations = train_total + val_total
all_crosswalks_per_image = train_crosswalks_per_image + val_crosswalks_per_image
all_file_names = train_file_names + val_file_names
images_with_crosswalks = len([count for count in all_crosswalks_per_image if count > 0])

count_per_image = 0
max_count = 0
min_count = 10

train_image_sizes = get_image_sizes(images_dir)
val_image_sizes = get_image_sizes(val_images_dir)
all_image_sizes = train_image_sizes + val_image_sizes

min_size, max_size, avg_size = calculate_image_stats(all_image_sizes)

max_crosswalks = max(all_crosswalks_per_image)
image_with_max_crosswalks = all_file_names[all_crosswalks_per_image.index(max_crosswalks)]

print("Crosswalk Statistics:")
print(f"Total number of crosswalk annotations: {total_annotations}")
print(f"Total number of images with crosswalks: {images_with_crosswalks}")
print("Number of crosswalks per image:")
for i, count in enumerate(all_crosswalks_per_image, start=1):
    count_per_image += count
    if count < min_count:
        min_count = count
    if count > max_count:
        max_count = count

print("Average crosswalk count per image is:", count_per_image / images_with_crosswalks)
print("Maximum crosswalk count per image is:", max_count)
print("Minimum crosswalk count per image is:", min_count)
print(f"Image with the maximum crosswalks ({max_crosswalks}): {image_with_max_crosswalks}.jpg")

print("\nImage Size Statistics:")
print(f"Minimum image size: {min_size[0]}x{min_size[1]}")
print(f"Maximum image size: {max_size[0]}x{max_size[1]}")
print(f"Average image size: {avg_size[0]:.2f}x{avg_size[1]:.2f}")
