import os
from pathlib import Path

# Paths
labels_dir = "yolo_dataset/labels/train"  # Folder containing training labels
val_labels_dir = "yolo_dataset/labels/val"  # Folder containing validation labels

# Function to count crosswalk annotations
def count_crosswalks(label_folder):
    total_annotations = 0
    crosswalks_per_image = []

    # Iterate over all label files
    for label_file in Path(label_folder).glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()
            crosswalk_count = sum(1 for line in lines if line.strip())  # Count lines with valid entries
            crosswalks_per_image.append(crosswalk_count)
            total_annotations += crosswalk_count

    return total_annotations, crosswalks_per_image

# Count for training and validation folders
train_total, train_crosswalks_per_image = count_crosswalks(labels_dir)
val_total, val_crosswalks_per_image = count_crosswalks(val_labels_dir)

# Combine statistics
total_annotations = train_total + val_total
all_crosswalks_per_image = train_crosswalks_per_image + val_crosswalks_per_image
images_with_crosswalks = len([count for count in all_crosswalks_per_image if count > 0])

count_per_image = 0
max_count = 0
min_count = 10


# Print the results
print("Crosswalk Statistics:")
print(f"Total number of crosswalk annotations: {total_annotations}")
print(f"Total number of images with crosswalks: {images_with_crosswalks}")
print("Number of crosswalks per image:")
for i, count in enumerate(all_crosswalks_per_image, start=1):
    print(f"Image {i}: {count} crosswalk(s)")
    count_per_image += count
    if count < min_count:
        min_count = count
    if count > max_count:
        max_count = count


print("Average crosswalk count per image is:", count_per_image / images_with_crosswalks)
print("Maximum crosswalk count per image is:", max_count)
print("Minimum crosswalk count per image is:", min_count)
