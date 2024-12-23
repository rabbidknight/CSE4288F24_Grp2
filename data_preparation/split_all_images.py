import os
import json
from sklearn.model_selection import train_test_split

#Paths
JSON_PATH = "../lane_train.json"    # Path of the JSON file containing the annotations
OUTPUT_PATH = "splits/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

with open(JSON_PATH, "r") as f:
    image_metadata = json.load(f)

image_names = [entry["name"] for entry in image_metadata]  # Extract image names from the JSON file

train_images, validation_images = train_test_split(image_names, test_size=0.2, random_state=42)    # Split the image names into training and validation sets (80% training, 20% validation)

with open(os.path.join(OUTPUT_PATH, "train_images.txt"), "w") as f:  # Save the image names to text files
    f.writelines("\n".join(train_images))
with open(os.path.join(OUTPUT_PATH, "val_images.txt"), "w") as f:
    f.writelines("\n".join(validation_images))

print(f"Training and validation sets are ready. Saved to {OUTPUT_PATH}.")
