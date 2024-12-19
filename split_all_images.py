import os
import json
from sklearn.model_selection import train_test_split

#Paths
image_dir = "images/100k/train/"
json_path = "lane_train.json"
output_dir = "splits/"
os.makedirs(output_dir, exist_ok=True)


with open(json_path, "r") as f:
    annotations = json.load(f)


image_names = [entry["name"] for entry in annotations]


train_images, val_images = train_test_split(image_names, test_size=0.2, random_state=42)


with open(os.path.join(output_dir, "train_images.txt"), "w") as f:
    f.writelines("\n".join(train_images))
with open(os.path.join(output_dir, "val_images.txt"), "w") as f:
    f.writelines("\n".join(val_images))

print(f"Training and validation sets are ready. Saved to {output_dir}.")
