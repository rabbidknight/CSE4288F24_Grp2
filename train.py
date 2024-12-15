import os
import requests
import zipfile
from ultralytics import YOLO


# Step 1: Download and Extract BDD100K Dataset
def download_bdd100k():
    """
    Downloads and extracts the BDD100K dataset (images and labels).
    """
    urls = {
        "train_images": "https://dl.cv.ethz.ch/bdd100k/data/100k_images_train.zip",
        "val_images": "https://dl.cv.ethz.ch/bdd100k/data/100k_images_val.zip",
        "labels": "https://dl.cv.ethz.ch/bdd100k/data/bdd100k_det_20_labels_trainval.zip"
    }

    dataset_dir = "bdd100k"
    os.makedirs(dataset_dir, exist_ok=True)

    for key, url in urls.items():
        output_path = os.path.join(dataset_dir, f"{key}.zip")
        if not os.path.exists(output_path):
            print(f"Downloading {key} from {url}...")
            download_file(url, output_path)
        else:
            print(f"{key} already downloaded.")
        
        # Extract the downloaded file
        extract_zip(output_path, dataset_dir)


def download_file(url, output_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


# Step 2: Organize Dataset for YOLO
def organize_bdd100k():
    print("Organizing BDD100K dataset for YOLOv8...")
    base_dir = "bdd100k"
    train_images_src = os.path.join(base_dir, "images", "100k", "train")
    val_images_src = os.path.join(base_dir, "images", "100k", "val")
    train_labels_src = os.path.join(base_dir, "labels", "det_20", "det_train.json")
    val_labels_src = os.path.join(base_dir, "labels", "det_20", "det_val.json")

    # YOLO-compatible directories
    dataset_dir = "dataset"
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Copy images (No need to copy, just point the paths)
    return train_images_src, val_images_src, train_labels_src, val_labels_src


# Step 3: Create YAML Configuration
def create_yaml(train_images, val_images, output_path="bdd100k.yaml"):
    print("Creating YAML configuration for YOLOv8...")
    yaml_content = f"""
        train: {train_images}
        val: {val_images}
        nc: 1
        names: ['crosswalk']
        """
    with open(output_path, "w") as f:
        f.write(yaml_content)
    print(f"YAML configuration saved to {output_path}")


# Step 4: Train YOLOv8 Model
def train_yolo_model(yaml_path, epochs=50, imgsz=640, batch_size=16, model_name="yolov8n.pt"):
    print("Starting YOLOv8 training...")
    model = YOLO(model_name)
    model.train(data=yaml_path, epochs=epochs, imgsz=imgsz, batch=batch_size, project="crosswalk_detection")
    print("Training complete.")


# Main Execution
if __name__ == "__main__":
    # Download and prepare BDD100K
    download_bdd100k()
    train_images, val_images, train_labels, val_labels = organize_bdd100k()

    # Create YAML configuration
    create_yaml(train_images, val_images)

    # Train YOLOv8 model
    train_yolo_model("bdd100k.yaml")
