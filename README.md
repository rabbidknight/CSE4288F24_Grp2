# Crosswalk Detection Using BDD100K

## Overview
This project uses the **BDD100K dataset** to train a YOLOv8 model for crosswalk detection. The dataset is recorded from a car's perspective and provides annotated bounding boxes, making it suitable for autonomous driving applications.

---

## Dataset

### BDD100K
- **Description**: A large-scale driving dataset with object annotations, including bounding boxes.
- **Download**:
  - [BDD100K Images](https://dl.cv.ethz.ch/bdd100k/data/bdd100k_images.zip)
  - [BDD100K Labels](https://dl.cv.ethz.ch/bdd100k/data/bdd100k_labels_release.zip)
- **Format**: Includes images and bounding box labels in JSON format.

---

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/bdd100k-crosswalk-detection.git
   cd bdd100k-crosswalk-detection

Install Dependencies:
pip install ultralytics pyyaml tqdm

Run the Script:
python convert_and_train_crosswalk.py
