# Crosswalk Detection Using BDD100K
GROUP 2 - CSE4288 project 
bora duman 150121043
## Overview
This project uses the **BDD100K dataset** to train a YOLOv8 model for crosswalk detection. The dataset is recorded from a car's perspective and provides annotated bounding boxes, making it suitable for autonomous driving applications.

---

## Dataset

### BDD100K
- **Description**: A large-scale driving dataset with object annotations, including bounding boxes.
- **Download**:
  https://dl.cv.ethz.ch/bdd100k/data/
- **Format**: Includes images and bounding box labels in JSON format.

---

## How to Run

Install Dependencies:
pip install ultralytics pyyaml tqdm

Run the Script:
python convert_and_train_crosswalk.py
