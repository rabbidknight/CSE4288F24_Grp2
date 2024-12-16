# Crosswalk Detection Using BDD100K
GROUP 2 - CSE4288 TERM PROJECT
150120042 Nidanur Demirhan
150120075 Duru Baştunalı
150120076 Furkan Gökgöz
150121043 Bora Duman
150121051 Arda Öztürk

## Overview
This project uses the **BDD100K dataset** to train a YOLOv8 model for crosswalk detection. The dataset is recorded from a car's perspective and provides annotated bounding boxes, making it suitable for autonomous driving applications.

---

## Dataset

### BDD100K
- **Description**: A large-scale driving dataset with object annotations, including bounding boxes.
- **Training Set**: https://dl.cv.ethz.ch/bdd100k/data/100k_images_train.zip
- **Labels**: https://dl.cv.ethz.ch/bdd100k/data/bdd100k_lane_labels_trainval.zip

- **Format**: Includes images and bounding box labels in JSON format.

---

## How to Run

1. Extract images from zip and add the images file to your project root directory.
2. Add labels json to your project root directory.
3. Run preprocessing code.
4. Run training code.
5. Add the image you would like to detect crosswalk in and run test.

