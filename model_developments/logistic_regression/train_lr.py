import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm  # Progress bar library

from ..load_features_labels import load_features,load_labels
from ..calculate_results import show_results

# Paths of the feature files
TRAIN_FEATURES = "features_train.npy"
VAL_FEATURES = "features_val.npy"  

# Label path
JSON_PATH = "../../lane_train.json"  

if __name__ == "__main__":
    # Load the training and validation feature vectors and names of the images
    print("Loading training and validation features...")
    X_train, train_names = load_features(TRAIN_FEATURES)
    X_val, val_names = load_features(VAL_FEATURES)

    # Load truth labels from JSON
    print("Loading labels...")
    labels = load_labels(JSON_PATH)

    # Map labels to training and validation images and save it as it will serve as y values (target values)
    print("Mapping labels to training and validation images...")
    y_train = np.array([labels[name] for name in tqdm(train_names, desc="Mapping training labels")])
    y_val = np.array([labels[name] for name in tqdm(val_names, desc="Mapping validation labels")])

    # Train the Logistic Regression model!
    print("Training Logistic Regression model...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    # Doing prediction
    print("Predicting on validation set...")
    y_pred = lr.predict(X_val)

    show_results(y_val, y_pred)