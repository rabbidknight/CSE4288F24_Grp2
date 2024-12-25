import os
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar library

# Path definitions
TRAIN_FEATURES = "features_train.npy"   # Path of the training features
VALIDATION_FEATURES = "features_val.npy"       # Path of the validation features
JSON_PATH = "lane_train.json"     # JSON file path
K = 5  # Number of neighbors

# Load the features from the saved .npy files
def load_features(feature_file):
    features_dict = np.load(feature_file, allow_pickle=True).item()  # Load the .npy file
    image_names = list(features_dict.keys())  # Get the list of image names
    features = np.array(list(features_dict.values()))  # Get the feature vectors
    return features, image_names

# Load labels from JSON
def load_labels(json_path):
    with open(json_path, "r") as f:
        image_metadata = json.load(f)

    labels = {}
    for entry in tqdm(image_metadata, desc="Processing labels", unit="image"):
        image_name = entry["name"]
        
        if "labels" in entry:   # Check if the image has labels
            has_crosswalk = any(label["category"] == "crosswalk" for label in entry["labels"])  # Check if the image has a crosswalk label
            labels[image_name] = 1 if has_crosswalk else 0  # Assign 1 if crosswalk is present, 0 otherwise
        else:
            labels[image_name] = 0

    return labels


if __name__ == "__main__":
    
    # Load the training and validation features
    print("Loading training and validation features...")
    X_train, train_names = load_features(TRAIN_FEATURES)
    X_validation, val_names = load_features(VALIDATION_FEATURES)

    # Load labels from JSON
    print("Loading labels...")
    labels = load_labels(JSON_PATH)

    # Map labels to training and validation images
    print("Mapping labels to training and validation images...")
    y_train = np.array([labels[name] for name in tqdm(train_names, desc="Mapping training labels")])
    y_validation = np.array([labels[name] for name in tqdm(val_names, desc="Mapping validation labels")])

    # Train the KNN model
    print(f"Training KNN model with K={K}...")
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train, y_train)

    # Make predictions on the validation set
    print("Predicting on validation set...")
    y_pred = knn.predict(X_validation)

    # Evaluate the model accuracy
    accuracy = accuracy_score(y_validation, y_pred)
    print(f"KNN Model Accuracy: {accuracy * 100:.2f}%")

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_validation, y_pred, target_names=['No Crosswalk', 'Crosswalk']))

    # Confusion Matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_validation, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Crosswalk', 'Crosswalk'], yticklabels=['No Crosswalk', 'Crosswalk'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    
