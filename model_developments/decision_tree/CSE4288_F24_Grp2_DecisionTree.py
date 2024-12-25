# Library imports
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Path definitions for image embedding results
TRAIN_FEATURES = "features_train.npy"
VAL_FEATURES = "features_val.npy"
JSON_PATH = "lane_train.json"  # JSON file path for labels

# Load the features from .npy files
def load_features(feature_file):
    features_dict = np.load(feature_file, allow_pickle=True).item()
    image_names = list(features_dict.keys())
    features = np.array(list(features_dict.values()))  # Get the feature vectors
    return features, image_names

# Load labels from JSON file
def load_labels(json_path):
    with open(json_path, "r") as f:
        annotations = json.load(f)

    labels = {}
    for entry in annotations:
        image_name = entry["name"]
        if "labels" in entry:
            has_crosswalk = any(label["category"] == "crosswalk" for label in entry["labels"])
            labels[image_name] = 1 if has_crosswalk else 0
        else:
            labels[image_name] = 0
    return labels

if __name__ == "__main__":
    # Load the training and validation features
    print("Loading training and validation features...")
    X_train, train_names = load_features(TRAIN_FEATURES)
    X_val, val_names = load_features(VAL_FEATURES)

    # Load labels from JSON
    print("Loading labels...")
    labels = load_labels(JSON_PATH)

    # Map labels to training and validation images
    print("Mapping labels to training and validation images...")
    y_train = np.array([labels[name] for name in train_names])
    y_val = np.array([labels[name] for name in val_names])

    # Training of Decision Tree model
    print("Training Decision Tree model...")
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model.fit(X_train, y_train)

    # Make prediction
    print("Predicting on validation set...")
    y_pred = decision_tree_model.predict(X_val)

    # Evaluating model accuracy
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Decision Tree Model Accuracy: {accuracy * 100:.2f}%")

    # Printing classification report
    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=['No Crosswalk', 'Crosswalk']))

    # Generate and print Confusion Matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Crosswalk', 'Crosswalk'], yticklabels=['No Crosswalk', 'Crosswalk'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
