import json
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Paths of the feature files
TRAIN_FEATURES = "features_train.npy"
VAL_FEATURES = "features_val.npy"  
# Label path
JSON_PATH = "lane_train.json"  

# Load the features from the saved .npy file
def load_features(feature_file):
    features_dict = np.load(feature_file, allow_pickle=True).item()  # Load the feature dictionary
    
    # Seperate the dictionary into image names and feature vectors
    image_names = list(features_dict.keys())  # Get the list of image names
    features = np.array(list(features_dict.values()))  # Get the feature vectors
    return features, image_names

# Load truth labels from JSON
def load_labels(json_path):
    
    # Read the json file
    with open(json_path, "r") as f:
        annotations = json.load(f)

    labels = {}
    
    # Go through each line in the json file
    for entry in annotations:
        # Get the name of the current image
        image_name = entry["name"]
        
        # Check if the image has labels
        if "labels" in entry:
            # Check if one of the labels contains crosswalk, if it does set the value for this image as true (1)
            has_crosswalk = any(label["category"] == "crosswalk" for label in entry["labels"])
            labels[image_name] = 1 if has_crosswalk else 0
        else:
            labels[image_name] = 0

    return labels

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
    y_train = np.array([labels[name] for name in train_names])
    y_val = np.array([labels[name] for name in val_names])


    # Train the Logistic Regression model!
    print("Training Logistic Regression model...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    # Doing prediction
    print("Predicting on validation set...")
    y_pred = lr.predict(X_val)
    
        # Calculate the accuracy using the predicted values
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Logistic Regression Model Accuracy: {accuracy * 100:.2f}%")

    # Print the classification report
    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=['No Crosswalk', 'Crosswalk']))

    # Show the confusion matrix using matlab
    print("Generating confusion matrix...")
    # create the confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    
    # create the matlab figure and setting up labels/titles for the figure
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Crosswalk', 'Crosswalk'], yticklabels=['No Crosswalk', 'Crosswalk'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show() # show the created figure