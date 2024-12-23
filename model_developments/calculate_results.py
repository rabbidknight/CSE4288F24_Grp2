
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from load_features_labels import *

# Paths of the feature files
TRAIN_FEATURES = "features_train.npy"
VAL_FEATURES = "features_val.npy"  
# Label path
JSON_PATH = "lane_train.json"  

def get_values():
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
    return X_train, X_val, y_train, y_val

def show_results(model_name, y_val, y_pred):
    # Calculate the accuracy using the predicted values
    accuracy = accuracy_score(y_val, y_pred)
    print(f"{model_name} Model Accuracy: {accuracy * 100:.2f}%")

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
