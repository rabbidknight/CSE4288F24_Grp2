import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Path definitions
TRAIN_FEATURES = "features_train.npy"
VAL_FEATURES = "features_val.npy"
K = 5  # Number of neighbors for KNN

# Load the features from the saved .npy files
def load_features(feature_file):
    features_dict = np.load(feature_file, allow_pickle=True).item()  # Load the feature dictionary
    image_names = list(features_dict.keys())  # Get the list of image names
    features = np.array(list(features_dict.values()))  # Get the feature vectors
    return features, image_names

if __name__ == "__main__":
    # Load the training and validation features
    X_train, train_names = load_features(TRAIN_FEATURES)
    X_val, val_names = load_features(VAL_FEATURES)

    # Create labels (dummy labels: crosswalk present/absent)
    y_train = np.array([1 if "crosswalk" in name else 0 for name in train_names])
    y_val = np.array([1 if "crosswalk" in name else 0 for name in val_names])

    # Train the KNN model
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = knn.predict(X_val)

    # Evaluate the model accuracy
    accuracy = accuracy_score(y_val, y_pred)
    print(f"KNN Model Accuracy: {accuracy * 100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Crosswalk', 'Crosswalk'], yticklabels=['No Crosswalk', 'Crosswalk'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=['No Crosswalk', 'Crosswalk']))
