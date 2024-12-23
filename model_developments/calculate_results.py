
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def show_results(y_val, y_pred):
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
