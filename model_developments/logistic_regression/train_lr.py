import os
import json
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm  # Progress bar library

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../../model_developments/'))
sys.path.append(os.path.join(dir_name,'../../'))
from model_developments.load_features_labels  import *
from calculate_results import *



if __name__ == "__main__":

    X_train, X_val, y_train, y_val = get_values()
    # Train the Logistic Regression model!
    print("Training Logistic Regression model...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    # Doing prediction
    print("Predicting on validation set...")
    y_pred = lr.predict(X_val)
    
    show_results("Logistic Regression", y_val, y_pred)