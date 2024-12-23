import numpy as np
import json
from tqdm import tqdm  # Progress bar library


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
    for entry in tqdm(annotations, desc="Processing labels", unit="image"):
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