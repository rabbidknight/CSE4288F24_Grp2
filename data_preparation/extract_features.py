import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Path definitions
IMAGE_PATH = "images/100k/train/"
TRAIN_LIST = "splits/train_images.txt"
VAL_LIST = "splits/val_images.txt"
TRAIN_FEATURES = "features_train.npy"
VAL_FEATURES = "features_val.npy"

# Prepare the feature extraction model
def prepare_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
    print(f"Using {device}!")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  # Load pre-trained ResNet50 model
    model.eval()  # Set the model to evaluation mode
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the fully connected layer
    return feature_extractor.to(device), device

# Preprocess the image
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    image = Image.open(image_path).convert('RGB')  # Load image and convert to RGB
    return preprocess(image).unsqueeze(0)

# Extract features from images and save them as .npy files
def extract_features(feature_extractor, image_list, image_dir, output_file, device):
    features = {}
    for img_name in tqdm(image_list, desc="Extracting features", unit="image"): # Loop over all images
        image_path = os.path.join(image_dir, img_name)
        if os.path.exists(image_path):
            try:
                input_tensor = preprocess_image(image_path).to(device)  # Preprocess the image and move to device
                with torch.no_grad():
                    feature_vector = feature_extractor(input_tensor).flatten().cpu().numpy()  # Extract feature vector
                features[img_name] = feature_vector # Save the feature vector
            except Exception as e:
                tqdm.write(f"Error processing {img_name}: {e}")
        else:
            tqdm.write(f"Warning: {image_path} does not exist!")
    np.save(output_file, features)  # Save extracted features as a .npy file
    print(f"Features saved to: {output_file}")

if __name__ == "__main__":

    feature_extractor, device = prepare_model()
    
    # Load image lists
    with open(TRAIN_LIST, "r") as f:
        train_images = [line.strip() for line in f.readlines()]
    with open(VAL_LIST, "r") as f:
        val_images = [line.strip() for line in f.readlines()]

    # Extract features and save them
    extract_features(feature_extractor, train_images, IMAGE_PATH, TRAIN_FEATURES, device)
    extract_features(feature_extractor, val_images, IMAGE_PATH, VAL_FEATURES, device)