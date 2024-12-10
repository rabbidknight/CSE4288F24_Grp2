import os
import yaml
import logging
from ultralytics import YOLO

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Step 1: Environment Setup
def setup_environment():
    logger.info("Setting up environment...")
    required_directories = ['dataset/train/images', 'dataset/train/labels', 'dataset/val/images', 'dataset/val/labels']
    for directory in required_directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
    logger.info("Environment setup complete.")

# Step 2: Dataset Configuration
def create_dataset_yaml(train_path, val_path, output_path='crosswalk_dataset.yaml'):
    logger.info("Creating dataset YAML configuration...")
    dataset_config = {
        'train': train_path,
        'val': val_path,
        'nc': 1,  # Number of classes
        'names': ['crosswalk']  # Class names
    }
    with open(output_path, 'w') as file:
        yaml.dump(dataset_config, file)
    logger.info(f"Dataset YAML saved to {output_path}")

# Step 3: Model Training
def train_yolo_model(yaml_path, epochs=50, imgsz=640, batch_size=16, model_name='yolov8n.pt', project_name='crosswalk_detection'):
    logger.info("Initializing YOLOv8 model...")
    model = YOLO(model_name)  # Automatically downloads the YOLOv8 model if not already present
    logger.info("Starting training...")
    model.train(
        data=yaml_path,  # Path to dataset YAML
        epochs=epochs,   # Number of training epochs
        imgsz=imgsz,     # Image size
        batch=batch_size,  # Batch size
        project=project_name,  # Project folder name
        name='crosswalk_model'  # Experiment name
    )
    logger.info("Training complete.")

# Step 4: Model Evaluation
def evaluate_model(model, data_path):
    logger.info("Evaluating model performance...")
    metrics = model.val(data=data_path)  # Path to validation data in the dataset YAML
    logger.info("Evaluation complete. Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value}")

# Step 5: Model Export
def export_model(model, output_format='onnx'):
    logger.info(f"Exporting model to {output_format} format...")
    model.export(format=output_format)
    logger.info(f"Model exported successfully to {output_format} format.")

# Main Pipeline
def main():
    # Paths to your dataset
    train_images = 'dataset/train/images'
    val_images = 'dataset/val/images'
    train_labels = 'dataset/train/labels'
    val_labels = 'dataset/val/labels'

    # Check and setup environment
    setup_environment()

    # Validate dataset paths
    assert os.path.exists(train_images), f"Training images path {train_images} does not exist."
    assert os.path.exists(val_images), f"Validation images path {val_images} does not exist."
    assert os.path.exists(train_labels), f"Training labels path {train_labels} does not exist."
    assert os.path.exists(val_labels), f"Validation labels path {val_labels} does not exist."

    # Create dataset YAML configuration
    create_dataset_yaml(train_path=train_images, val_path=val_images)

    # Load and train YOLOv8 model
    model_path = 'yolov8n.pt'  # YOLOv8 nano model for speed; choose 's', 'm', 'l', or 'x' for larger versions
    train_yolo_model(yaml_path='crosswalk_dataset.yaml', model_name=model_path)

    # Load trained model
    trained_model = YOLO('runs/train/crosswalk_model/weights/best.pt')  # Update path if using custom project/experiment names

    # Evaluate model
    evaluate_model(trained_model, data_path='crosswalk_dataset.yaml')

    # Export model
    export_model(trained_model, output_format='onnx')

if __name__ == "__main__":
    main()
