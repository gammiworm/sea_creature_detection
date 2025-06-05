import os
from ultralytics import YOLO
import kagglehub
import shutil

# 1. Download the dataset
dataset_path = kagglehub.dataset_download("slavkoprytula/aquarium-data-cots")

# 2. Define the destination (same folder as this Python file)
script_dir = os.path.dirname(os.path.abspath(__file__))
destination_path = os.path.join(script_dir, "aquarium-data-cots")

# 3. Copy dataset to your local project directory
if not os.path.exists(destination_path):
    shutil.copytree(dataset_path, destination_path)
    print(f"Dataset copied to: {destination_path}")
else:
    print(f"Dataset already exists at: {destination_path}")


    

# Load a pretrained YOLOv11 model (n = nano, s = small, m = medium, l = large)
model = YOLO("yolo11n.pt")

# Train the model
model.train(
    data="ocean.yaml",     # Path to your YAML config file
    epochs=10,             # Number of training epochs
    imgsz=160,             # Image size (can go higher if you have GPU power)
    batch=64,              # Batch size (adjust based on your GPU RAM)
    project="output", # Output folder
    name="yolov8-ocean",     # Experiment name
    exist_ok=True            # Overwrite if the folder already exists
)

predict_model = YOLO("output/yolov8-ocean/weights/best.pt")

results = predict_model.predict(
    source="aquarium-data-cots/aquarium_pretrain/test/images",  # folder of .jpg/.png images
    conf=0.25,                     # confidence threshold
    save=True,                     # saves images with boxes to disk
    save_txt=True,                 # also saves YOLO-format predictions
    name="yolo-test-output"        # folder inside runs/predict/
)