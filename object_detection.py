import tensorflow as tf
from tensorflow import keras
layers = tf.keras.layers
import numpy as np
import os
import matplotlib.pyplot as plt

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