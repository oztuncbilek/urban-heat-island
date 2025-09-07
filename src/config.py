# src/config.py

import os

# --- Main Project Directory ---
# Assuming this script is in 'src', we navigate one level up to get the base directory.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Paths ---
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions") # Directory to save model predictions

# --- Model Storage Path ---
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "unet_model.keras") # Can also be .h5

# --- Model and Training Parameters ---
# Adjust these according to your image dimensions
IMG_HEIGHT = 256
IMG_WIDTH = 256
# Adjust this based on the number of bands in your satellite imagery (e.g., 3 for RGB, 7 for Landsat)
IMG_CHANNELS = 7 

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4

# --- Filenames (Example: replace with your actual filenames) ---
# Image and mask files to be used for training
TRAIN_IMAGE_PATH = os.path.join(PROCESSED_DATA_DIR, "training_images.tif")
TRAIN_MASK_PATH = os.path.join(PROCESSED_DATA_DIR, "training_masks.tif")

# Image file to be used for prediction
PREDICT_IMAGE_PATH = os.path.join(PROCESSED_DATA_DIR, "image_to_predict.tif")
