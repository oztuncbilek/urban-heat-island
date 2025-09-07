# src/data_loader.py

import rasterio
import numpy as np
from sklearn.model_selection import train_test_split
from config import IMG_HEIGHT, IMG_WIDTH

def load_geotiff(filepath):
    """
    Reads a GeoTIFF file and returns it as a NumPy array.
    It also transposes the axes from (channels, height, width) to (height, width, channels).
    """
    with rasterio.open(filepath) as src:
        # Reads in (channels, height, width) format
        img_array = src.read()
        # Transpose to (height, width, channels) format
        img_array = np.transpose(img_array, (1, 2, 0))
    return img_array

def create_patches(image, mask, patch_size_h=IMG_HEIGHT, patch_size_w=IMG_WIDTH):
    """
    Divides a large image and its corresponding mask into smaller patches.
    """
    patches_img = []
    patches_mask = []
    
    # Get image dimensions
    height, width, _ = image.shape
    
    for y in range(0, height - patch_size_h + 1, patch_size_h):
        for x in range(0, width - patch_size_w + 1, patch_size_w):
            patch_img = image[y:y+patch_size_h, x:x+patch_size_w]
            patch_mask = mask[y:y+patch_size_h, x:x+patch_size_w]
            patches_img.append(patch_img)
            patches_mask.append(patch_mask)
            
    return np.array(patches_img), np.array(patches_mask)

def get_training_data(image_path, mask_path):
    """
    Loads data, creates patches, and splits it into training and validation sets.
    """
    print("Loading and processing data...")
    
    # Load the full-size images and masks
    image = load_geotiff(image_path)
    mask = load_geotiff(mask_path)

    # Normalize image to a 0-1 range. This step might need adjustment based on your data type.
    # E.g., if your images are 16-bit (0-65535), you might divide by 65535.0.
    # For now, we normalize by the max value in the image.
    image = image / np.max(image) 
    
    # Normalize mask (usually contains values of 0 and 255)
    mask = mask / 255.0 
    if mask.ndim == 2: # If mask is (H, W), add a channel dimension
        mask = np.expand_dims(mask, axis=-1)

    # Create patches
    X, y = create_patches(image, mask)

    # Split patches into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Created {len(X_train)} patches for training and {len(X_val)} for validation.")
    
    return X_train, X_val, y_train, y_val

