# src/predict.py

import os
import rasterio
import numpy as np
import tensorflow as tf

from config import MODEL_PATH, PREDICT_IMAGE_PATH, PREDICTIONS_DIR, IMG_HEIGHT, IMG_WIDTH

def main():
    """
    Main function to load a trained model and make predictions on a new image.
    """
    print("--- Starting Prediction ---")
    
    # 1. Load the trained model
    print(f"Loading model from: {MODEL_PATH}")
    # We don't need to compile the model for prediction, so compile=False is faster.
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # 2. Load the image to predict on
    print(f"Loading image for prediction from: {PREDICT_IMAGE_PATH}")
    with rasterio.open(PREDICT_IMAGE_PATH) as src:
        # Get metadata to save the output correctly (georeferencing, etc.)
        meta = src.meta
        # Read the image and transpose to (H, W, C)
        image = np.transpose(src.read(), (1, 2, 0))

    # 3. Preprocess the image (must be the same as in training)
    # Normalize the image
    original_shape = image.shape
    image_normalized = image / np.max(image)
    
    # Create patches from the image
    # Note: This is a simple patching approach. For more complex cases,
    # you might need overlapping patches and averaging the results.
    patches = []
    for y in range(0, original_shape[0] - IMG_HEIGHT + 1, IMG_HEIGHT):
        for x in range(0, original_shape[1] - IMG_WIDTH + 1, IMG_WIDTH):
            patch = image_normalized[y:y+IMG_HEIGHT, x:x+IMG_WIDTH]
            patches.append(patch)
    
    patches = np.array(patches)

    # 4. Make predictions on the patches
    print(f"Making predictions on {len(patches)} patches...")
    predicted_patches = model.predict(patches)
    
    # 5. Stitch the predicted patches back together
    # Create an empty canvas to place the predicted patches on
    prediction_full = np.zeros((original_shape[0], original_shape[1]), dtype=np.float32)
    
    patch_idx = 0
    for y in range(0, original_shape[0] - IMG_HEIGHT + 1, IMG_HEIGHT):
        for x in range(0, original_shape[1] - IMG_WIDTH + 1, IMG_WIDTH):
            # Squeeze to remove single dimensions, e.g., (256, 256, 1) -> (256, 256)
            predicted_patch = np.squeeze(predicted_patches[patch_idx])
            prediction_full[y:y+IMG_HEIGHT, x:x+IMG_WIDTH] = predicted_patch
            patch_idx += 1

    # 6. Post-process the final prediction mask
    # Convert probabilities (0-1) to a binary mask (0 or 1) using a 0.5 threshold
    binary_mask = (prediction_full > 0.5).astype(np.uint8)

    # 7. Save the final mask as a new GeoTIFF file
    # Update metadata for the output file
    meta.update(dtype=rasterio.uint8, count=1)
    
    # Define output path
    output_filename = "predicted_uhi_mask.tif"
    output_path = os.path.join(PREDICTIONS_DIR, output_filename)
    
    print(f"Saving prediction mask to: {output_path}")
    with rasterio.open(output_path, 'w', **meta) as dst:
        # Add a new dimension to make it (1, H, W) for saving
        dst.write(binary_mask[np.newaxis, :, :])

    print("--- Prediction Finished ---")

if __name__ == '__main__':
    main()
