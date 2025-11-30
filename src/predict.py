import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import the model architecture
from model import UNet

# --- CONFIGURATION ---
CROP_SIZE = 256        # Size of the patches fed into the model
BATCH_SIZE = 1         # Keep it 1 for inference to avoid complexity with stitching
THRESHOLD = 0.5        # Threshold to convert probability to binary mask

# --- DEVICE CONFIGURATION ---
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Inference running on: {DEVICE}")

def predict_full_map(model, full_tensor, device, patch_size=256):
    """
    Takes a massive tensor (Channels, Height, Width), splits it into smaller patches,
    runs inference on each patch, and stitches them back into a full-size prediction map.
    
    Args:
        model: Trained PyTorch U-Net model.
        full_tensor: Numpy array of shape (5, H, W). The input features.
        device: 'cuda', 'mps', or 'cpu'.
        patch_size: The size of the square crop (default 256).
        
    Returns:
        prediction_map: Numpy array of shape (H, W) containing probability values (0.0 - 1.0).
    """
    model.eval()  # Set model to evaluation mode (disable dropout/batchnorm updates)
    
    channels, height, width = full_tensor.shape
    
    # Initialize an empty canvas for the result
    prediction_map = np.zeros((height, width), dtype=np.float32)
    
    # --- SLIDING WINDOW / PATCHING LOOP ---
    # We iterate over the image in steps of 'patch_size'
    print(f"Processing Image Size: {height}x{width}")
    
    for y in tqdm(range(0, height, patch_size), desc="Scanning Map"):
        for x in range(0, width, patch_size):
            
            # Determine patch coordinates
            y_end = min(y + patch_size, height)
            x_end = min(x + patch_size, width)
            
            # Handle boundary conditions:
            # If we are at the edge and the remaining size is smaller than patch_size,
            # we shift back to grab a full 256x256 patch.
            y_start = y
            x_start = x
            
            if (y_end - y_start) < patch_size:
                y_start = height - patch_size
            if (x_end - x_start) < patch_size:
                x_start = width - patch_size
            
            # Crop the input features
            patch = full_tensor[:, y_start:y_start+patch_size, x_start:x_start+patch_size]
            
            # Convert to Tensor and move to Device
            # Add Batch Dimension: (5, 256, 256) -> (1, 5, 256, 256)
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device, dtype=torch.float32)
            
            # --- INFERENCE ---
            with torch.no_grad():
                output = model(patch_tensor)
                # Apply Sigmoid to convert logits to probability (0 to 1)
                prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
            
            # --- STITCHING ---
            # Place the prediction back into the large map
            prediction_map[y_start:y_start+patch_size, x_start:x_start+patch_size] = prob_map
            
    return prediction_map

if __name__ == "__main__":
    # --- PATH SETUP ---
    # Determine the project root automatically
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'src':
        PROJECT_ROOT = os.path.dirname(current_dir)
    else:
        PROJECT_ROOT = current_dir
    
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_unet_model.pth')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'predictions')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 1. LOAD MODEL ---
    print(f"Loading Model from {MODEL_PATH}...")
    # Initialize model structure (Must match training: 5 input channels)
    model = UNet(n_channels=5, n_classes=1).to(DEVICE)
    
    # Load trained weights
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()
    
    # --- 2. SELECT DATA FOR INFERENCE ---
    # We use a year from the validation set (e.g., 2023) to see how well it generalizes
    TEST_YEAR = 2023 
    print(f"Loading Data for Year: {TEST_YEAR}")
    
    file_path = os.path.join(PROCESSED_DIR, f"{TEST_YEAR}.npy")
    if not os.path.exists(file_path):
        print(f"Data file not found: {file_path}")
        exit()
        
    full_data = np.load(file_path)
    
    # Split Ground Truth (LST) and Input Features
    # Channel 0: LST (Ground Truth for visual comparison)
    # Channel 1-5: NDVI, NDBI, Building, Road, Water (Input Features)
    ground_truth_lst = full_data[0]
    input_features = full_data[1:] 
    
    # --- 3. RUN PREDICTION ---
    print("Running Inference...")
    prediction = predict_full_map(model, input_features, DEVICE)
    
    # --- 4. SAVE RESULTS ---
    print("Saving Results...")
    
    # Create Binary Mask based on threshold
    binary_prediction = (prediction > THRESHOLD).astype(np.uint8)
    
    # Save raw probability map and binary mask as .npy for future analysis
    np.save(os.path.join(OUTPUT_DIR, f"pred_{TEST_YEAR}.npy"), prediction)
    np.save(os.path.join(OUTPUT_DIR, f"mask_{TEST_YEAR}.npy"), binary_prediction)
    
    print(f"Numerical results saved to {OUTPUT_DIR}")
    
    # --- 5. GENERATE PREVIEW IMAGE (DEBUG) ---
    print("Generating preview image...")
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Ground Truth LST (What really happened)
    plt.subplot(1, 3, 1)
    plt.title(f"Ground Truth LST ({TEST_YEAR})")
    plt.imshow(ground_truth_lst, cmap='inferno')
    plt.axis('off')
    
    # Plot 2: Model Probability (What model thinks)
    plt.subplot(1, 3, 2)
    plt.title("AI Prediction (Probability)")
    plt.imshow(prediction, cmap='jet', vmin=0, vmax=1)
    plt.axis('off')
    
    # Plot 3: Final Binary Mask (Risk Zones)
    plt.subplot(1, 3, 3)
    plt.title("Detected UHI Zones (Binary)")
    plt.imshow(binary_prediction, cmap='gray')
    plt.axis('off')
    
    preview_path = os.path.join(OUTPUT_DIR, f"preview_{TEST_YEAR}.png")
    plt.savefig(preview_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Preview saved: {preview_path}")