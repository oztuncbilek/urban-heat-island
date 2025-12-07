import torch
import numpy as np
import os
import glob
import rasterio
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import the model architecture
from model import UNet

# --- CONFIGURATION ---
CROP_SIZE = 256
BATCH_SIZE = 1
THRESHOLD = 0.5

# --- DEVICE CONFIGURATION ---
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Inference running on: {DEVICE}")

# --- HELPER: SAVE GEOTIFF ---
def save_prediction_as_geotiff(prediction_array, year, output_dir, reference_path):
    if not os.path.exists(reference_path):
        print(f"Reference file not found: {reference_path}. Cannot save GeoTIFF.")
        return

    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
        
    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw'
    )
    
    filename = f"UHI_Prediction_{year}.tif"
    save_path = os.path.join(output_dir, filename)
    
    with rasterio.open(save_path, 'w', **profile) as dst:
        dst.write(prediction_array.astype(rasterio.float32), 1)
        
    print(f"GeoTIFF Saved: {filename}")

def predict_full_map(model, full_tensor, device, patch_size=256):
    model.eval()
    channels, height, width = full_tensor.shape
    prediction_map = np.zeros((height, width), dtype=np.float32)
    
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            y_end = min(y + patch_size, height)
            x_end = min(x + patch_size, width)
            y_start = y
            x_start = x
            
            if (y_end - y_start) < patch_size:
                y_start = height - patch_size
            if (x_end - x_start) < patch_size:
                x_start = width - patch_size
            
            patch = full_tensor[:, y_start:y_start+patch_size, x_start:x_start+patch_size]
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device, dtype=torch.float32)
            
            with torch.no_grad():
                output = model(patch_tensor)
                prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
            
            prediction_map[y_start:y_start+patch_size, x_start:x_start+patch_size] = prob_map
            
    return prediction_map

if __name__ == "__main__":
    # --- PATH SETUP ---
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'src':
        PROJECT_ROOT = os.path.dirname(current_dir)
    else:
        PROJECT_ROOT = current_dir
    
    # Input Paths
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_unet_model.pth')
    RAW_OPTICAL_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'sentinel_2')
    
    # Output Paths (Organized Subfolders)
    BASE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'predictions')
    
    DIR_NPY = os.path.join(BASE_OUTPUT_DIR, 'numpy')
    DIR_TIF = os.path.join(BASE_OUTPUT_DIR, 'geotiff')
    DIR_PNG = os.path.join(BASE_OUTPUT_DIR, 'preview')
    
    for d in [DIR_NPY, DIR_TIF, DIR_PNG]:
        os.makedirs(d, exist_ok=True)
    
    # --- 1. LOAD MODEL ---
    print(f"Loading Model from {MODEL_PATH}...")
    model = UNet(n_channels=5, n_classes=1).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()
    else:
        print(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        exit()
    
    # --- 2. FIND FILES ---
    search_pattern = os.path.join(PROCESSED_DIR, "*_tensor.npy")
    files = glob.glob(search_pattern)
    
    if not files:
        # Fallback for generic names like 2014.npy
        search_pattern = os.path.join(PROCESSED_DIR, "*.npy")
        all_npy = glob.glob(search_pattern)
        files = [f for f in all_npy if os.path.basename(f)[0].isdigit()]

    files.sort()
    
    if not files:
        print("No .npy files found in processed directory.")
        exit()
        
    print(f"Found {len(files)} years to process.")

    # --- 3. MAIN PREDICTION LOOP ---
    for file_path in tqdm(files, desc="Total Progress"):
        # Correct Year Extraction (Fixes the .npy.npy issue)
        filename_base = os.path.basename(file_path) # e.g., "2014.npy" or "2014_tensor.npy"
        filename_no_ext = os.path.splitext(filename_base)[0] # e.g., "2014" or "2014_tensor"
        year_str = filename_no_ext.split('_')[0] # Always "2014"
        
        # Load Data
        full_data = np.load(file_path)
        ground_truth_lst = full_data[0]
        input_features = full_data[1:] 
        
        # Predict
        prediction = predict_full_map(model, input_features, DEVICE)
        binary_prediction = (prediction > THRESHOLD).astype(np.uint8)
        
        # --- SAVE RESULTS (To Specific Folders) ---
        
        # A. Save Numpy
        np.save(os.path.join(DIR_NPY, f"pred_{year_str}.npy"), prediction)
        np.save(os.path.join(DIR_NPY, f"mask_{year_str}.npy"), binary_prediction)
        
        # B. Save GeoTIFF
        ref_tif_path = os.path.join(RAW_OPTICAL_DIR, f"Hamburg_Composite_{year_str}.tif")
        save_prediction_as_geotiff(prediction, year_str, DIR_TIF, ref_tif_path)
        
        # C. Save Preview PNG
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title(f"{year_str} Ground Truth (LST)")
        plt.imshow(ground_truth_lst, cmap='inferno')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title(f"{year_str} UHI Prediction")
        plt.imshow(prediction, cmap='jet', vmin=0, vmax=1)
        plt.axis('off')
        
        plt.savefig(os.path.join(DIR_PNG, f"preview_{year_str}.png"), dpi=100, bbox_inches='tight')
        plt.close()

    print(f"All predictions completed.")
    print(f"Numpy files -> {DIR_NPY}")
    print(f"GeoTIFF files -> {DIR_TIF}")
    print(f"Previews -> {DIR_PNG}")