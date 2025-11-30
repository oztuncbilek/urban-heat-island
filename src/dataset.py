import os
import numpy as np
import torch
from torch.utils.data import Dataset

class UHIDataset(Dataset):
    """
    Custom Dataset for Urban Heat Island (UHI) Segmentation.
    
    It loads 6-channel tensors (.npy files) and applies random cropping
    to generate manageable patches for the U-Net model.
    
    Channels:
    0: LST (Target) -> Will be converted to Binary Mask
    1: NDVI
    2: NDBI
    3: Buildings
    4: Roads
    5: Water
    """
    def __init__(self, data_dir, years, crop_size=256, samples_per_epoch=100):
        """
        Args:
            data_dir (str): Path to 'data/processed' directory.
            years (list): List of years to include (e.g., [2014, 2015, ...]).
            crop_size (int): Size of the square patch (e.g., 256 for 256x256).
            samples_per_epoch (int): Virtual length of dataset. Since we use random crops,
                                     we define how many crops constitute "one epoch".
        """
        self.data_dir = data_dir
        self.years = years
        self.crop_size = crop_size
        self.samples_per_epoch = samples_per_epoch
        
        # Pre-load data into RAM to avoid disk I/O bottlenecks during training.
        # Since we have ~10 files of ~500MB, checking RAM is important.
        # If RAM is low, we would load inside __getitem__ (Lazy Loading).
        # For now, we will store file paths and load on demand to be safe.
        self.file_paths = [os.path.join(data_dir, f"{year}.npy") for year in years]
        
        print(f"Dataset initialized with {len(self.years)} years. Virtual size: {len(self)}")

    def __len__(self):
        # The model thinks the dataset has this many items.
        # We multiply years by samples_per_epoch to train extensively on each year.
        return len(self.years) * self.samples_per_epoch
    
    def __getitem__(self, idx):
        # 1. Select a random year file
        # We use modulo operator to cycle through years
        year_idx = idx % len(self.years)
        file_path = self.file_paths[year_idx]
        
        # 2. Load the full tensor (6, H, W)
        # Note: Ideally, we should cache this in RAM if possible, but loading from disk is safer for now.
        full_tensor = np.load(file_path)
        
        # 3. Random Crop Logic
        # Get dimensions
        _, h, w = full_tensor.shape
        
        # Pick a random top-left corner
        top = np.random.randint(0, h - self.crop_size)
        left = np.random.randint(0, w - self.crop_size)
        
        # Crop the tensor: (6, crop_size, crop_size)
        patch = full_tensor[:, top:top+self.crop_size, left:left+self.crop_size]
        
        # 4. Separate Input Features and Target
        # Channel 0 is LST (Target)
        # Channels 1-5 are Features (NDVI, NDBI, Building, Road, Water)
        
        lst_map = patch[0]      # Shape: (256, 256)
        features = patch[1:]    # Shape: (5, 256, 256)
        
        # 5. Create Binary Target Mask (The "UHI" Definition)
        # Logic: If pixel temperature > Mean + 0.5 * StdDev -> It is a Heat Island (1)
        # This is a dynamic thresholding approach.
        mean_temp = np.mean(lst_map)
        std_temp = np.std(lst_map)
        threshold = mean_temp + (0.5 * std_temp)
        
        # Create binary mask (1 for UHI, 0 for Normal)
        binary_mask = (lst_map > threshold).astype(np.float32)
        
        # Add channel dimension to mask: (1, 256, 256) needed for U-Net output
        binary_mask = np.expand_dims(binary_mask, axis=0)
        
        # 6. Convert to PyTorch Tensors
        input_tensor = torch.from_numpy(features)
        target_tensor = torch.from_numpy(binary_mask)
        
        return input_tensor, target_tensor
    

if __name__ == "__main__":
    # Test the Dataset
    # Define paths (assuming running from project root)
    root_dir = os.path.dirname(os.getcwd()) # Go up one level if running from src
    # Correction: If running as "python src/dataset.py", getcwd is project root usually.
    # Let's assume absolute path safety or relative from execution.
    
    processed_path = "data/processed"
    years_to_test = [2023] # Test with just one year
    
    if os.path.exists(processed_path):
        dataset = UHIDataset(processed_path, years_to_test, crop_size=256)
        
        # Get one sample
        img, mask = dataset[0]
        
        print(f"Input Shape: {img.shape}") # Should be (5, 256, 256)
        print(f"Target Shape: {mask.shape}") # Should be (1, 256, 256)
        print(f"Max Value in Mask: {mask.max()}") # Should be 1.0
        print("Dataset implementation successful!")
    else:
        print(f"Path not found: {processed_path}. Make sure you run this from the project root.")