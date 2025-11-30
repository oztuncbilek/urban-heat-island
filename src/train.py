import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # For progress bars
import numpy as np
import os

# Import our custom modules
from model import UNet
from dataset import UHIDataset

# --- 1. CUSTOM LOSS FUNCTION (DICE LOSS) ---
class DiceLoss(nn.Module):
    """
    Dice Loss helps with class imbalance and focuses on the overlap
    between the predicted segmentation mask and the ground truth.
    Formula: 2 * (Intersection) / (Union)
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # Epsilon to avoid division by zero

    def forward(self, inputs, targets):
        # inputs: Model predictions (logits or probabilities)
        # targets: Ground truth binary mask
        
        # Flatten the tensors to 1D vectors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection
        intersection = (inputs * targets).sum()
        
        # Calculate Dice Coefficient
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        # Return Dice Loss (1 - Dice Coefficient)
        return 1 - dice

# --- 2. COMBINED LOSS FUNCTION ---
def combined_loss(pred, target, bce_weight=0.5):
    """
    Combines Binary Cross Entropy (BCE) and Dice Loss.
    BCE focuses on pixel-level accuracy.
    Dice focuses on shape/overlap accuracy.
    """
    # 1. Binary Cross Entropy (with built-in Sigmoid)
    bce = nn.BCEWithLogitsLoss()(pred, target)
    
    # 2. Dice Loss (Needs manual Sigmoid activation)
    pred_sigmoid = torch.sigmoid(pred)
    dice = DiceLoss()(pred_sigmoid, target)
    
    # 3. Weighted Sum
    return (bce * bce_weight) + (dice * (1 - bce_weight))

# --- 3. TRAINING FUNCTION (ONE EPOCH) ---
def train_one_epoch(model, loader, optimizer, device):
    """
    Runs one full pass over the training dataset.
    """
    model.train()  # Set model to training mode (enables Dropout/BatchNorm)
    epoch_loss = 0
    
    # Initialize progress bar
    loop = tqdm(loader, desc="Training", leave=False)
    
    for batch_idx, (data, targets) in enumerate(loop):
        # Move data to the selected device (GPU/MPS/CPU)
        data = data.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        
        # A. Forward Pass
        predictions = model(data)
        
        # B. Calculate Loss
        loss = combined_loss(predictions, targets)
        
        # C. Backward Pass (Backpropagation)
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Calculate new gradients
        optimizer.step()       # Update weights
        
        # Update statistics
        epoch_loss += loss.item()
        
        # Update progress bar
        loop.set_postfix(loss=loss.item())
    
    return epoch_loss / len(loader)

# --- 4. VALIDATION FUNCTION ---
def evaluate(model, loader, device):
    """
    Evaluates the model on the validation dataset.
    No gradients are calculated here.
    """
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    
    with torch.no_grad():  # Disable gradient calculation for efficiency
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)
            
            preds = model(data)
            loss = combined_loss(preds, targets)
            val_loss += loss.item()
            
    return val_loss / len(loader)

# --- 5. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # --- HYPERPARAMETERS ---
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8       # Reduce to 4 or 2 if you run out of memory
    NUM_EPOCHS = 10      # Number of complete passes through the dataset
    CROP_SIZE = 256
    
    # --- PATHS ---
    # Assuming this script is run from the project root or src folder
    # We navigate up one level if we are in 'src' to find 'data'
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'src':
        PROJECT_ROOT = os.path.dirname(current_dir)
    else:
        PROJECT_ROOT = current_dir
        
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    
    # --- DEVICE SETUP ---
    # Check for Apple Silicon (MPS) or NVIDIA (CUDA)
    if torch.backends.mps.is_available():
        DEVICE = "mps"
        print("Using Apple Metal Performance Shaders (MPS)")
    elif torch.cuda.is_available():
        DEVICE = "cuda"
        print("Using NVIDIA CUDA")
    else:
        DEVICE = "cpu"
        print("Using CPU (Slow training)")

    # --- DATASET PREPARATION ---
    # Full range: 2014-2024
    all_years = list(range(2014, 2025))
    
    # Training: 2014-2021 (8 Years) - The model learns from this
    # Validation: 2022-2024 (3 Years) - The model is tested on this (Future prediction)
    train_years = all_years[:-3]
    val_years = all_years[-3:]
    
    print(f"Training Years: {train_years}")
    print(f"Validation Years: {val_years}")
    
    # Initialize Datasets
    # samples_per_epoch defines how many random crops we take per year per epoch
    train_ds = UHIDataset(DATA_DIR, train_years, crop_size=CROP_SIZE, samples_per_epoch=200)
    val_ds = UHIDataset(DATA_DIR, val_years, crop_size=CROP_SIZE, samples_per_epoch=50)
    
    # Initialize DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- MODEL INITIALIZATION ---
    # Input Channels: 5 (NDVI, NDBI, Buildings, Roads, Water)
    # Output Class: 1 (Binary UHI Mask)
    model = UNet(n_channels=5, n_classes=1).to(DEVICE)

    # Optimizer (Adam is standard for U-Net)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- TRAINING LOOP ---
    best_loss = float('inf')
    
    print("\nStarting Training Loop...")

    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    for epoch in range(NUM_EPOCHS):
        # Run training and validation
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_loss = evaluate(model, val_loader, DEVICE)
        
        # Log results
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save Best Model Logic
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = os.path.join(MODEL_DIR, "best_unet_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model Saved to {save_path}")
            
    print("\nTraining Complete! Best weights saved as 'best_unet_model.pth'")