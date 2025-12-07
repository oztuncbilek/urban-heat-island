import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import json  # Added for exporting metrics

# Import our custom modules
from model import UNet
from dataset import UHIDataset

# --- 1. CUSTOM LOSS FUNCTION (DICE LOSS) ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

# --- 2. COMBINED LOSS FUNCTION ---
def combined_loss(pred, target, bce_weight=0.5):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    pred_sigmoid = torch.sigmoid(pred)
    dice = DiceLoss()(pred_sigmoid, target)
    return (bce * bce_weight) + (dice * (1 - bce_weight))

# --- 3. TRAINING FUNCTION ---
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    epoch_loss = 0
    loop = tqdm(loader, desc="Training", leave=False)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        
        predictions = model(data)
        loss = combined_loss(predictions, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    return epoch_loss / len(loader)

# --- 4. VALIDATION FUNCTION ---
def evaluate(model, loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
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
    BATCH_SIZE = 8       
    NUM_EPOCHS = 10      # Reduced to 10 for testing
    CROP_SIZE = 256
    
    # --- PATHS ---
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'src':
        PROJECT_ROOT = os.path.dirname(current_dir)
    else:
        PROJECT_ROOT = current_dir
        
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # --- DEVICE SETUP ---
    if torch.backends.mps.is_available():
        DEVICE = "mps"
        print("Using Apple MPS")
    elif torch.cuda.is_available():
        DEVICE = "cuda"
        print("Using NVIDIA CUDA")
    else:
        DEVICE = "cpu"
        print("Using CPU")

    # --- DATASET PREPARATION ---
    all_years = list(range(2014, 2025))
    train_years = all_years[:-3]
    val_years = all_years[-3:]
    
    print(f"Training Years: {train_years}")
    print(f"Validation Years: {val_years}")
    
    train_ds = UHIDataset(DATA_DIR, train_years, crop_size=CROP_SIZE, samples_per_epoch=200)
    val_ds = UHIDataset(DATA_DIR, val_years, crop_size=CROP_SIZE, samples_per_epoch=50)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- MODEL & OPTIMIZER ---
    model = UNet(n_channels=5, n_classes=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- TRAINING LOOP ---
    best_loss = float('inf')
    
    # Dictionary to store metrics for JSON export
    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': []
    }
    
    print("\nStarting Training Loop...")

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_loss = evaluate(model, val_loader, DEVICE)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch + 1)
        
        # Save Best Model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_unet_model.pth"))
            print("Model Saved.")
            
    # --- SAVE METRICS TO JSON ---
    json_path = os.path.join(MODEL_DIR, "evaluation_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    print(f"\nTraining Complete. Metrics saved to {json_path}")