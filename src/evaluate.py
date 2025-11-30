import torch
import numpy as np
import os
import json
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import our custom modules
from model import UNet
from dataset import UHIDataset

# --- DEVICE SETUP ---
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Evaluation running on: {DEVICE}")

def calculate_metrics(pred_tensor, target_tensor, threshold=0.5):
    """
    Calculates True Positives, False Positives, False Negatives for binary segmentation.
    
    Args:
        pred_tensor: Probabilities from model (0.0 - 1.0)
        target_tensor: Ground truth binary mask (0 or 1)
        threshold: Cutoff value to decide class (default 0.5)
    
    Returns:
        TP, FP, FN, TN (Integers)
    """
    # Convert probabilities to binary mask (0 or 1)
    pred_mask = (pred_tensor > threshold).float()
    
    # Flatten tensors to simplify calculation (treat as 1D array)
    pred_flat = pred_mask.view(-1)
    target_flat = target_tensor.view(-1)
    
    # Calculate confusion matrix components
    # TP: Model said 1, Truth is 1
    TP = (pred_flat * target_flat).sum().item()
    
    # FP: Model said 1, Truth is 0 (False Alarm)
    FP = ((pred_flat == 1) & (target_flat == 0)).sum().item()
    
    # FN: Model said 0, Truth is 1 (Missed it)
    FN = ((pred_flat == 0) & (target_flat == 1)).sum().item()
    
    # TN: Model said 0, Truth is 0 (Correctly ignored)
    TN = ((pred_flat == 0) & (target_flat == 0)).sum().item()
    
    return TP, FP, FN, TN

def evaluate_model(model, loader, device):
    model.eval() # Set to evaluation mode
    
    # Global counters for the entire dataset
    total_TP, total_FP, total_FN, total_TN = 0, 0, 0, 0
    
    print("Evaluating model on validation set...")
    
    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Testing Batches"):
            data = data.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            
            # Forward pass
            outputs = model(data)
            probs = torch.sigmoid(outputs) # Convert logits to probability
            
            # Calculate metrics for this batch
            tp, fp, fn, tn = calculate_metrics(probs, targets)
            
            # Accumulate totals
            total_TP += tp
            total_FP += fp
            total_FN += fn
            total_TN += tn
            
    # --- FINAL METRIC CALCULATION ---
    # Add epsilon to avoid division by zero
    epsilon = 1e-6
    
    # 1. IoU (Intersection over Union) - The most important metric for segmentation
    iou = total_TP / (total_TP + total_FP + total_FN + epsilon)
    
    # 2. Precision (Ne kadar güvenilir?) -> "Sıcak dediysem sıcaktır."
    precision = total_TP / (total_TP + total_FP + epsilon)
    
    # 3. Recall (Ne kadarını yakaladı?) -> "Sıcak olanların %kaçı yakalandı?"
    recall = total_TP / (total_TP + total_FN + epsilon)
    
    # 4. F1 Score (Precision ve Recall'un harmonik ortalaması)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    
    # 5. Pixel Accuracy (Genel doğruluk - dikkatli yorumlanmalı)
    accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN + epsilon)
    
    results = {
        "IoU": round(iou, 4),
        "F1_Score": round(f1_score, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Accuracy": round(accuracy, 4)
    }
    
    return results

if __name__ == "__main__":
    # --- PATHS ---
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'src':
        PROJECT_ROOT = os.path.dirname(current_dir)
    else:
        PROJECT_ROOT = current_dir
        
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_unet_model.pth')
    
    # --- CONFIG ---
    BATCH_SIZE = 8
    CROP_SIZE = 256
    VAL_YEARS = [2022, 2023, 2024] # Validation Years
    
    # --- LOAD DATASET ---
    print(f"Loading Validation Data: {VAL_YEARS}")
    val_ds = UHIDataset(PROCESSED_DIR, VAL_YEARS, crop_size=CROP_SIZE, samples_per_epoch=100)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- LOAD MODEL ---
    print("Loading Model...")
    model = UNet(n_channels=5, n_classes=1).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Weights loaded.")
    except FileNotFoundError:
        print(f"Model not found at {MODEL_PATH}")
        exit()
        
    # --- RUN EVALUATION ---
    metrics = evaluate_model(model, val_loader, DEVICE)
    
    # --- PRINT REPORT ---
    print("\n" + "="*30)
    print("FINAL EVALUATION REPORT")
    print("="*30)
    for k, v in metrics.items():
        print(f"{k:<10}: {v}")
    print("="*30)
    
    # --- SAVE REPORT ---
    # Save as JSON for the paper/notebook
    report_path = os.path.join(PROJECT_ROOT, 'data', 'predictions', 'evaluation_metrics.json')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Report saved to: {report_path}")