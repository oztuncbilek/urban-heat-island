import torch
import numpy as np
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import custom modules
from model import UNet
from dataset import UHIDataset

# --- CONFIGURATION ---
BATCH_SIZE = 8
CROP_SIZE = 256
THRESHOLD = 0.5  # Threshold to convert probability to binary mask

# --- DEVICE SETUP ---
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Evaluation running on: {DEVICE}")

def calculate_confusion_matrix_elements(pred_tensor, target_tensor):
    """
    Calculates TP, TN, FP, FN for binary segmentation.
    """
    # Convert probabilities to binary mask (0 or 1)
    pred_binary = (pred_tensor > THRESHOLD).float()
    target_binary = target_tensor.float()
    
    # Flatten tensors to 1D
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    # Calculate elements
    TP = (pred_flat * target_flat).sum().item()
    TN = ((1 - pred_flat) * (1 - target_flat)).sum().item()
    FP = (pred_flat * (1 - target_flat)).sum().item()
    FN = ((1 - pred_flat) * target_flat).sum().item()
    
    return TP, TN, FP, FN

if __name__ == "__main__":
    # --- PATH SETUP ---
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'src':
        PROJECT_ROOT = os.path.dirname(current_dir)
    else:
        PROJECT_ROOT = current_dir
        
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_unet_model.pth')
    OUTPUT_JSON_PATH = os.path.join(PROJECT_ROOT, 'models', 'final_scores.json')
    
    # --- LOAD VALIDATION DATA ---
    # We evaluate only on unseen years (2022, 2023, 2024)
    val_years = [2022, 2023, 2024]
    print(f"Evaluating on Validation Years: {val_years}")
    
    val_ds = UHIDataset(DATA_DIR, val_years, crop_size=CROP_SIZE, samples_per_epoch=200)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- LOAD MODEL ---
    print(f"Loading Model from {MODEL_PATH}...")
    model = UNet(n_channels=5, n_classes=1).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        print("Model file not found. Train first.")
        exit()
        
    model.eval()
    
    # --- METRICS ACCUMULATORS ---
    total_TP = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0
    
    print("Starting Final Evaluation...")
    
    with torch.no_grad():
        for data, targets in tqdm(val_loader, desc="Calculating Metrics"):
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Forward Pass
            outputs = model(data)
            probs = torch.sigmoid(outputs)
            
            # Calculate Batch Metrics
            tp, tn, fp, fn = calculate_confusion_matrix_elements(probs, targets)
            
            total_TP += tp
            total_TN += tn
            total_FP += fp
            total_FN += fn
            
    # --- FINAL CALCULATION ---
    # Avoid division by zero
    epsilon = 1e-6
    
    # 1. IoU (Intersection over Union) = TP / (TP + FP + FN)
    iou = total_TP / (total_TP + total_FP + total_FN + epsilon)
    
    # 2. Precision = TP / (TP + FP)
    precision = total_TP / (total_TP + total_FP + epsilon)
    
    # 3. Recall = TP / (TP + FN)
    recall = total_TP / (total_TP + total_FN + epsilon)
    
    # 4. F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    # 5. Accuracy = (TP + TN) / (Total)
    accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN + epsilon)
    
    results = {
        "IoU": round(iou, 4),
        "F1_Score": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Accuracy": round(accuracy, 4)
    }
    
    # --- PRINT AND SAVE ---
    print("\n" + "="*30)
    print("FINAL MODEL SCORE CARD")
    print("="*30)
    print(json.dumps(results, indent=4))
    print("="*30)
    
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Scores saved to: {OUTPUT_JSON_PATH}")