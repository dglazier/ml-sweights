import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PureDNN(nn.Module):
    def __init__(self, input_dim, use_relu=True, hidden_dim=128, num_layers=4):
        super(PureDNN, self).__init__()
        
        activation = nn.ReLU() if use_relu else nn.SiLU()
        
        layers = [nn.Linear(input_dim, hidden_dim), activation]
        
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation])
            
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_dnn(model, X_train, y_train, w_train, device, epochs=100, batch_size=32768, 
              lr=0.001, patience=5, min_delta=1e-4, loss_fn='BCE', logit_l2_penalty=0.0):
    """
    Trains the DNN with Autonomous Early Stopping evaluated on a 90/10 validation split.
    Autonomously handles VRAM loading or asynchronous PCIe transfers if data is too large.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    in_vram = True
    
    # 1. ATTEMPT TO PUSH ENTIRE DATASET TO VRAM
    if 'cuda' in str(device):
        try:
            X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
            y_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
            w_t = torch.tensor(w_train, dtype=torch.float32, device=device).unsqueeze(1)
            dataset = TensorDataset(X_t, y_t, w_t)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.info("CUDA OOM detected. Falling back to pinned System RAM.")
                in_vram = False
                torch.cuda.empty_cache()
            else:
                raise e
    else:
        in_vram = False

    # 2. FALLBACK: ASYNCHRONOUS PINNED MEMORY CPU -> GPU
    if not in_vram:
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        w_t = torch.tensor(w_train, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(X_t, y_t, w_t)
        
    # 3. CREATE 90/10 VALIDATION SPLIT
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    if in_vram:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    def compute_loss(logits, y_batch, w_batch):
        """Helper to calculate loss consistently across Train and Val loops"""
        if loss_fn == 'BCE':
            criterion = nn.BCEWithLogitsLoss(weight=None, reduction='none')
            loss = torch.mean(criterion(logits, y_batch) * w_batch)
        elif loss_fn == 'uLSIF':
            R = torch.exp(torch.clamp(logits, -15.0, 15.0))
            sum_w_ref = torch.sum((1.0 - y_batch) * w_batch) + 1e-8
            sum_w_data = torch.sum(y_batch * w_batch) + 1e-8
            loss_ref = 0.5 * torch.sum((1.0 - y_batch) * w_batch * (R ** 2)) / sum_w_ref
            loss_data = torch.sum(y_batch * w_batch * R) / sum_w_data
            loss = loss_ref - loss_data
        elif loss_fn == 'RuLSIF':
            alpha = 0.5 
            R = torch.exp(torch.clamp(logits, -15.0, 15.0))
            R_alpha = R / (alpha * R + (1.0 - alpha))
            sum_w_mix = torch.sum(w_batch * (alpha * y_batch + (1.0 - alpha) * (1.0 - y_batch))) + 1e-8
            sum_w_data = torch.sum(y_batch * w_batch) + 1e-8
            loss_mix = 0.5 * torch.sum(w_batch * (R_alpha ** 2) * (alpha * y_batch + (1.0 - alpha) * (1.0 - y_batch))) / sum_w_mix
            loss_data = torch.sum(y_batch * w_batch * R_alpha) / sum_w_data
            loss = loss_mix - loss_data
        else:
            raise ValueError(f"Unsupported loss_fn: {loss_fn}")
            
        if logit_l2_penalty > 0.0:
            loss += logit_l2_penalty * torch.mean(logits ** 2)
            
        return loss

    best_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        for X_batch, y_batch, w_batch in train_loader:
            if not in_vram:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                w_batch = w_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)
            loss = compute_loss(logits, y_batch, w_batch)
            loss.backward()
            optimizer.step()
            
        # --- VALIDATION PHASE (Autonomous Thermostat) ---
        model.eval()
        val_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for X_batch, y_batch, w_batch in val_loader:
                if not in_vram:
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)
                    w_batch = w_batch.to(device, non_blocking=True)
                
                logits = model(X_batch)
                loss = compute_loss(logits, y_batch, w_batch)
                val_loss += loss.item()
                batch_count += 1
                
        avg_val_loss = val_loss / max(1, batch_count)
        
        # Monitor Validation Loss to prevent statistical overfitting
        if avg_val_loss < (best_loss - min_delta):
            best_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            break
            
    return model

def predict_in_batches(model, X_tensor, batch_size=32768):
    """Helper function to process large datasets through the DNN."""
    model.eval()
    n_samples = X_tensor.shape[0]
    preds = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            X_batch = X_tensor[i:i+batch_size]
            logits = model(X_batch)
            preds.append(logits)
            
    return torch.cat(preds, dim=0)
