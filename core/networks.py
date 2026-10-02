import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
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

def train_dnn(model, X_train, y_train, w_train, device, epochs=100, batch_size=2048, 
              lr=0.001, patience=5, min_delta=1e-4, loss_fn='BCE', logit_l2_penalty=0.0):
    """
    Trains the DNN with Autonomous Early Stopping. 
    Autonomously handles VRAM loading or asynchronous PCIe transfers if data is too large.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    in_vram = True
    
    # 1. ATTEMPT TO PUSH ENTIRE DATASET TO VRAM
    if 'cuda' in str(device):
        try:
            X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
            y_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
            w_t = torch.tensor(w_train, dtype=torch.float32, device=device).unsqueeze(1)
            
            dataset = TensorDataset(X_t, y_t, w_t)
            # num_workers MUST be 0 if tensors are already on the GPU
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
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
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=True, # Page-locks memory for Direct Memory Access
            num_workers=2    # Background data fetching
        )
    
    best_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for X_batch, y_batch, w_batch in dataloader:
            if not in_vram:
                # non_blocking=True allows computation and transfer to overlap
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                w_batch = w_batch.to(device, non_blocking=True)
            
            # set_to_none=True reduces memory footprint compared to zeroing
            optimizer.zero_grad(set_to_none=True)
            
            logits = model(X_batch)
            
            if loss_fn == 'BCE':
                criterion = nn.BCEWithLogitsLoss(weight=None, reduction='none')
                unweighted_loss = criterion(logits, y_batch)
                loss = torch.mean(unweighted_loss * w_batch)
                
            elif loss_fn == 'uLSIF':
                # FIXED: Added clamping to prevent exp(logits) from exploding to Infinity
                R = torch.exp(torch.clamp(logits, -15.0, 15.0))
                # FIXED: Batch-normalized expectations based on class weights
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
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
        avg_loss = epoch_loss / max(1, batch_count)
        
        if avg_loss < (best_loss - min_delta):
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            break
            
    return model

def predict_in_batches(model, X_tensor, batch_size=32768):
    """
    Helper function to process large datasets through the DNN.
    Increased batch size significantly to fully saturate GPU cores.
    """
    model.eval()
    n_samples = X_tensor.shape[0]
    preds = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            X_batch = X_tensor[i:i+batch_size]
            logits = model(X_batch)
            preds.append(logits)
            
    return torch.cat(preds, dim=0)
