import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class PureDNN(nn.Module):
    def __init__(self, input_dim, use_relu=True, hidden_dim=128, num_layers=4):
        super(PureDNN, self).__init__()
        
        activation = nn.ReLU() if use_relu else nn.SiLU()
        
        # Dynamically build the network based on requested capacity
        layers = [nn.Linear(input_dim, hidden_dim), activation]
        
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation])
            
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_dnn(model, X_train, y_train, w_train, device, epochs=100, batch_size=2048, lr=0.001, patience=5, min_delta=1e-4, loss_fn='BCE', logit_l2_penalty=0.0):
    """
    Trains the DNN with Autonomous Early Stopping. 
    Supports BCE, uLSIF, and RuLSIF loss functions with Logit Squeezing.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Convert to tensors
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    w_t = torch.tensor(w_train, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_t, y_t, w_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for X_batch, y_batch, w_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            w_batch = w_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: network outputs unscaled logits (z = ln(R))
            logits = model(X_batch)
            
            if loss_fn == 'BCE':
                criterion = nn.BCEWithLogitsLoss(weight=None, reduction='none')
                unweighted_loss = criterion(logits, y_batch)
                loss = torch.mean(unweighted_loss * w_batch)
                
            elif loss_fn == 'uLSIF':
                # Direct Density Ratio: R = exp(z)
                R = torch.exp(logits)
                # Loss = 0.5 * E_ref[R^2] - E_data[R]
                loss_ref = 0.5 * (1.0 - y_batch) * w_batch * (R ** 2)
                loss_data = y_batch * w_batch * R
                loss = torch.mean(loss_ref - loss_data)
                
            elif loss_fn == 'RuLSIF':
                alpha = 0.5 # Standard relative blending parameter
                R = torch.exp(logits)
                
                # Transform absolute ratio R to relative ratio R_alpha
                R_alpha = R / (alpha * R + (1.0 - alpha))
                
                # Loss = 0.5 * E_mix[R_alpha^2] - E_data[R_alpha]
                loss_mix = 0.5 * w_batch * (R_alpha ** 2) * (alpha * y_batch + (1.0 - alpha) * (1.0 - y_batch))
                loss_data = y_batch * w_batch * R_alpha
                loss = torch.mean(loss_mix - loss_data)
            else:
                raise ValueError(f"Unsupported loss_fn: {loss_fn}. Use 'BCE', 'uLSIF', or 'RuLSIF'.")
            
            # Apply Direct L2 Penalty on Output Logits (Logit Squeezing)
            if logit_l2_penalty > 0.0:
                loss += logit_l2_penalty * torch.mean(logits ** 2)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
        avg_loss = epoch_loss / max(1, batch_count)
        
        # Autonomous Early Stopping
        if avg_loss < (best_loss - min_delta):
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            break
            
    return model

def predict_in_batches(model, X_tensor, batch_size=10000):
    """
    Helper function to process large datasets through the DNN without blowing up VRAM.
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
