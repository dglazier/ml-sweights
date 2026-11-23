# core/networks.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import logging
try:
    import xgboost as xgb
except ImportError:
    xgb = None
logger = logging.getLogger(__name__)


def map_trig_topology(x, cos_theta_indices=(0,), periodic_phi_indices=(1, 2), passthrough_indices=()):
    """
    Maps periodic angles into continuous sine/cosine components while passing through
    both explicitly specified passthrough_indices and any trailing columns (e.g.
    remaining discriminating variables y_{j > k} appended during sequential unfolding).
    """
    features = []
    for idx in cos_theta_indices:
        cT = x[:, idx:idx + 1]
        sT = np.sqrt(np.maximum(1.0 - cT**2, 0.0))
        features.extend([cT, sT])
    for idx in periodic_phi_indices:
        ang = x[:, idx:idx + 1]
        features.extend([np.cos(ang), np.sin(ang)])
    for idx in passthrough_indices:
        features.append(x[:, idx:idx + 1])

    all_explicit = list(cos_theta_indices) + list(periodic_phi_indices) + list(passthrough_indices)
    max_explicit = max(all_explicit) if all_explicit else -1
    if x.shape[1] > max_explicit + 1:
        features.append(x[:, max_explicit + 1:])

    return np.hstack(features)


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
              lr=0.001, patience=5, min_delta=1e-4, loss_fn='BCE', logit_l2_penalty=0.0,
              optimizer_type='Adam'):
    """
    Trains the DNN. Supports stochastic Adam with early stopping, or 
    deterministic full-batch L-BFGS for exact mathematical optimization.
    Exactly respects signed weights in w_train without clipping or distortion.
    """
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

    def compute_loss(logits, y_batch, w_batch):
        """Helper to calculate loss consistently across Train and Val loops"""
        if loss_fn == 'BCE':
            criterion = nn.BCEWithLogitsLoss(weight=None, reduction='none')
            # Natively applies negative weights to the BCE gradient
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

    # ==============================================================================
    # DETERMINISTIC FULL-BATCH OPTIMIZATION
    # ==============================================================================
    if optimizer_type == 'LBFGS':
        if not in_vram:
            logger.warning("L-BFGS requires the full dataset in VRAM. Falling back to Adam.")
            optimizer_type = 'Adam'
        else:
            optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=epochs, line_search_fn="strong_wolfe")
            
            def closure():
                optimizer.zero_grad(set_to_none=True)
                logits = model(X_t)
                loss = compute_loss(logits, y_t, w_t)
                loss.backward()
                return loss
                
            model.train()
            optimizer.step(closure)
            return model

    # ==============================================================================
    # STOCHASTIC BATCH OPTIMIZATION
    # ==============================================================================
    optimizer = optim.Adam(model.parameters(), lr=lr)

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    if in_vram:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    best_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
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
# ==============================================================================
# SIGNED-BCE GBDT WRAPPER (drWeights Exact Loss Formulation)
# ==============================================================================
class SignedXGBClassifier:
    def __init__(self, n_estimators=50, max_depth=4, learning_rate=0.1,
                 min_child_weight=5.0, reg_lambda=2.0, device='cuda'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.device = device
        self.booster = None

    def fit(self, X, y, sample_weight):
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X, dtype=np.float32)
        y_np = y.detach().cpu().numpy().astype(np.float32) if isinstance(y, torch.Tensor) else np.asarray(y, dtype=np.float32)
        w_np = sample_weight.detach().cpu().numpy().astype(np.float32) if isinstance(sample_weight, torch.Tensor) else np.asarray(sample_weight, dtype=np.float32)

        w_scale = float(np.mean(np.abs(w_np))) + 1e-12
        w_norm = w_np / w_scale

        dtrain = xgb.DMatrix(X_np, label=y_np)
        params = {
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'tree_method': 'hist',
            'device': self.device,
            'min_child_weight': self.min_child_weight,
            'reg_lambda': self.reg_lambda,
            'base_score': 0.0,
            'verbosity': 0
        }

        def signed_bce_obj(preds: np.ndarray, dmat: xgb.DMatrix):
            z = np.clip(preds, -15.0, 15.0)
            p = 1.0 / (1.0 + np.exp(-z))
            grad = w_norm * (p - y_np)
            hess = w_norm * p * (1.0 - p)
            return grad, hess

        self.booster = xgb.train(
            params, dtrain, num_boost_round=self.n_estimators, obj=signed_bce_obj
        )
        return self

    def predict_proba(self, X):
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X, dtype=np.float32)
        dtest = xgb.DMatrix(X_np)
        margins = self.booster.predict(dtest, output_margin=True)
        z = np.clip(margins, -15.0, 15.0)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p1, p1])
