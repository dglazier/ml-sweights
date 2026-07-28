import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import logging
import zuko

logger = logging.getLogger(__name__)

def train_flow_model(flow_model, Z_train, y_train, base_loc, base_scale, epochs=30, batch_size=8192, device='cuda', lr=1e-3):
    logger.info("\nTraining Conditional Normalizing Flow on Base Reference Data...")
    opt = torch.optim.Adam(flow_model.parameters(), lr=lr)
    
    Z_t = torch.tensor(Z_train, dtype=torch.float32, device=device)
    y_t = torch.tensor((y_train - base_loc) / base_scale, dtype=torch.float32, device=device).unsqueeze(1)
    loader = DataLoader(TensorDataset(Z_t, y_t), batch_size=batch_size, shuffle=True)
    
    flow_model.train()
    for ep in range(epochs): 
        for z_b, y_b in loader:
            opt.zero_grad(set_to_none=True)
            loss = -flow_model(z_b).log_prob(y_b).mean()
            loss.backward()
            opt.step()
            
    flow_model.eval()
    return flow_model

class DynamicFlowWrapper:
    """
    Wraps the pre-trained CNF. Calculates event-by-event kinematic distortions.
    Automatically detected and trained by the EM Loop via the .update() method.
    """
    def __init__(self, flow_model, n_context, base_loc, base_scale, init_shift, init_scale, device='cuda'):
        self.flow_model = flow_model
        self.base_loc = base_loc
        self.base_scale = base_scale
        self.init_shift = init_shift
        self.init_scale = init_scale
        self.device = device
        
        # Freeze the base Monte Carlo flow
        for param in self.flow_model.parameters():
            param.requires_grad = False
        self.flow_model.eval()
        
        # The Calibration Network: Maps Z -> [delta_shift, delta_log_scale]
        self.calib_net = nn.Sequential(
            nn.Linear(n_context, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 2)
        ).to(device)
        
        # IMPORTANT: Initialize weights/bias to zero. 
        # This guarantees it starts EXACTLY at the global 1D fit!
        self.calib_net[-1].weight.data.zero_()
        self.calib_net[-1].bias.data.zero_()
        
        self.optimizer = torch.optim.Adam(self.calib_net.parameters(), lr=1e-3)
        
    def __call__(self, Z_arr, y_arr):
        """Evaluates the Z-dependent signal probability."""
        self.calib_net.eval()
        z_t = torch.tensor(Z_arr, dtype=torch.float32, device=self.device)
        y_flat = np.array(y_arr).flatten()
        y_t = torch.tensor(y_flat, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        log_p = []
        with torch.no_grad():
            for i in range(0, len(y_flat), 32768):
                z_b = z_t[i:i+32768]
                y_b = y_t[i:i+32768]
                
                out = self.calib_net(z_b)
                delta_shift = out[:, 0:1]
                delta_log_scale = out[:, 1:2]
                
                # Apply the Z-dependent distortions to the global baseline
                total_shift = self.init_shift + delta_shift
                total_scale = self.init_scale * torch.exp(delta_log_scale)
                
                # Dynamically map the data mass into the MC reference frame
                y_eq = self.base_loc + (y_b - (self.base_loc + total_shift)) / total_scale
                y_norm = (y_eq - self.base_loc) / self.base_scale
                
                # Evaluate probability and scale by Jacobian to conserve probability
                log_prob_base = self.flow_model(z_b).log_prob(y_norm)
                log_prob_final = log_prob_base - torch.log(self.base_scale * total_scale).squeeze()
                
                log_p.append(log_prob_final)
                
        return torch.exp(torch.cat(log_p)).cpu().numpy().flatten()
        
    def update(self, Z_arr, y_arr, w_s_arr, epochs=5, batch_size=8192, penalty_strength=10.0):
        """Called autonomously by the EM Loop to warp the parameters using signal weights."""
        self.calib_net.train()
        z_t = torch.tensor(Z_arr, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_arr, dtype=torch.float32, device=self.device).view(-1, 1)
        w_t = torch.tensor(w_s_arr, dtype=torch.float32, device=self.device).view(-1, 1)
        
        dataset = TensorDataset(z_t, y_t, w_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for ep in range(epochs):
            for z_b, y_b, w_b in loader:
                self.optimizer.zero_grad(set_to_none=True)
                
                out = self.calib_net(z_b)
                delta_shift = out[:, 0:1]
                delta_log_scale = out[:, 1:2]
                
                total_shift = self.init_shift + delta_shift
                total_scale = self.init_scale * torch.exp(delta_log_scale)
                
                y_eq = self.base_loc + (y_b - (self.base_loc + total_shift)) / total_scale
                y_norm = (y_eq - self.base_loc) / self.base_scale
                
                log_prob_base = self.flow_model(z_b).log_prob(y_norm)
                log_prob_final = log_prob_base - torch.log(self.base_scale * total_scale).squeeze()
                
                # ---------------------------------------------------------
                # NEW: Nuisance Parameter Regularization
                # Penalizes the network for straying from the 1D global fit
                # ---------------------------------------------------------
                # delta_shift is squared directly
                # delta_log_scale is squared (0 means scale = 1.0)
                l2_nuisance_penalty = penalty_strength * torch.mean(delta_shift**2 + delta_log_scale**2)
                
                # Maximize Negative Log-Likelihood + Penalty
                loss = -torch.mean(log_prob_final * w_b.squeeze()) + l2_nuisance_penalty
                
                loss.backward()
                self.optimizer.step() 

def calibrate_flow_model(flow_model, Z_data, y_data, bkg_pdf_func, f0, base_loc, base_scale, peak_loc, epochs=50, batch_size=16384, device='cuda'):
    logger.info("\nExecuting Conditional Likelihood CNF Calibration...")
    
    Z_t = torch.tensor(Z_data, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_data, dtype=torch.float32, device=device).unsqueeze(1)
    f0_t = torch.tensor(f0, dtype=torch.float32, device=device)
    
    # Fit the Global Averages first
    shift_param = nn.Parameter(torch.tensor(peak_loc - base_loc, dtype=torch.float32, device=device))
    log_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
    
    opt_calib = torch.optim.Adam([shift_param, log_scale], lr=0.005) 
    loader_calib = DataLoader(TensorDataset(Z_t, y_t), batch_size=batch_size, shuffle=True)
    
    for ep in range(epochs): 
        for z_b, y_b in loader_calib:
            opt_calib.zero_grad(set_to_none=True)
            scale = torch.exp(log_scale)
            shift = shift_param
            
            fb = torch.tensor(bkg_pdf_func(y_b.cpu().numpy().flatten()), dtype=torch.float32, device=device).unsqueeze(1)
            y_eq = base_loc + (y_b - (base_loc + shift)) / scale
            y_norm = (y_eq - base_loc) / base_scale
            
            fs = torch.exp(flow_model(z_b).log_prob(y_norm)) / (base_scale * scale)
            loss = -torch.log(f0_t * fs + (1 - f0_t) * fb + 1e-10).mean()
            loss.backward()
            opt_calib.step()
            
    shift_val = shift_param.item()
    scale_val = torch.exp(log_scale).item()
    
    logger.info(f"================= CNF CALIBRATION DIAGNOSTICS =================")
    logger.info(f"Extracted Global Nuisance: Shift = {shift_val*1000:.1f} (Units), Smear = {(scale_val - 1.0)*100:.1f}%")
    logger.info(f"===============================================================\n")
    
    # Return the Dynamic Wrapper instead of the static closure!
    n_context = Z_data.shape[1]
    return DynamicFlowWrapper(flow_model, n_context, base_loc, base_scale, shift_val, scale_val, device)

def build_signal_flow(Z_mc, y_mc, base_loc, base_scale, peak_loc, device='cuda'):
    """
    High-level wrapper to instantiate and train the Normalizing Flow on MC.
    Initializes the dynamic wrapper strictly at the 1D Anchor (no pre-calibration).
    """
    n_context = Z_mc.shape[1]
    
    # 1. Initialize and Train Flow Network on pure MC
    flow_cnf = zuko.flows.NSF(features=1, context=n_context, hidden_features=[64, 64], bins=8).to(device)
    train_flow_model(flow_cnf, Z_mc, y_mc, base_loc, base_scale, device=device)
    
    # 2. Instantiate Dynamic Wrapper locked to the 1D MINUIT Anchor
    init_shift = peak_loc - base_loc
    sig_pdf_Z = DynamicFlowWrapper(
        flow_model=flow_cnf, 
        n_context=n_context,     # <-- ADDED: Wrapper needs this for its internal network
        base_loc=base_loc, 
        base_scale=base_scale, 
        init_shift=init_shift, 
        init_scale=1.0, 
        device=device
    )
    
    return sig_pdf_Z
