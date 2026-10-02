import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import logging
import zuko

logger = logging.getLogger(__name__)

def _build_dataloader(Z_arr, y_arr, w_arr=None, batch_size=32768, device='cuda'):
    """
    Helper function to safely load data into VRAM, 
    falling back to Asynchronous Pinned Memory if the dataset is too large.
    """
    in_vram = True
    if 'cuda' in str(device):
        try:
            z_t = torch.tensor(Z_arr, dtype=torch.float32, device=device)
            y_t = torch.tensor(y_arr, dtype=torch.float32, device=device)
            if w_arr is not None:
                w_t = torch.tensor(w_arr, dtype=torch.float32, device=device)
                dataset = TensorDataset(z_t, y_t, w_t)
            else:
                dataset = TensorDataset(z_t, y_t)
            
            # num_workers=0 is required if data is already in VRAM
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            return loader, True
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.info("CUDA OOM detected in Flow. Falling back to pinned System RAM.")
                torch.cuda.empty_cache()
                in_vram = False
            else:
                raise e
    else:
        in_vram = False
        
    if not in_vram:
        z_t = torch.tensor(Z_arr, dtype=torch.float32)
        y_t = torch.tensor(y_arr, dtype=torch.float32)
        if w_arr is not None:
            w_t = torch.tensor(w_arr, dtype=torch.float32)
            dataset = TensorDataset(z_t, y_t, w_t)
        else:
            dataset = TensorDataset(z_t, y_t)
            
        # pin_memory=True allows direct memory access (DMA) across the PCIe bus
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
        return loader, False

def train_flow_model(flow_model, Z_train, y_train, base_loc, base_scale, epochs=30, batch_size=32768, device='cuda', lr=1e-3):
    logger.info("\nTraining Conditional Normalizing Flow on Base Reference Data...")
    opt = torch.optim.Adam(flow_model.parameters(), lr=lr)
    
    y_scaled = (y_train - base_loc) / base_scale
    loader, in_vram = _build_dataloader(Z_train, y_scaled.reshape(-1, 1), batch_size=batch_size, device=device)
    
    flow_model.train()
    for ep in range(epochs): 
        for z_b, y_b in loader:
            if not in_vram:
                z_b = z_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)
                
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
        
        for param in self.flow_model.parameters():
            param.requires_grad = False
        self.flow_model.eval()
        
        self.calib_net = nn.Sequential(
            nn.Linear(n_context, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 2)
        ).to(device)
        
        self.calib_net[-1].weight.data.zero_()
        self.calib_net[-1].bias.data.zero_()
        
        self.optimizer = torch.optim.Adam(self.calib_net.parameters(), lr=1e-3)
        
    def __call__(self, Z_arr, y_arr, batch_size=32768):
        self.calib_net.eval()
        z_t = torch.tensor(Z_arr, dtype=torch.float32)
        y_flat = np.array(y_arr).flatten()
        y_t = torch.tensor(y_flat, dtype=torch.float32).unsqueeze(1)
        
        log_p = []
        with torch.no_grad():
            # Process in batches to prevent evaluation OOM
            for i in range(0, len(y_flat), batch_size):
                z_b = z_t[i:i+batch_size].to(self.device)
                y_b = y_t[i:i+batch_size].to(self.device)
                
                out = self.calib_net(z_b)
                delta_shift = out[:, 0:1]
                delta_log_scale = out[:, 1:2]
                
                total_shift = self.init_shift + delta_shift
                total_scale = self.init_scale * torch.exp(delta_log_scale)
                
                y_eq = self.base_loc + (y_b - (self.base_loc + total_shift)) / total_scale
                y_norm = (y_eq - self.base_loc) / self.base_scale
                
                log_prob_base = self.flow_model(z_b).log_prob(y_norm)
                log_prob_final = log_prob_base - torch.log(self.base_scale * total_scale).squeeze()
                
                log_p.append(log_prob_final)
                
        return torch.exp(torch.cat(log_p)).cpu().numpy().flatten()
        
    def update(self, Z_arr, y_arr, w_s_arr, epochs=5, batch_size=32768, penalty_strength=10.0):
        self.calib_net.train()
        
        y_flat = np.array(y_arr).reshape(-1, 1)
        w_flat = np.array(w_s_arr).reshape(-1, 1)
        loader, in_vram = _build_dataloader(Z_arr, y_flat, w_flat, batch_size=batch_size, device=self.device)
        
        for ep in range(epochs):
            for z_b, y_b, w_b in loader:
                if not in_vram:
                    z_b = z_b.to(self.device, non_blocking=True)
                    y_b = y_b.to(self.device, non_blocking=True)
                    w_b = w_b.to(self.device, non_blocking=True)
                    
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
                
                l2_nuisance_penalty = penalty_strength * torch.mean(delta_shift**2 + delta_log_scale**2)
                loss = -torch.mean(log_prob_final * w_b.squeeze()) + l2_nuisance_penalty
                
                loss.backward()
                self.optimizer.step() 
                
    def get_state(self):
        """Required by the EM loop to rollback cannibalization."""
        return {k: v.cpu().clone() for k, v in self.calib_net.state_dict().items()}
        
    def load_state(self, state_dict):
        """Required by the EM loop to restore optimal physics state."""
        self.calib_net.load_state_dict({k: v.to(self.device) for k, v in state_dict.items()})

def calibrate_flow_model(flow_model, Z_data, y_data, bkg_pdf_func, f0, base_loc, base_scale, peak_loc, epochs=50, batch_size=32768, device='cuda'):
    logger.info("\nExecuting Conditional Likelihood CNF Calibration...")
    
    y_flat = np.array(y_data).reshape(-1, 1)
    loader_calib, in_vram = _build_dataloader(Z_data, y_flat, batch_size=batch_size, device=device)
    f0_t = torch.tensor(f0, dtype=torch.float32, device=device)
    
    shift_param = nn.Parameter(torch.tensor(peak_loc - base_loc, dtype=torch.float32, device=device))
    log_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
    
    opt_calib = torch.optim.Adam([shift_param, log_scale], lr=0.005) 
    
    for ep in range(epochs): 
        for z_b, y_b in loader_calib:
            if not in_vram:
                z_b = z_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)
                
            opt_calib.zero_grad(set_to_none=True)
            scale = torch.exp(log_scale)
            shift = shift_param
            
            # Note: bkg_pdf_func is assumed to be an analytical NumPy function
            fb_cpu = bkg_pdf_func(y_b.cpu().numpy().flatten())
            fb = torch.tensor(fb_cpu, dtype=torch.float32, device=device).unsqueeze(1)
            
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
    
    n_context = Z_data.shape[1]
    return DynamicFlowWrapper(flow_model, n_context, base_loc, base_scale, shift_val, scale_val, device)

def build_signal_flow(Z_mc, y_mc, base_loc, base_scale, peak_loc, device='cuda'):
    """
    High-level wrapper to instantiate and train the Normalizing Flow on MC.
    Initializes the dynamic wrapper strictly at the 1D Anchor (no pre-calibration).
    """
    n_context = Z_mc.shape[1]
    
    flow_cnf = zuko.flows.NSF(features=1, context=n_context, hidden_features=[64, 64], bins=8).to(device)
    train_flow_model(flow_cnf, Z_mc, y_mc, base_loc, base_scale, device=device)
    
    init_shift = peak_loc - base_loc
    sig_pdf_Z = DynamicFlowWrapper(
        flow_model=flow_cnf, 
        n_context=n_context,
        base_loc=base_loc, 
        base_scale=base_scale, 
        init_shift=init_shift, 
        init_scale=1.0, 
        device=device
    )
    
    return sig_pdf_Z
