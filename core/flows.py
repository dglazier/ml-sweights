# core/flows.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss
import logging
import zuko

from .config import FlowConfig

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

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
        return loader, False


def _detect_tiled_period_1d(y_flat, max_period=512):
    """Detects if a 1D evaluation array is a tiled quadrature grid of period M."""
    n_tot = len(y_flat)
    if n_tot < 4:
        return None
    limit = min(n_tot // 2, max_period)
    matches = np.where(y_flat[1:limit + 1] == y_flat[0])[0]
    for idx in matches:
        m_grid = int(idx + 1)
        if m_grid >= 2 and (n_tot % m_grid == 0):
            if np.allclose(y_flat[:m_grid], y_flat[m_grid:2 * m_grid]):
                return m_grid
    return None


def _detect_tiled_period_2d(y1_flat, y2_flat, max_period=2048):
    """Detects if a 2D evaluation array is a tiled tensor-product quadrature grid of period M1*M2."""
    n_tot = len(y1_flat)
    if n_tot < 8:
        return None
    limit = min(n_tot // 2, max_period)
    matches = np.where((y1_flat[1:limit + 1] == y1_flat[0]) & (y2_flat[1:limit + 1] == y2_flat[0]))[0]
    for idx in matches:
        m_grid = int(idx + 1)
        if m_grid >= 4 and (n_tot % m_grid == 0):
            if (np.allclose(y1_flat[:m_grid], y1_flat[m_grid:2 * m_grid]) and
                    np.allclose(y2_flat[:m_grid], y2_flat[m_grid:2 * m_grid])):
                return m_grid
    return None


def _project_to_context(X_ctrl, n_ctx_dims):
    """
    Projects control features (including trig-expanded features and any appended
    remaining discriminants y_{j > k} in sequential unfolding) back to the flow context dimension.
    """
    X_arr = np.asarray(X_ctrl, dtype=np.float64)
    if X_arr.shape[1] == n_ctx_dims:
        return X_arr
    if X_arr.shape[1] >= 7 and n_ctx_dims >= 4:
        cT = X_arr[:, 0]
        cp, sp = X_arr[:, 2], X_arr[:, 3]
        cP, sP = X_arr[:, 4], X_arr[:, 5]
        z = X_arr[:, 6]
        phi, Phi = np.arctan2(sp, cp), np.arctan2(sP, cP)
        base_ctx = np.column_stack([cT, phi, Phi, z])
        if n_ctx_dims == 4:
            return base_ctx
        extra_cols = n_ctx_dims - 4
        return np.hstack([base_ctx, X_arr[:, 7:7 + extra_cols]])
    return X_arr[:, :n_ctx_dims]


def train_flow_model(flow_model, Z_train, y_train, base_loc, base_scale, epochs=30, batch_size=32768, device='cuda', lr=1e-3):
    logger.info("\nTraining Conditional Normalizing Flow on Base Reference Data...")
    opt = torch.optim.Adam(flow_model.parameters(), lr=lr)

    if y_train.ndim == 1:
        y_scaled = (y_train - base_loc) / base_scale
        y_scaled = y_scaled.reshape(-1, 1)
    else:
        y_scaled = (y_train - base_loc) / base_scale

    loader, in_vram = _build_dataloader(Z_train, y_scaled, batch_size=batch_size, device=device)

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
    """Wraps the pre-trained 1D CNF for event-by-event kinematic distortions."""
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
        y_flat = np.asarray(y_arr).flatten()
        y_t = torch.tensor(y_flat, dtype=torch.float32).unsqueeze(1)

        log_p = []
        with torch.no_grad():
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

        y_flat = np.asarray(y_arr).reshape(-1, 1)
        w_flat = np.asarray(w_s_arr).reshape(-1, 1)
        loader, in_vram = _build_dataloader(Z_arr, y_flat, w_flat, batch_size=batch_size, device=self.device)

        for ep in range(epochs):
            for z_b, y_b, w_b in loader:
                if not in_vram:
                    z_b = z_b.to(self.device, non_blocking=True)
                    y_b = y_b.to(device=self.device, non_blocking=True)
                    w_b = w_b.to(device=self.device, non_blocking=True)

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
        return {k: v.cpu().clone() for k, v in self.calib_net.state_dict().items()}

    def load_state(self, state_dict):
        self.calib_net.load_state_dict({k: v.to(self.device) for k, v in state_dict.items()})


def calibrate_flow_model(flow_model, Z_data, y_data, bkg_pdf_func, f0, base_loc, base_scale, peak_loc,
                         epochs=50, batch_size=32768, device='cuda', w_in=None):
    logger.info("\nExecuting Conditional Likelihood CNF Calibration...")
    y_flat = np.asarray(y_data).reshape(-1, 1)
    w_flat = np.ones_like(y_flat, dtype=np.float32) if w_in is None else np.asarray(w_in, dtype=np.float32).reshape(-1, 1)
    loader_calib, in_vram = _build_dataloader(Z_data, y_flat, w_flat, batch_size=batch_size, device=device)
    f0_t = torch.tensor(f0, dtype=torch.float32, device=device)

    shift_param = nn.Parameter(torch.tensor(peak_loc - base_loc, dtype=torch.float32, device=device))
    log_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))

    opt_calib = torch.optim.Adam([shift_param, log_scale], lr=0.005)

    for ep in range(epochs):
        for z_b, y_b, w_b in loader_calib:
            if not in_vram:
                z_b = z_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)
                w_b = w_b.to(device, non_blocking=True)

            opt_calib.zero_grad(set_to_none=True)
            scale = torch.exp(log_scale)
            shift = shift_param

            fb_cpu = bkg_pdf_func(y_b.cpu().numpy().flatten())
            fb = torch.tensor(fb_cpu, dtype=torch.float32, device=device).unsqueeze(1)

            y_eq = base_loc + (y_b - (base_loc + shift)) / scale
            y_norm = (y_eq - base_loc) / base_scale

            fs = torch.exp(flow_model(z_b).log_prob(y_norm)) / (base_scale * scale)
            loss = -torch.mean(w_b * torch.log(f0_t * fs + (1.0 - f0_t) * fb + 1e-10))
            loss.backward()
            opt_calib.step()

    shift_val = shift_param.item()
    scale_val = torch.exp(log_scale).item()

    logger.info(f"  --> Extracted Global Nuisance: Shift = {shift_val*1000:.1f} (Units), Smear = {(scale_val - 1.0)*100:.1f}%")
    n_context = Z_data.shape[1]
    return DynamicFlowWrapper(flow_model, n_context, base_loc, base_scale, shift_val, scale_val, device)


def build_signal_flow(Z_mc, y_mc, base_loc, base_scale, peak_loc, device='cuda'):
    n_context = Z_mc.shape[1]
    flow_cnf = zuko.flows.NSF(features=1, context=n_context, hidden_features=[64, 64], bins=8).to(device)
    train_flow_model(flow_cnf, Z_mc, y_mc, base_loc, base_scale, device=device)

    init_shift = peak_loc - base_loc
    sig_pdf_Z = DynamicFlowWrapper(
        flow_model=flow_cnf, n_context=n_context,
        base_loc=base_loc, base_scale=base_scale,
        init_shift=init_shift, init_scale=1.0, device=device
    )
    return sig_pdf_Z


# ==============================================================================
# 1D NORMALIZING FLOW SIGNAL MODEL (UNCONDITIONAL & CONDITIONAL)
# ==============================================================================
class MCFlow1D:
    """
    1D Normalizing Flow signal model using the `flows_2.py` architecture
    (`zuko.flows.NSF(features=1, context=n_context, hidden_features=[64, 64], bins=8)`
    trained via `train_flow_model` with dummy zero context in 1D unconditional mode so
    Zuko's MaskedMLP keeps all hidden layers unmasked), paired with two-way
    (`shift`, `scale`) domain adaptation and sideband-subtracted core moment matching.
    """
    # --- Inside class MCFlow1D ---
    def __init__(self, mc_y, y_min, y_max, mc_x=None, burn_in_iters=1, freeze_domain=False,
                 epochs=30, batch_size=32768, lr=1e-3, hidden_features=(64, 64), bins=8,
                 gh_nodes=9, init_shift=0.0, init_smear=0.0, init_scale=1.0, device='cuda'):
        mc_y = np.asarray(mc_y, dtype=np.float64).reshape(-1)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.device = device
        self.burn_in_iters = int(burn_in_iters)
        self.freeze_domain = bool(freeze_domain)

        self.mc_mu = float(np.mean(mc_y))
        self.mc_sig = float(max(np.std(mc_y), 1e-6))
        self.mc_var = float(np.var(mc_y))

        # Windowed reference variance on MC (standard 2.5*sigma window)
        win_mc = np.exp(-0.5 * ((mc_y - self.mc_mu) / max(2.5 * self.mc_sig, 1e-4)) ** 2)
        self.mc_var_win = float(np.sum(win_mc * (mc_y - self.mc_mu) ** 2) / np.maximum(np.sum(win_mc), 1e-12))

        self.init_shift = float(init_shift)
        self.init_smear = float(init_smear)
        self.init_scale = float(init_scale)
        self.shift = float(init_shift)
        self.scale = float(init_scale)
        self.smear = float(init_smear)
        self.iters = 0

        self.gh_nodes, self.gh_weights = hermgauss(gh_nodes)

        if mc_x is not None:
            self.is_conditional = True
            mc_x_arr = np.asarray(mc_x, dtype=np.float64)
            if mc_x_arr.ndim == 1:
                mc_x_arr = mc_x_arr.reshape(-1, 1)
            if mc_x_arr.shape[1] == 7:
                mc_x_arr = _project_to_context(mc_x_arr, 4)
            self.n_context = mc_x_arr.shape[1]
            train_Z_mc = mc_x_arr
        else:
            self.is_conditional = False
            self.n_context = 4
            train_Z_mc = np.zeros((len(mc_y), self.n_context), dtype=np.float64)

        self.flow = zuko.flows.NSF(
            features=1, context=self.n_context, hidden_features=list(hidden_features), bins=bins
        ).to(device)

        train_flow_model(
            self.flow, train_Z_mc, mc_y, self.mc_mu, self.mc_sig,
            epochs=epochs, batch_size=batch_size, device=device, lr=lr
        )
        for p in self.flow.parameters():
            p.requires_grad = False


    def reset_domain(self):
        self.shift = float(self.init_shift)
        self.scale = float(self.init_scale)
        self.smear = float(self.init_smear)
        self.iters = 0

    def _eval_uncond_1d(self, y_vec):
        eff_sig = self.mc_sig * self.scale
        u_vec = (y_vec - (self.mc_mu + self.shift)) / eff_sig
        dummy_z_t = torch.zeros((len(y_vec), self.n_context), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            dist = self.flow(dummy_z_t)
            u_t = torch.tensor(u_vec, dtype=torch.float32, device=self.device).unsqueeze(-1)
            val = torch.exp(dist.log_prob(u_t)).cpu().numpy() / eff_sig
            return np.clip(val, 1e-10, None)

    def _eval_cond_grid_1d(self, Z_unique, y_grid, batch_events=8192):
        n_events = len(Z_unique)
        m_grid = len(y_grid)
        out = np.empty((n_events, m_grid), dtype=np.float64)
        eff_sig = self.mc_sig * self.scale
        u_grid_1d = (y_grid - (self.mc_mu + self.shift)) / eff_sig

        with torch.no_grad():
            for start_ev in range(0, n_events, batch_events):
                end_ev = min(start_ev + batch_events, n_events)
                b_ev = end_ev - start_ev
                z_b_t = torch.tensor(Z_unique[start_ev:end_ev], dtype=torch.float32, device=self.device)
                dist = self.flow(z_b_t)
                u_grid = torch.tensor(
                    u_grid_1d, dtype=torch.float32, device=self.device
                ).view(m_grid, 1, 1).expand(m_grid, b_ev, 1)
                p_b = (torch.exp(dist.log_prob(u_grid)) / eff_sig).T.cpu().numpy()
                out[start_ev:end_ev] = np.clip(p_b, 1e-10, None)

        return out.reshape(-1)

    def _eval_cond_pointwise_1d(self, Z_ctx, y_flat, batch_size=32768):
        n_tot = len(y_flat)
        out = np.empty(n_tot, dtype=np.float64)
        eff_sig = self.mc_sig * self.scale
        u_all = (y_flat - (self.mc_mu + self.shift)) / eff_sig

        with torch.no_grad():
            for start_idx in range(0, n_tot, batch_size):
                end_idx = min(start_idx + batch_size, n_tot)
                z_b_t = torch.tensor(Z_ctx[start_idx:end_idx], dtype=torch.float32, device=self.device)
                u_b = u_all[start_idx:end_idx]
                dist = self.flow(z_b_t)
                u_t = torch.tensor(u_b, dtype=torch.float32, device=self.device).unsqueeze(-1)
                p_b = (torch.exp(dist.log_prob(u_t)) / eff_sig).cpu().numpy()
                out[start_idx:end_idx] = np.clip(p_b, 1e-10, None)

        return out

    def __call__(self, x, y):
        y_flat = np.asarray(y, dtype=np.float64).reshape(-1)
        m_grid = _detect_tiled_period_1d(y_flat)

        if not self.is_conditional:
            if m_grid is not None:
                return np.tile(self._eval_uncond_1d(y_flat[:m_grid]), len(y_flat) // m_grid)
            return self._eval_uncond_1d(y_flat)

        Z_ctx = _project_to_context(x, self.n_context)
        if m_grid is not None and len(Z_ctx) == len(y_flat):
            if np.allclose(Z_ctx[0], Z_ctx[m_grid - 1]):
                return self._eval_cond_grid_1d(Z_ctx[::m_grid], y_flat[:m_grid])

        return self._eval_cond_pointwise_1d(Z_ctx, y_flat)

    def update(self, x, y, w_eff):
        if self.freeze_domain:
            return
        self.iters += 1
        if self.iters >= self.burn_in_iters:
            y_flat = np.asarray(y, dtype=np.float64).reshape(-1)
            w_flat = np.asarray(w_eff, dtype=np.float64).reshape(-1)

            tune_steps = self.iters - self.burn_in_iters
            anneal = 1.0 if tune_steps == 0 else max(0.05, 0.5 * (0.85 ** tune_steps))

            peak_loc = self.mc_mu + self.shift
            # Scale-covariant window width: scales with self.scale so d_var / mc_var_win == scale^2
            win_width = max(2.5 * self.scale * self.mc_sig, 1e-4)

            for _ in range(3):
                window = np.exp(-0.5 * ((y_flat - peak_loc) / win_width) ** 2)
                w_win = w_flat * window
                w_sum = float(np.sum(w_win))
                if w_sum > 1e-10:
                    peak_loc = float(np.sum(w_win * y_flat) / w_sum)

            if w_sum > 1e-10:
                d_mu = peak_loc
                target_shift = d_mu - self.mc_mu
                self.shift = float(self.shift + anneal * (target_shift - self.shift))

                # Update scale only after Iteration 1 (tune_steps >= 1) once w_s has been
                # evaluated with the actual Flow template rather than the crude Gaussian anchor
                if tune_steps >= 1:
                    d_var = float(np.sum(w_win * (y_flat - d_mu) ** 2) / w_sum)
                    if d_var > 1e-10:
                        target_scale = float(np.clip(np.sqrt(d_var / max(self.mc_var_win, 1e-12)), 0.50, 2.00))
                        self.scale = float(np.clip(self.scale + 0.5 * anneal * (target_scale - self.scale), 0.50, 2.00))
                        self.smear = float(self.mc_sig * np.sqrt(max(0.0, self.scale ** 2 - 1.0)))

                mode_tag = "CondFlow1D" if self.is_conditional else "Flow1D"
                logger.info(
                    f"  [Domain Tune] {mode_tag} updated -> Shift: {self.shift:+.5f} | "
                    f"Scale: {self.scale:.4f} (Smear eq: {self.smear:.5f})"
                )


    def get_state(self):
        return {'shift': self.shift, 'scale': self.scale, 'smear': self.smear, 'iters': self.iters}

    def load_state(self, s):
        if s is not None:
            self.shift = float(s['shift'])
            self.scale = float(s.get('scale', 1.0))
            self.smear = float(s.get('smear', 0.0))
            self.iters = int(s['iters'])


# ==============================================================================
# N-DIMENSIONAL NORMALIZING FLOW SIGNAL MODEL (UNCONDITIONAL & CONDITIONAL)
# ==============================================================================
class MCFlowND:
    """
    N-Dimensional Normalizing Flow signal model (supporting K >= 1 dimensions and both
    Unconditional and Conditional modes) using non-zero context dimension so Zuko's
    MaskedMLP keeps all hidden layers active for every feature, paired with per-dimension
    (`shift`, `scale`) nuisance parameters.
    """
    # --- Inside class MCFlowND ---
    def __init__(self, mc_y_nd, mc_x=None, burn_in_iters=1, freeze_domain=False,
                 epochs=30, batch_size=32768, lr=1e-3, hidden_features=(64, 64), bins=8,
                 gh_nodes=5, device='cuda'):
        mc_y_nd = np.column_stack([mc_y_nd]) if np.asarray(mc_y_nd).ndim == 1 else np.asarray(mc_y_nd, dtype=np.float64)
        self.K = mc_y_nd.shape[1]
        self.device = device
        self.burn_in_iters = int(burn_in_iters)
        self.freeze_domain = bool(freeze_domain)

        self.mc_mu = np.mean(mc_y_nd, axis=0)
        self.mc_sig = np.maximum(np.std(mc_y_nd, axis=0), 1e-6)
        self.jac_inv = float(np.prod(1.0 / self.mc_sig))

        z_mc_win = (mc_y_nd - self.mc_mu) / (2.5 * self.mc_sig)
        win_mc = np.exp(-0.5 * np.sum(z_mc_win ** 2, axis=1))
        win_mc_sum = max(float(np.sum(win_mc)), 1e-12)
        self.mc_var_win = np.sum(win_mc[:, None] * ((mc_y_nd - self.mc_mu) ** 2), axis=0) / win_mc_sum

        self.shift = np.zeros(self.K, dtype=np.float64)
        self.scale = np.ones(self.K, dtype=np.float64)
        self.smear = np.zeros(self.K, dtype=np.float64)
        self.iters = 0

        gh_x, gh_w = hermgauss(gh_nodes)
        self.gh_nodes = gh_x
        self.gh_weights = gh_w / np.sqrt(np.pi)

        if mc_x is not None:
            self.is_conditional = True
            mc_x_arr = np.asarray(mc_x, dtype=np.float64)
            if mc_x_arr.ndim == 1:
                mc_x_arr = mc_x_arr.reshape(-1, 1)
            if mc_x_arr.shape[1] == 7:
                mc_x_arr = _project_to_context(mc_x_arr, 4)
            self.n_context = mc_x_arr.shape[1]
            train_Z_mc = mc_x_arr
        else:
            self.is_conditional = False
            self.n_context = 4
            train_Z_mc = np.zeros((len(mc_y_nd), self.n_context), dtype=np.float64)

        self.flow = zuko.flows.NSF(
            features=self.K, context=self.n_context, hidden_features=list(hidden_features), bins=bins
        ).to(device)

        train_flow_model(
            self.flow, train_Z_mc, mc_y_nd, self.mc_mu, self.mc_sig,
            epochs=epochs, batch_size=batch_size, device=device, lr=lr
        )
        for p in self.flow.parameters():
            p.requires_grad = False



    def reset_domain(self):
        self.shift = np.zeros(self.K, dtype=np.float64)
        self.scale = np.ones(self.K, dtype=np.float64)
        self.smear = np.zeros(self.K, dtype=np.float64)
        self.iters = 0

    @staticmethod
    def _pack_y(y_args):
        if len(y_args) == 1:
            arr = np.asarray(y_args[0], dtype=np.float64)
            return arr.reshape(-1, 1) if arr.ndim == 1 else arr
        return np.column_stack([np.asarray(a, dtype=np.float64).reshape(-1) for a in y_args])

    def _eval_uncond_nd(self, y_nd):
        eff_sig = self.mc_sig * self.scale
        eff_jac_inv = float(np.prod(1.0 / eff_sig))
        u_nd = (y_nd - (self.mc_mu[None, :] + self.shift[None, :])) / eff_sig[None, :]
        dummy_z_t = torch.zeros((len(y_nd), self.n_context), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            dist = self.flow(dummy_z_t)
            u_t = torch.tensor(u_nd, dtype=torch.float32, device=self.device)
            val = torch.exp(dist.log_prob(u_t)).cpu().numpy() * eff_jac_inv
            return np.clip(val, 1e-10, None)

    def _eval_cond_grid_nd(self, Z_unique, y_grid_nd, batch_events=4096):
        n_events = len(Z_unique)
        m_grid = len(y_grid_nd)
        out = np.empty((n_events, m_grid), dtype=np.float64)
        eff_sig = self.mc_sig * self.scale
        eff_jac_inv = float(np.prod(1.0 / eff_sig))
        u_grid_nd = (y_grid_nd - (self.mc_mu[None, :] + self.shift[None, :])) / eff_sig[None, :]

        with torch.no_grad():
            for start_ev in range(0, n_events, batch_events):
                end_ev = min(start_ev + batch_events, n_events)
                b_ev = end_ev - start_ev
                z_b_t = torch.tensor(Z_unique[start_ev:end_ev], dtype=torch.float32, device=self.device)
                dist = self.flow(z_b_t)
                u_grid_t = torch.tensor(u_grid_nd, dtype=torch.float32, device=self.device).unsqueeze(1).expand(m_grid, b_ev, self.K)
                p_b = (torch.exp(dist.log_prob(u_grid_t)) * eff_jac_inv).T.cpu().numpy()
                out[start_ev:end_ev] = np.clip(p_b, 1e-10, None)

        return out.reshape(-1)

    def _eval_cond_pointwise_nd(self, Z_ctx, y_nd, batch_size=32768):
        n_tot = len(y_nd)
        out = np.empty(n_tot, dtype=np.float64)
        eff_sig = self.mc_sig * self.scale
        eff_jac_inv = float(np.prod(1.0 / eff_sig))
        u_all = (y_nd - (self.mc_mu[None, :] + self.shift[None, :])) / eff_sig[None, :]

        with torch.no_grad():
            for start_idx in range(0, n_tot, batch_size):
                end_idx = min(start_idx + batch_size, n_tot)
                z_b_t = torch.tensor(Z_ctx[start_idx:end_idx], dtype=torch.float32, device=self.device)
                u_b = u_all[start_idx:end_idx]
                dist = self.flow(z_b_t)
                u_t = torch.tensor(u_b, dtype=torch.float32, device=self.device)
                p_b = (torch.exp(dist.log_prob(u_t)) * eff_jac_inv).cpu().numpy()
                out[start_idx:end_idx] = np.clip(p_b, 1e-10, None)

        return out

    def __call__(self, x, *y_args):
        y_nd = self._pack_y(y_args)
        m_grid = None
        if self.K == 1:
            m_grid = _detect_tiled_period_1d(y_nd[:, 0])
        elif self.K == 2:
            m_grid = _detect_tiled_period_2d(y_nd[:, 0], y_nd[:, 1])

        if not self.is_conditional:
            if m_grid is not None:
                return np.tile(self._eval_uncond_nd(y_nd[:m_grid]), len(y_nd) // m_grid)
            return self._eval_uncond_nd(y_nd)

        Z_ctx = _project_to_context(x, self.n_context)
        if m_grid is not None and len(Z_ctx) == len(y_nd):
            if np.allclose(Z_ctx[0], Z_ctx[m_grid - 1]):
                return self._eval_cond_grid_nd(Z_ctx[::m_grid], y_nd[:m_grid])

        return self._eval_cond_pointwise_nd(Z_ctx, y_nd)

    # --- Inside class MCFlowND ---
    def __init__(self, mc_y_nd, mc_x=None, burn_in_iters=1, freeze_domain=False,
                 epochs=30, batch_size=32768, lr=1e-3, hidden_features=(64, 64), bins=8,
                 gh_nodes=5, device='cuda'):
        mc_y_nd = np.column_stack([mc_y_nd]) if np.asarray(mc_y_nd).ndim == 1 else np.asarray(mc_y_nd, dtype=np.float64)
        self.K = mc_y_nd.shape[1]
        self.device = device
        self.burn_in_iters = int(burn_in_iters)
        self.freeze_domain = bool(freeze_domain)

        self.mc_mu = np.mean(mc_y_nd, axis=0)
        self.mc_sig = np.maximum(np.std(mc_y_nd, axis=0), 1e-6)
        self.jac_inv = float(np.prod(1.0 / self.mc_sig))

        z_mc_win = (mc_y_nd - self.mc_mu) / (2.5 * self.mc_sig)
        win_mc = np.exp(-0.5 * np.sum(z_mc_win ** 2, axis=1))
        win_mc_sum = max(float(np.sum(win_mc)), 1e-12)
        self.mc_var_win = np.sum(win_mc[:, None] * ((mc_y_nd - self.mc_mu) ** 2), axis=0) / win_mc_sum

        self.shift = np.zeros(self.K, dtype=np.float64)
        self.scale = np.ones(self.K, dtype=np.float64)
        self.smear = np.zeros(self.K, dtype=np.float64)
        self.iters = 0

        gh_x, gh_w = hermgauss(gh_nodes)
        self.gh_nodes = gh_x
        self.gh_weights = gh_w / np.sqrt(np.pi)

        if mc_x is not None:
            self.is_conditional = True
            mc_x_arr = np.asarray(mc_x, dtype=np.float64)
            if mc_x_arr.ndim == 1:
                mc_x_arr = mc_x_arr.reshape(-1, 1)
            if mc_x_arr.shape[1] == 7:
                mc_x_arr = _project_to_context(mc_x_arr, 4)
            self.n_context = mc_x_arr.shape[1]
            train_Z_mc = mc_x_arr
        else:
            self.is_conditional = False
            self.n_context = 4
            train_Z_mc = np.zeros((len(mc_y_nd), self.n_context), dtype=np.float64)

        self.flow = zuko.flows.NSF(
            features=self.K, context=self.n_context, hidden_features=list(hidden_features), bins=bins
        ).to(device)

        train_flow_model(
            self.flow, train_Z_mc, mc_y_nd, self.mc_mu, self.mc_sig,
            epochs=epochs, batch_size=batch_size, device=device, lr=lr
        )
        for p in self.flow.parameters():
            p.requires_grad = False

    def update(self, x, *args):
        if self.freeze_domain:
            return
        w_eff = np.asarray(args[-1], dtype=np.float64).reshape(-1)
        y_nd = self._pack_y(args[:-1])
        self.iters += 1
        if self.iters >= self.burn_in_iters:
            tune_steps = self.iters - self.burn_in_iters
            anneal = 1.0 if tune_steps == 0 else max(0.05, 0.5 * (0.85 ** tune_steps))

            peak_loc = self.mc_mu + self.shift
            win_width = np.maximum(2.5 * self.scale * self.mc_sig, 1e-4)

            for _ in range(3):
                z_win = (y_nd - peak_loc[None, :]) / win_width[None, :]
                window = np.exp(-0.5 * np.sum(z_win ** 2, axis=1))
                w_win = w_eff * window
                w_sum = float(np.sum(w_win))
                if w_sum > 1e-10:
                    peak_loc = np.sum(w_win[:, None] * y_nd, axis=0) / w_sum

            if w_sum > 1e-10:
                d_mu = peak_loc
                target_shift = d_mu - self.mc_mu
                self.shift = self.shift + anneal * (target_shift - self.shift)

                if tune_steps >= 1:
                    d_var = np.sum(w_win[:, None] * ((y_nd - d_mu[None, :]) ** 2), axis=0) / w_sum
                    valid_v = d_var > 1e-10
                    if np.all(valid_v):
                        target_scale = np.clip(np.sqrt(d_var / np.maximum(self.mc_var_win, 1e-12)), 0.50, 2.00)
                        self.scale = np.clip(self.scale + 0.5 * anneal * (target_scale - self.scale), 0.50, 2.00)
                        self.smear = self.mc_sig * np.sqrt(np.maximum(0.0, self.scale ** 2 - 1.0))

                mode_tag = f"CondFlow{self.K}D" if self.is_conditional else f"Flow{self.K}D"
                sh_str = ", ".join([f"{s:+.5f}" for s in self.shift])
                sc_str = ", ".join([f"{s:.4f}" for s in self.scale])
                logger.info(f"  [Domain Tune] {mode_tag} updated -> Shift: [{sh_str}] | Scale: [{sc_str}]")

    def get_state(self):
        return {
            'shift': self.shift.copy(),
            'scale': self.scale.copy(),
            'smear': self.smear.copy(),
            'iters': self.iters
        }

    def load_state(self, s):
        if s is not None:
            self.shift = np.asarray(s['shift'], dtype=np.float64).copy()
            if 'scale' in s:
                self.scale = np.asarray(s['scale'], dtype=np.float64).copy()
            self.smear = np.asarray(s.get('smear', np.zeros(self.K)), dtype=np.float64).copy()
            self.iters = int(s['iters'])


MCFlow2D = MCFlowND


# ==============================================================================
# BUILDER FUNCTIONS
# ==============================================================================
def build_domain_adapted_signal_flow(
    Z_ctx_mc, y_mc, base_loc=None, base_scale=None, peak_loc=None, config: FlowConfig = None, device='cuda'
):
    """Unified 1D builder for Conditional CNF and Angle-Blind Unconditional Flow."""
    cfg = config or FlowConfig()
    mc_x = Z_ctx_mc if getattr(cfg, 'flow_mode', 'unconditional') == "conditional" else None
    init_sh = float(peak_loc - base_loc) if (peak_loc is not None and base_loc is not None) else float(getattr(cfg, 'init_shift', 0.0))
    return MCFlow1D(
        mc_y=y_mc,
        y_min=float(np.min(y_mc)),
        y_max=float(np.max(y_mc)),
        mc_x=mc_x,
        burn_in_iters=max(1, int(getattr(cfg, 'burn_in_iters', 1))),
        gh_nodes=int(getattr(cfg, 'gh_nodes', 9)),
        init_shift=init_sh,
        init_smear=float(getattr(cfg, 'init_smear', 0.0)),
        init_scale=float(getattr(cfg, 'init_scale', 1.0)),
        device=device
    )


def build_2d_domain_adapted_signal_flow(
    y_mc_2d, base_loc_2d=None, base_scale_2d=None, peak_loc_2d=None,
    Z_ctx_mc=None, config: FlowConfig = None, device='cuda'
):
    """Unified 2D/ND builder for Unconditional or Conditional 2D Normalizing Flows."""
    cfg = config or FlowConfig()
    mc_x = Z_ctx_mc if getattr(cfg, 'flow_mode', 'unconditional') == "conditional" else None
    flow_2d = MCFlowND(
        mc_y_nd=y_mc_2d,
        mc_x=mc_x,
        burn_in_iters=max(1, int(getattr(cfg, 'burn_in_iters', 1))),
        gh_nodes=int(getattr(cfg, 'gh_nodes', 5)),
        device=device
    )
    if peak_loc_2d is not None and base_loc_2d is not None:
        flow_2d.shift = np.asarray(peak_loc_2d, dtype=np.float64) - np.asarray(base_loc_2d, dtype=np.float64)
    return flow_2d
