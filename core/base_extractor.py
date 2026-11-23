# core/base_extractor.py
import numpy as np
import torch
import logging
from sklearn.model_selection import KFold

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from .networks import PureDNN, train_dnn, predict_in_batches
from .quadrature import get_gauss_legendre_grid
from .extractor import SignedXGBClassifier

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class BaseMLSWeightExtractor:
    """
    Pure mathematical implementation of Density-Ratio EM Background Subtraction.
    - Cold Start DNN incremental training (warm_start=False by default).
    - Strict Local COWs Denominator I(y|x) clamped for matrix stability.
    - 2D Row-Wise Whittaker-Eilers Log-Spline Smoothing (Mandatory Sideband Interpolator).
    - Unbounded Dynamic Alpha steering.
    - Optional Yield Locking (lock_yield) to allow organic equilibrium drift.
    """
    def __init__(self, disc_bounds, n_nodes=64, device='cuda', gbdt_mode='XGB_GPU', 
                 dnn_epochs=50, marginal_alpha=1.0, max_iters=30, tol=1e-4, 
                 omega_init=2.0, omega_decay_iters=25.0,
                 ry_smoothing=5.0, peak_curv_penalty=50.0,
                 warm_start=True, lock_yield=True, extended_cows_derivatives=1, n_folds=1, **kwargs):
        
        self.y_min, self.y_max = disc_bounds
        self.n_nodes = n_nodes
        self.device = device
        self.gbdt_mode = gbdt_mode
        self.dnn_epochs = dnn_epochs          
        self.marginal_alpha = marginal_alpha 
        self.max_iters = max_iters
        self.tol = tol 
        
        self.omega_init = omega_init
        self.omega_decay_iters = omega_decay_iters
        
        self.ry_smoothing = ry_smoothing
        self.peak_curv_penalty = peak_curv_penalty
        
        self.warm_start = warm_start
        self.lock_yield = lock_yield
        self.extended_cows_derivatives = extended_cows_derivatives
        self.n_folds = n_folds
        
        self.y_grid_cpu = None
        self.model_A = None
        self.gbdt_A = None
        self.model_B = None
        self.gbdt_B = None
        self.w_s = None
        self.history = {}

    def _build_second_diff_matrix(self, y_nodes):
        M = len(y_nodes)
        span = float(y_nodes[-1] - y_nodes[0])
        u = (y_nodes - y_nodes[0]) / max(span, 1e-12)
        D2 = np.zeros((M - 2, M), dtype=np.float64)
        for i in range(1, M - 1):
            h_prev = max(u[i] - u[i - 1], 1e-8)
            h_next = max(u[i + 1] - u[i], 1e-8)
            h_sum = h_prev + h_next
            scale = ((1.0 / max(M - 1, 1)) ** 2) * 2.0 / (h_prev * h_next * h_sum)
            D2[i - 1, i - 1] = scale * h_next
            D2[i - 1, i] = -scale * h_sum
            D2[i - 1, i + 1] = scale * h_prev
        return D2

    def _build_row_smoother_matrix(self, fs_1d_grid_cpu):
        M = len(self.y_grid_cpu)
        if M < 4 or self.ry_smoothing <= 0.0:
            return torch.eye(M, dtype=torch.float32, device=self.device)

        fs_max = max(float(np.max(fs_1d_grid_cpu)), 1e-12)
        w_sig = fs_1d_grid_cpu / fs_max

        D2 = self._build_second_diff_matrix(self.y_grid_cpu)
        w_interior = w_sig[1:-1]
        lam_vec = (self.ry_smoothing ** 2) * (1.0 + self.peak_curv_penalty * w_interior)
        p_diag = 1.0 - 0.80 * w_sig
        
        A_sys = np.diag(p_diag) + D2.T @ (lam_vec[:, None] * D2)
        S_cpu = np.linalg.solve(A_sys, np.diag(p_diag))
        return torch.tensor(S_cpu, dtype=torch.float32, device=self.device)

    def _apply_row_smoothing(self, R_raw_grid_t, y_obs_t, y_grid_t, S_t):
        B, M = R_raw_grid_t.shape
        idx = torch.clamp(torch.searchsorted(y_grid_t, y_obs_t.contiguous()), 1, M - 1)
        y0, y1 = y_grid_t[idx - 1], y_grid_t[idx]
        w_interp = (y_obs_t - y0) / torch.clamp(y1 - y0, min=1e-12)
        row_idx = torch.arange(B, device=self.device)

        if self.ry_smoothing <= 0.0:
            v0, v1 = R_raw_grid_t[row_idx, idx - 1], R_raw_grid_t[row_idx, idx]
            R_obs_t = v0 + w_interp * (v1 - v0)
            return R_raw_grid_t, R_obs_t
            
        log_R_raw_t = torch.log(torch.clamp(R_raw_grid_t, min=1e-6, max=1e6))
        log_R_sm_grid_t = torch.clamp(torch.matmul(log_R_raw_t, S_t.T), min=-15.0, max=15.0)
        
        v0, v1 = log_R_sm_grid_t[row_idx, idx - 1], log_R_sm_grid_t[row_idx, idx]
        log_R_sm_obs_t = v0 + w_interp * (v1 - v0)
        
        return torch.exp(log_R_sm_grid_t), torch.exp(log_R_sm_obs_t)

    def _compute_algebraic_marginal_correction(self, R_grid_t, fb_base_grid_t, dy_weights_t, w_in_t, w_b_t):
        I_base_t = torch.clamp(torch.sum(fb_base_grid_t * dy_weights_t), min=1e-12)
        Z_b_row_t = torch.clamp(torch.sum(R_grid_t * fb_base_grid_t * dy_weights_t, dim=1, keepdim=True), min=1e-12)
        R_norm_grid_t = R_grid_t * (I_base_t / Z_b_row_t)
        
        w_eff_b = w_in_t * w_b_t
        R_y_margin_t = torch.sum(w_eff_b.unsqueeze(1) * R_norm_grid_t, dim=0) / torch.clamp(torch.sum(w_eff_b), min=1e-12)
        R_y_margin_safe = torch.clamp(R_y_margin_t, min=0.01)
        
        f_margin_grid_t = fb_base_grid_t * R_y_margin_safe
        I_margin_t = torch.sum(f_margin_grid_t * dy_weights_t)
        
        raw_corr_t = (I_margin_t / I_base_t) / R_y_margin_safe
        corr_factor_t = torch.clamp(raw_corr_t, min=1e-6, max=1e6)
        return torch.pow(corr_factor_t, float(self.marginal_alpha))

    def _compute_cows_weights_pytorch(self, Z_ctrl, y_obs, fs_grid_t, fb_grid_t, fs_obs_t, fb_obs_t, 
                                      I_grid_t, I_obs_t, dy_weights_t, y_grid_cpu, f_sig_cond, 
                                      active_derivatives, batch_size=8192):
        N_tot = len(y_obs)
        n_nodes = len(y_grid_cpu)
        sW_out = torch.zeros(N_tot, dtype=torch.float32, device=self.device)
        dy_64_full = dy_weights_t.double().unsqueeze(0) if dy_weights_t.ndim == 1 else dy_weights_t.double()
        
        for start_idx in range(0, N_tot, batch_size):
            end_idx = min(start_idx + batch_size, N_tot)
            b_len = end_idx - start_idx
            
            Z_b = Z_ctrl[start_idx:end_idx]
            y_obs_b = y_obs[start_idx:end_idx]
            
            fs_g_b = fs_grid_t[start_idx:end_idx]
            fb_g_b = fb_grid_t[start_idx:end_idx]
            fs_o_b = fs_obs_t[start_idx:end_idx]
            fb_o_b = fb_obs_t[start_idx:end_idx]
            I_g_b = I_grid_t[start_idx:end_idx]
            I_o_b = I_obs_t[start_idx:end_idx]
            dy_64 = dy_64_full if dy_64_full.shape[0] == 1 else dy_64_full[start_idx:end_idx]
            
            if active_derivatives <= 0:
                fs_g_64, fb_g_64, I_g_64 = fs_g_b.double(), fb_g_b.double(), I_g_b.double()
                W_ss = torch.sum((fs_g_64**2 / I_g_64) * dy_64, dim=1)
                W_bb = torch.sum((fb_g_64**2 / I_g_64) * dy_64, dim=1)
                W_sb = torch.sum((fs_g_64 * fb_g_64 / I_g_64) * dy_64, dim=1)
                
                Det = torch.clamp(W_ss * W_bb - W_sb**2, min=1e-14)
                A_ss = (W_bb / Det).float()
                A_sb = (-W_sb / Det).float()
                sW_out[start_idx:end_idx] = (A_ss * fs_o_b + A_sb * fb_o_b) / I_o_b
                continue

            eps = 1e-4
            Z_ctrl_rep = np.repeat(Z_b, n_nodes, axis=0)
            y_tiled = np.tile(y_grid_cpu, b_len)
            
            fs_plus_raw = torch.tensor(f_sig_cond(Z_ctrl_rep, y_tiled + eps), dtype=torch.float32, device=self.device).view(b_len, n_nodes)
            fs_minus_raw = torch.tensor(f_sig_cond(Z_ctrl_rep, y_tiled - eps), dtype=torch.float32, device=self.device).view(b_len, n_nodes)
            fs_raw = torch.tensor(f_sig_cond(Z_ctrl_rep, y_tiled), dtype=torch.float32, device=self.device).view(b_len, n_nodes)
            
            dy_32 = dy_weights_t.unsqueeze(0) if dy_weights_t.ndim == 1 else dy_weights_t[start_idx:end_idx]
            Z_s = torch.clamp(torch.sum(fs_raw * dy_32, dim=1, keepdim=True), min=1e-12)
            
            dfs1_raw = (fs_plus_raw - fs_minus_raw) / (2 * eps * Z_s)
            fs_plus_obs = torch.tensor(f_sig_cond(Z_b, y_obs_b + eps), dtype=torch.float32, device=self.device)
            fs_minus_obs = torch.tensor(f_sig_cond(Z_b, y_obs_b - eps), dtype=torch.float32, device=self.device)
            dfs1_obs_raw = (fs_plus_obs - fs_minus_obs) / (2 * eps * Z_s.squeeze(1))
            
            int_dfs1 = torch.sum(dfs1_raw * dy_32, dim=1, keepdim=True)
            dfs1_grid = dfs1_raw - fs_g_b * int_dfs1
            rms_dfs1 = torch.sqrt(torch.clamp(torch.sum((dfs1_grid**2) * dy_32, dim=1, keepdim=True), min=1e-12))
            dfs1_grid = dfs1_grid / rms_dfs1
            dfs1_obs = (dfs1_obs_raw - fs_o_b * int_dfs1.squeeze(1)) / rms_dfs1.squeeze(1)
            
            basis_grid = torch.stack([fs_g_b, fb_g_b, dfs1_grid], dim=1) 
            basis_obs = torch.stack([fs_o_b, fb_o_b, dfs1_obs], dim=1)     
            
            dy_div_I = dy_32 / I_g_b
            w_basis = basis_grid * dy_div_I.unsqueeze(1)
            W_mat = torch.bmm(basis_grid, w_basis.transpose(1, 2))
            
            diag_w = torch.diagonal(W_mat, dim1=1, dim2=2)
            reg = torch.zeros_like(diag_w)
            reg[:, :] = 1e-8 * torch.clamp(diag_w, min=1e-12)
            W_mat = W_mat + torch.diag_embed(reg)
                
            A_mat = torch.linalg.inv(W_mat)
            sW_out[start_idx:end_idx] = torch.sum(A_mat[:, 0, :] * basis_obs, dim=1) / I_o_b
            
        return sW_out
    
    def _eval_sig_grid_batched(self, f_sig_cond, Z_ctrl, y_grid_cpu, batch_size=8192):
        N_tot = len(Z_ctrl)
        n_nodes = len(y_grid_cpu)
        out = torch.zeros((N_tot, n_nodes), dtype=torch.float32, device=self.device)
        for start_idx in range(0, N_tot, batch_size):
            end_idx = min(start_idx + batch_size, N_tot)
            Z_b = Z_ctrl[start_idx:end_idx]
            Z_b_rep = np.repeat(Z_b, n_nodes, axis=0)
            y_tiled = np.tile(y_grid_cpu, len(Z_b))
            res = f_sig_cond(Z_b_rep, y_tiled)
            out[start_idx:end_idx] = torch.tensor(res, dtype=torch.float32, device=self.device).view(-1, n_nodes)
        return out

    def _torch_interp1d_local(self, x, xp, yp):
        x = x.contiguous()
        idx = torch.clamp(torch.searchsorted(xp, x), 1, len(xp) - 1)
        x0, x1 = xp[idx - 1], xp[idx]
        y0, y1 = yp[idx - 1], yp[idx]
        w_interp = (x - x0) / torch.clamp(x1 - x0, min=1e-12)
        return y0 + w_interp * (y1 - y0)
    
    def fit_transform(self, X_ctrl, y_disc, w_s_init, f_sig_cond, f_bkg_1d, bkg_sampler, w_in=None, validation_data=None):
        N_tot, n_ctrl_dims = X_ctrl.shape
        w_in_cpu = np.ones(N_tot, dtype=np.float64) if w_in is None else np.asarray(w_in, dtype=np.float64)
        w_in_t = torch.tensor(w_in_cpu, dtype=torch.float32, device=self.device)
        
        Z_ctrl_t = torch.tensor(X_ctrl, dtype=torch.float32, device=self.device)
        y_disc_t = torch.tensor(y_disc, dtype=torch.float32, device=self.device)
        w_s_t = torch.tensor(w_s_init, dtype=torch.float32, device=self.device)
        
        y_grid_t, dy_weights_t = get_gauss_legendre_grid(self.y_min, self.y_max, n_nodes=self.n_nodes, device=self.device)
        self.y_grid_cpu = y_grid_t.cpu().numpy()
        
        fb_base_grid_t = torch.tensor(f_bkg_1d(self.y_grid_cpu), dtype=torch.float32, device=self.device)
        fb_global_t = torch.tensor(f_bkg_1d(y_disc), dtype=torch.float32, device=self.device)
        
        fs_local_raw_t = torch.tensor(f_sig_cond(X_ctrl, y_disc), dtype=torch.float32, device=self.device)
        fs_grid_raw_t = self._eval_sig_grid_batched(f_sig_cond, X_ctrl, self.y_grid_cpu)
        
        Z_s_ev_t = torch.sum(fs_grid_raw_t * dy_weights_t, dim=1)
        fs_local_t = fs_local_raw_t / torch.clamp(Z_s_ev_t, min=1e-12)
        fs_cond_grid_t = fs_grid_raw_t / torch.clamp(Z_s_ev_t.unsqueeze(1), min=1e-12)
        
        start_msg = "Warm Start" if self.warm_start else "Cold Start"
        lock_msg = "Yield Locked" if self.lock_yield else "Yield Free (Organic Float)"
        logger.info(f"Launching Pure EM Loop on {self.device} [{start_msg} | {lock_msg} | Row-Wise Smoothing]...")
        
        l1_history = []
        self.history = {'yield': [], 'l1_shift': [], 'chi2s': [], 'mean_chi2': []}
        
        for it in range(1, self.max_iters + 1):
            w_b_t = 1.0 - w_s_t
            w_b_cpu = w_b_t.cpu().numpy()
            w_s_cpu = w_s_t.cpu().numpy()
            
            decay_factor = max(0.0, (self.omega_decay_iters - it) / max(1.0, self.omega_decay_iters - 1.0))
            omega_curr = 1.0 + (self.omega_init - 1.0) * decay_factor

            if not self.warm_start or it == 1:
                self.model_A = PureDNN(input_dim=n_ctrl_dims + 1, hidden_dim=256, num_layers=5).to(self.device)
                self.model_B = PureDNN(input_dim=n_ctrl_dims, hidden_dim=256, num_layers=5).to(self.device)

            if hasattr(f_sig_cond, 'update'):
                f_sig_cond.update(X_ctrl, y_disc, w_in_cpu * w_s_cpu)
                fs_local_raw_t = torch.tensor(f_sig_cond(X_ctrl, y_disc), dtype=torch.float32, device=self.device)
                fs_grid_raw_t = self._eval_sig_grid_batched(f_sig_cond, X_ctrl, self.y_grid_cpu)
                Z_s_ev_t = torch.sum(fs_grid_raw_t * dy_weights_t, dim=1)
                fs_local_t = fs_local_raw_t / torch.clamp(Z_s_ev_t, min=1e-12)
                fs_cond_grid_t = fs_grid_raw_t / torch.clamp(Z_s_ev_t.unsqueeze(1), min=1e-12)

            # ==============================================================================
            # MODEL A: DENSITY RATIO ESTIMATION (DNN + GBDT)
            # ==============================================================================
            bkg_sample_cpu = bkg_sampler(N_tot)
            X_targ_A_np = np.hstack([X_ctrl, y_disc.reshape(-1, 1)])
            X_ref_A_np = np.hstack([X_ctrl, bkg_sample_cpu.reshape(-1, 1)])
            X_train_A = np.vstack([X_targ_A_np, X_ref_A_np])
            y_train_A = np.concatenate([np.ones(N_tot), np.zeros(N_tot)])
            
            w_num = w_in_cpu * w_b_cpu
            w_train_A_dnn = np.concatenate([w_num, w_num])
            
            self.model_A = train_dnn(self.model_A, X_train_A, y_train_A, w_train_A_dnn, self.device, epochs=self.dnn_epochs)
            
            Z_eval_Zb_t = torch.cat([
                torch.tensor(np.repeat(X_ctrl, self.n_nodes, axis=0), dtype=torch.float32, device=self.device),
                torch.tensor(np.tile(self.y_grid_cpu, N_tot), dtype=torch.float32, device=self.device).unsqueeze(1)
            ], dim=1)
            
            with torch.no_grad():
                logits_ref_A = predict_in_batches(self.model_A, torch.tensor(X_ref_A_np, dtype=torch.float32, device=self.device))
                R_DNN_ref_t = torch.exp(torch.clamp(logits_ref_A, -15, 15)).squeeze()
                
                logits_data_A = predict_in_batches(self.model_A, torch.tensor(X_targ_A_np, dtype=torch.float32, device=self.device))
                R_DNN_data_t = torch.exp(torch.clamp(logits_data_A, -15, 15)).squeeze()
                
                logits_grid_A = predict_in_batches(self.model_A, Z_eval_Zb_t)
                R_DNN_grid_t = torch.exp(torch.clamp(logits_grid_A, -15, 15)).view(N_tot, self.n_nodes)

            if self.gbdt_mode == 'XGB_GPU' and xgb is not None:
                w_train_A_gbdt = np.concatenate([w_num, w_num * R_DNN_ref_t.cpu().numpy()])
                self.gbdt_A = SignedXGBClassifier(n_estimators=10, max_depth=2, learning_rate=0.1, device=self.device)
                self.gbdt_A.fit(X_train_A, y_train_A, sample_weight=w_train_A_gbdt)
                
                p_data_np = np.clip(self.gbdt_A.predict_proba(X_targ_A_np)[:, 1], 1e-7, 1-1e-7)
                R_GBDT_data_t = torch.tensor(p_data_np / (1.0 - p_data_np), dtype=torch.float32, device=self.device)
                
                p_grid_np = np.clip(self.gbdt_A.predict_proba(Z_eval_Zb_t.cpu().numpy())[:, 1], 1e-7, 1-1e-7)
                R_GBDT_grid_t = torch.tensor(p_grid_np / (1.0 - p_grid_np), dtype=torch.float32, device=self.device).view(N_tot, self.n_nodes)
            else:
                R_GBDT_data_t = 1.0
                R_GBDT_grid_t = 1.0
                
            R_raw_data_t = torch.clamp(R_DNN_data_t * R_GBDT_data_t, 1e-6, 1e6)
            R_raw_grid_t = torch.clamp(R_DNN_grid_t * R_GBDT_grid_t, 1e-6, 1e6)

            # ROW-WISE SPLINE SMOOTHING 
            fs_1d_grid_temp = torch.sum(w_in_t.unsqueeze(1) * w_s_t.unsqueeze(1) * fs_cond_grid_t, dim=0) / torch.clamp(torch.sum(w_in_t * w_s_t), min=1e-12)
            S_t = self._build_row_smoother_matrix(fs_1d_grid_temp.cpu().numpy())
            R_sm_grid_t, R_sm_data_t = self._apply_row_smoothing(R_raw_grid_t, y_disc_t, y_grid_t, S_t)

            correction_grid_t = self._compute_algebraic_marginal_correction(
                R_sm_grid_t, fb_base_grid_t, dy_weights_t, w_in_t, w_b_t
            )
            correction_data_t = self._torch_interp1d_local(y_disc_t, y_grid_t, correction_grid_t)
            
            R_final_grid_t = R_sm_grid_t * correction_grid_t
            R_final_data_t = R_sm_data_t * correction_data_t
            
            Z_b_eval_t = torch.clamp(torch.sum(R_final_grid_t * fb_base_grid_t * dy_weights_t, dim=1), min=1e-12)
            fb_local_t = (fb_global_t * R_final_data_t) / Z_b_eval_t
            fb_cond_grid_t = (R_final_grid_t * fb_base_grid_t) / Z_b_eval_t.unsqueeze(1)

            # ==============================================================================
            # MODEL B: PRIOR ODDS ESTIMATION (DNN + GBDT)
            # ==============================================================================
            X_train_B = np.vstack([X_ctrl, X_ctrl])
            y_train_B = np.concatenate([np.ones(N_tot), np.zeros(N_tot)])
            w_train_B_dnn = np.concatenate([w_in_cpu * w_s_cpu, w_in_cpu * w_b_cpu])
            
            self.model_B = train_dnn(self.model_B, X_train_B, y_train_B, w_train_B_dnn, self.device, epochs=self.dnn_epochs)
            
            with torch.no_grad():
                logits_B = predict_in_batches(self.model_B, Z_ctrl_t)
                A_DNN_data_t = torch.exp(torch.clamp(logits_B, -15, 15)).squeeze()
                
            if self.gbdt_mode == 'XGB_GPU' and xgb is not None:
                w_train_B_gbdt = np.concatenate([w_in_cpu * w_s_cpu, w_in_cpu * w_b_cpu * A_DNN_data_t.cpu().numpy()])
                self.gbdt_B = SignedXGBClassifier(n_estimators=40, max_depth=4, learning_rate=0.1, device=self.device)
                self.gbdt_B.fit(X_train_B, y_train_B, sample_weight=w_train_B_gbdt)
                p_phys_np = np.clip(self.gbdt_B.predict_proba(X_ctrl)[:, 1], 1e-7, 1-1e-7)
                A_GBDT_data_t = torch.tensor(p_phys_np / (1.0 - p_phys_np), dtype=torch.float32, device=self.device)
            else:
                A_GBDT_data_t = 1.0
                
            A_final_data_t = A_DNN_data_t * A_GBDT_data_t

            # ==============================================================================
            # YIELD CALIBRATION (Optional Yield Locking)
            # ==============================================================================
            with torch.no_grad():
                pi_s_cond_t = torch.clamp(w_s_t, 0.0001, 0.9999)
                I_grid_t_eval = torch.clamp(pi_s_cond_t.unsqueeze(1) * fs_cond_grid_t + (1.0 - pi_s_cond_t).unsqueeze(1) * fb_cond_grid_t, min=1e-12)
                I_obs_t_eval = torch.clamp(pi_s_cond_t * fs_local_t + (1.0 - pi_s_cond_t) * fb_local_t, min=1e-12)
                
                sW_iter_t = self._compute_cows_weights_pytorch(
                    X_ctrl, y_disc, fs_cond_grid_t, fb_cond_grid_t, fs_local_t, fb_local_t,
                    I_grid_t_eval, I_obs_t_eval, dy_weights_t, self.y_grid_cpu, f_sig_cond, 
                    active_derivatives=self.extended_cows_derivatives
                )
                Ns_cows = torch.clamp(torch.sum(w_in_t * sW_iter_t), min=1.0)
                cows_yield_curr = Ns_cows.item()
            
            # Optional Synchronization to strictly lock EM yield to intermediate COWs target
            if self.lock_yield:
                c_scale = torch.ones((), dtype=torch.float32, device=self.device)
                for _ in range(6):
                    num_c = fs_local_t * (c_scale * A_final_data_t)
                    w_s_bayes_t = num_c / torch.clamp(num_c + fb_local_t, min=1e-12)
                    curr_bayes_sum = torch.clamp(torch.sum(w_in_t * w_s_bayes_t), min=1e-6)
                    c_scale = c_scale * (Ns_cows / curr_bayes_sum)
                A_final_data_t = c_scale * A_final_data_t
            
            num_bayes = fs_local_t * A_final_data_t
            
            # OMEGA ACCELERATED EM UPDATE
            w_s_bayes_t = num_bayes / torch.clamp(num_bayes + fb_local_t, min=1e-12)
            w_s_next_t = torch.clamp(w_s_t + omega_curr * (w_s_bayes_t - w_s_t), 0.0, 1.0)
            
            # Optional Clamping Repair
            if self.lock_yield and omega_curr > 1.0:
                for _ in range(2):
                    curr_sum = torch.sum(w_in_t * w_s_next_t)
                    if curr_sum > 1e-6:
                        w_s_next_t = torch.clamp(w_s_next_t * (Ns_cows / curr_sum), 0.0, 1.0)
            
            l1_shift = (torch.sum(torch.abs(w_s_next_t - w_s_t)) / torch.clamp(torch.sum(torch.abs(w_in_t)), min=1.0)).item()
            l1_history.append(l1_shift)
            current_Ns = torch.sum(w_in_t * w_s_next_t).item()
            
            # --- EVALUATE DIAGNOSTICS ---
            w_prob_cpu_eval = w_in_cpu * w_s_next_t.cpu().numpy()
            sW_eff_cpu_eval = w_in_cpu * sW_iter_t.cpu().numpy()
            
            chi2_str = ""
            chi2s = None
            if validation_data is not None and 'x_true' in validation_data:
                from utils.diagnostics import get_chi2
                chi2s = get_chi2(validation_data['x_raw'], w_prob_cpu_eval, validation_data['x_true'], validation_data['bounds'], w_true=validation_data.get('w_true'))
                sw_chi2s = get_chi2(validation_data['x_raw'], sW_eff_cpu_eval, validation_data['x_true'], validation_data['bounds'], w_true=validation_data.get('w_true'))
                mean_chi2 = np.mean(chi2s)
                self.history['mean_chi2'].append(mean_chi2)
                self.history['chi2s'].append(chi2s)
                
                p_c2_str = ", ".join([f"{c:.1f}" for c in chi2s])
                cw_c2_str = ", ".join([f"{c:.1f}" for c in sw_chi2s])
                
                chi2_str = (f" | Prob χ2: [{p_c2_str}] (Avg {mean_chi2:.1f}) | "
                            f"COWs χ2: [{cw_c2_str}] (Avg {np.mean(sw_chi2s):.1f}) | "
                            f"COWs Yield: {cows_yield_curr:.0f}")

            logger.info(f"Iter {it:02d} [w={omega_curr:.2f}] | Yield: {current_Ns:.0f} | L1 Shift: {l1_shift:.5f}{chi2_str}")
            
            if validation_data is not None and validation_data.get('plot_iterations', False):
                from utils.diagnostics import plot_dynamic_iteration
                
                sig_grid_t = (w_in_t * w_s_next_t).unsqueeze(1) * fs_cond_grid_t
                bkg_grid_t = (w_in_t * (1.0 - w_s_next_t)).unsqueeze(1) * fb_cond_grid_t
                p_mix_grid_t = sig_grid_t + bkg_grid_t

                pred_tot_density = torch.sum(p_mix_grid_t, dim=0).cpu().numpy()
                pred_sig_density = torch.sum(sig_grid_t, dim=0).cpu().numpy()
                pred_bkg_density = torch.sum(bkg_grid_t, dim=0).cpu().numpy()

                obs_counts, edges = np.histogram(y_disc, bins=25, range=(self.y_min, self.y_max), weights=w_in_cpu)
                m_centers = 0.5 * (edges[:-1] + edges[1:])
                m_bw = edges[1] - edges[0]
                
                mass_curves = {
                    'y_grid': self.y_grid_cpu, 'tot': pred_tot_density, 'sig': pred_sig_density, 'bkg': pred_bkg_density,
                    'centers': m_centers, 'obs_counts': obs_counts, 
                    'tot_counts': np.interp(m_centers, self.y_grid_cpu, pred_tot_density) * m_bw,
                    'sig_counts': np.interp(m_centers, self.y_grid_cpu, pred_sig_density) * m_bw, 
                    'bkg_counts': np.interp(m_centers, self.y_grid_cpu, pred_bkg_density) * m_bw, 
                    'morozov_chi2': 0.0
                }
                
                plot_dynamic_iteration(
                    x_raw=validation_data['x_raw'], w_s=w_prob_cpu_eval, sW_s=sW_eff_cpu_eval,
                    x_true=validation_data.get('x_true'), bounds=validation_data['bounds'],
                    labels=validation_data.get('labels'), it=it, chi2s=chi2s,
                    y_grid_cpu=self.y_grid_cpu, correction_grid=correction_grid_t.cpu().numpy(),
                    y_raw=validation_data.get('y_raw'), y_bounds=validation_data.get('y_bounds'),
                    w_true=validation_data.get('w_true'), mass_curves=mass_curves, w_in=w_in_cpu
                )

            w_s_t = w_s_next_t.clone()
            
            if len(l1_history) >= 5:
                recent_l1 = l1_history[-5:]
                slope, _ = np.polyfit(np.arange(5), recent_l1, 1)
                if (slope > -self.tol and l1_shift < 0.02) or (l1_shift < 0.001):
                    logger.info(f"\n--> CONVERGENCE REACHED: L1 gradient flattened (Slope: {slope:.2e}).")
                    break

        logger.info("\n--> Executing final pure COWs Projection...")
        w_s_cpu = w_s_t.cpu().numpy()
        self.w_s = w_s_cpu
        
        with torch.no_grad():
            pi_s_cond_t = torch.clamp(w_s_t, 0.0001, 0.9999)
            I_grid_t = torch.clamp(pi_s_cond_t.unsqueeze(1) * fs_cond_grid_t + (1.0 - pi_s_cond_t).unsqueeze(1) * fb_cond_grid_t, min=1e-12)
            I_obs_t = torch.clamp(pi_s_cond_t * fs_local_t + (1.0 - pi_s_cond_t) * fb_local_t, min=1e-12)
            
            sW_final_t = self._compute_cows_weights_pytorch(
                X_ctrl, y_disc, fs_cond_grid_t, fb_cond_grid_t, fs_local_t, fb_local_t,
                I_grid_t, I_obs_t, dy_weights_t, self.y_grid_cpu, f_sig_cond, 
                active_derivatives=self.extended_cows_derivatives
            )
            sWeights = w_in_cpu * sW_final_t.cpu().numpy()
            
        logger.info(f"Final Bayes Yield:  {current_Ns:.0f}")
        logger.info(f"Final COWs Yield: {np.sum(sWeights):.0f}")
        
        return sWeights
