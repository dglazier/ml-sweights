import numpy as np
import torch
import time
import copy
import logging
import scipy.ndimage  # <-- Added for Gaussian smoothing
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from .networks import PureDNN, train_dnn, predict_in_batches
from .quadrature import get_gauss_legendre_grid

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def torch_interp1d(x, xp, yp):
    """1D Linear interpolation executed entirely on the GPU."""
    idx = torch.searchsorted(xp, x)
    idx = torch.clamp(idx, 1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = yp[idx - 1], yp[idx]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

class MLSWeightExtractor:
    def __init__(self, disc_bounds, n_nodes=20, device='cuda', gbdt_mode='XGB_GPU', 
                 use_relu=True, reinstate_nn=False, 
                 marginal_alpha=1.0, dynamic_alpha=False, 
                 use_gbdt=True, dnn_epochs=50, ml_transform=None,
                 loss_fn='BCE', logit_l2_penalty=0.0, 
                 yield_boost=0.0, ry_smoothing=1.5):  # <-- NEW RUNTIME PARAMETER
        
        self.y_min, self.y_max = disc_bounds
        self.n_nodes = n_nodes
        self.device = device
        self.use_relu = use_relu
        self.reinstate_nn = reinstate_nn
        self.marginal_alpha = marginal_alpha 
        self.dynamic_alpha = dynamic_alpha 
        self.use_gbdt = use_gbdt
        self.dnn_epochs = dnn_epochs
        self.ml_transform = ml_transform 
        self.loss_fn = loss_fn 
        self.logit_l2_penalty = logit_l2_penalty 
        self.yield_boost = yield_boost
        self.ry_smoothing = ry_smoothing  # <-- STORED PARAMETER
        
        self.gbdt_mode = gbdt_mode if self.use_gbdt else 'NONE'
        if self.gbdt_mode == 'XGB_GPU' and (xgb is None or not torch.cuda.is_available()):
            logger.warning("XGBoost GPU not available. Falling back to SKLEARN.")
            self.gbdt_mode = 'SKLEARN'
            
        self.model_A = None
        self.gbdt_A = None
        self.model_B = None
        self.gbdt_B = None
        self.w_s = None
        self.correction_grid_t = None
        self.y_grid_cpu = None

    def fit_transform(self, X_ctrl, y_disc, w_s_init, f_sig_cond, f_bkg_1d, bkg_sampler, 
                      max_iters=40, tol=0.002, validation_data=None, 
                      omega_init=1.5, omega_decay_iters=10.0):
        
        # 1. INITIALIZE ENTIRE STATE SPACE ON GPU
        Z_ctrl = self.ml_transform(X_ctrl) if self.ml_transform else X_ctrl
        N_tot, n_ctrl_dims = Z_ctrl.shape
        
        Z_ctrl_t = torch.tensor(Z_ctrl, dtype=torch.float32, device=self.device)
        y_disc_t = torch.tensor(y_disc, dtype=torch.float32, device=self.device)
        w_s_prev_t = torch.tensor(w_s_init, dtype=torch.float32, device=self.device)
        
        y_grid_t, dy_weights_t = get_gauss_legendre_grid(self.y_min, self.y_max, n_nodes=self.n_nodes, device=self.device)
        self.y_grid_cpu = y_grid_t.cpu().numpy()
        
        fb_base_grid_t = torch.tensor(f_bkg_1d(self.y_grid_cpu), dtype=torch.float32, device=self.device)
        fb_global_t = torch.tensor(f_bkg_1d(y_disc), dtype=torch.float32, device=self.device)
        
        fs_local_raw_cpu = f_sig_cond(Z_ctrl, y_disc) 
        fs_local_raw_t = torch.tensor(fs_local_raw_cpu, dtype=torch.float32, device=self.device)
        
        Z_ctrl_repeated = np.repeat(Z_ctrl, self.n_nodes, axis=0)
        y_tiled = np.tile(self.y_grid_cpu, N_tot)
        fs_grid_raw_t = torch.tensor(f_sig_cond(Z_ctrl_repeated, y_tiled), dtype=torch.float32, device=self.device).view(N_tot, self.n_nodes)
        
        Z_s_ev_t = torch.sum(fs_grid_raw_t * dy_weights_t, dim=1)
        fs_local_t = fs_local_raw_t / torch.clamp(Z_s_ev_t, min=1e-12)
        
        Z_eval_Zb_t = torch.cat([
            torch.tensor(Z_ctrl_repeated, dtype=torch.float32, device=self.device),
            torch.tensor(y_tiled, dtype=torch.float32, device=self.device).unsqueeze(1)
        ], dim=1)

        if self.model_A is None or self.reinstate_nn:
            self.model_A = PureDNN(input_dim=n_ctrl_dims + 1, use_relu=self.use_relu, hidden_dim=256, num_layers=5).to(self.device)
            self.model_B = PureDNN(input_dim=n_ctrl_dims, use_relu=self.use_relu, hidden_dim=256, num_layers=5).to(self.device)

        stable_iters = 0
        required_stable_iters = 3
        best_w_s = w_s_init.copy()
        best_correction_grid_t = None
        best_model_A_state = None
        best_gbdt_A = None
        best_sig_state = None
        
        min_metric = float('inf')
        max_kl = -float('inf')
        best_iter = 0
        
        kl_history = []
        yield_history = []
        cooldown = 0
        current_alpha = self.marginal_alpha
        
        logger.info(f"Launching Accelerated GPU-Native EM Loop on {self.device}...")
        
        for it in range(1, max_iters + 1):
            t0 = time.time()
            w_b_t = 1.0 - w_s_prev_t
            
            decay_factor = max(0.0, (omega_decay_iters - it) / max(1.0, omega_decay_iters - 1.0))
            omega_curr = 1.0 + (omega_init - 1.0) * decay_factor
            
            if self.reinstate_nn:
                self.model_A = PureDNN(input_dim=n_ctrl_dims + 1, use_relu=self.use_relu, hidden_dim=256, num_layers=5).to(self.device)
                self.model_B = PureDNN(input_dim=n_ctrl_dims, use_relu=self.use_relu, hidden_dim=256, num_layers=5).to(self.device)
                
            if hasattr(f_sig_cond, 'update'):
                w_s_prev_cpu = w_s_prev_t.cpu().numpy()
                f_sig_cond.update(Z_ctrl, y_disc, w_s_prev_cpu)
                fs_local_raw_t = torch.tensor(f_sig_cond(Z_ctrl, y_disc), dtype=torch.float32, device=self.device)
                fs_grid_raw_t = torch.tensor(f_sig_cond(Z_ctrl_repeated, y_tiled), dtype=torch.float32, device=self.device).view(N_tot, self.n_nodes)
                Z_s_ev_t = torch.sum(fs_grid_raw_t * dy_weights_t, dim=1)
                fs_local_t = fs_local_raw_t / torch.clamp(Z_s_ev_t, min=1e-12)
            
            bkg_sample_cpu = bkg_sampler(N_tot)
            X_targ_A_t = torch.cat([Z_ctrl_t, y_disc_t.unsqueeze(1)], dim=1)
            X_ref_A_t = torch.cat([Z_ctrl_t, torch.tensor(bkg_sample_cpu, dtype=torch.float32, device=self.device).unsqueeze(1)], dim=1)
            
            X_train_A = torch.cat([X_targ_A_t, X_ref_A_t], dim=0).cpu().numpy()
            y_train_A = np.concatenate([np.ones(N_tot), np.zeros(N_tot)])
            
            w_b_cpu = w_b_t.cpu().numpy()
            w_train_A_dnn = np.concatenate([w_b_cpu, w_b_cpu]) 
            
            self.model_A = train_dnn(self.model_A, X_train_A, y_train_A, w_train_A_dnn, self.device, 
                                     epochs=self.dnn_epochs, loss_fn=self.loss_fn, logit_l2_penalty=self.logit_l2_penalty)
            
            with torch.no_grad():
                logits_ref_A = predict_in_batches(self.model_A, X_ref_A_t)
                R_DNN_ref_t = torch.exp(torch.clamp(logits_ref_A, -15, 15)).squeeze()
                
                logits_data_A = predict_in_batches(self.model_A, X_targ_A_t)
                R_DNN_data_t = torch.exp(torch.clamp(logits_data_A, -15, 15)).squeeze()
                
                logits_Zb_A = predict_in_batches(self.model_A, Z_eval_Zb_t)
                R_DNN_grid_t = torch.exp(torch.clamp(logits_Zb_A, -15, 15)).view(N_tot, self.n_nodes)

            w_train_A_gbdt = np.concatenate([w_b_cpu, w_b_cpu * R_DNN_ref_t.cpu().numpy()])
            w_train_A_gbdt /= np.mean(w_train_A_gbdt)
            
            if self.gbdt_mode == 'XGB_GPU':
                self.gbdt_A = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, tree_method='hist', device='cuda')
                X_train_A_t = torch.tensor(X_train_A, dtype=torch.float32, device=self.device)
                y_train_A_t = torch.tensor(y_train_A, dtype=torch.float32, device=self.device)
                w_train_A_t = torch.tensor(w_train_A_gbdt, dtype=torch.float32, device=self.device)
                
                self.gbdt_A.fit(X_train_A_t, y_train_A_t, sample_weight=w_train_A_t)
                
                p_data_raw = self.gbdt_A.predict_proba(X_targ_A_t)[:, 1]
                if isinstance(p_data_raw, np.ndarray): p_data_raw = torch.tensor(p_data_raw, device=self.device)
                p_data_t = torch.clamp(p_data_raw, 1e-7, 1-1e-7)
                R_GBDT_data_t = p_data_t / (1.0 - p_data_t)
                
                p_grid_raw = self.gbdt_A.predict_proba(Z_eval_Zb_t)[:, 1]
                if isinstance(p_grid_raw, np.ndarray): p_grid_raw = torch.tensor(p_grid_raw, device=self.device)
                p_grid_t = torch.clamp(p_grid_raw, 1e-7, 1-1e-7)
                R_GBDT_grid_t = (p_grid_t / (1.0 - p_grid_t)).view(N_tot, self.n_nodes)
                
            elif self.gbdt_mode == 'SKLEARN':
                self.gbdt_A = HistGradientBoostingClassifier(max_iter=50, max_depth=4, learning_rate=0.1)
                self.gbdt_A.fit(X_train_A, y_train_A, sample_weight=w_train_A_gbdt)
                
                p_data = np.clip(self.gbdt_A.predict_proba(X_train_A[:N_tot])[:, 1], 1e-7, 1-1e-7)
                R_GBDT_data_t = torch.tensor(p_data / (1.0 - p_data), dtype=torch.float32, device=self.device)
                
                p_grid = np.clip(self.gbdt_A.predict_proba(Z_eval_Zb_t.cpu().numpy())[:, 1], 1e-7, 1-1e-7)
                R_GBDT_grid_t = torch.tensor((p_grid / (1.0 - p_grid)).reshape(N_tot, self.n_nodes), dtype=torch.float32, device=self.device)
            else:
                R_GBDT_data_t = torch.ones(N_tot, device=self.device)
                R_GBDT_grid_t = torch.ones((N_tot, self.n_nodes), device=self.device)
            
            R_raw_data_t = torch.clamp(R_DNN_data_t * R_GBDT_data_t, 1e-4, 1e4)
            R_raw_grid_t = torch.clamp(R_DNN_grid_t * R_GBDT_grid_t, 1e-4, 1e4)
            
            R_y_margin_t = torch.mean(R_raw_grid_t, dim=0)
            f_margin_grid_t = fb_base_grid_t * R_y_margin_t
            
            I_base_t = torch.sum(fb_base_grid_t * dy_weights_t)
            I_margin_t = torch.sum(f_margin_grid_t * dy_weights_t)
            
            correction_factor_grid_t = (I_margin_t / torch.clamp(I_base_t, min=1e-10)) / torch.clamp(R_y_margin_t, min=1e-10)
            
            # -------------------------------------------------------------
            # NEW HANDLE: ENFORCE BACKGROUND SMOOTHNESS UNDER SIGNAL PEAK
            # -------------------------------------------------------------
            if self.ry_smoothing > 0.0:
                corr_cpu = correction_factor_grid_t.cpu().numpy()
                corr_smoothed = scipy.ndimage.gaussian_filter1d(corr_cpu, sigma=self.ry_smoothing)
                correction_factor_grid_t = torch.tensor(corr_smoothed, dtype=torch.float32, device=self.device)
            # -------------------------------------------------------------
            
            self.correction_grid_t = torch.pow(correction_factor_grid_t, current_alpha)
            
            correction_interp_data_t = torch_interp1d(y_disc_t, y_grid_t, self.correction_grid_t)
            R_final_grid_t = R_raw_grid_t * self.correction_grid_t
            R_final_data_t = R_raw_data_t * correction_interp_data_t
            
            Z_b_ev_t = torch.sum(R_final_grid_t * fb_base_grid_t * dy_weights_t, dim=1)
            fb_local_t = (fb_global_t * R_final_data_t) / torch.clamp(Z_b_ev_t, min=1e-12)
            
            X_train_B = torch.cat([Z_ctrl_t, Z_ctrl_t], dim=0).cpu().numpy()
            y_train_B = np.concatenate([np.ones(N_tot), np.zeros(N_tot)])
            w_train_B_dnn = np.concatenate([w_s_prev_t.cpu().numpy(), w_b_cpu])
            
            self.model_B = train_dnn(self.model_B, X_train_B, y_train_B, w_train_B_dnn, self.device, 
                                     epochs=self.dnn_epochs, loss_fn=self.loss_fn, logit_l2_penalty=self.logit_l2_penalty)
            
            with torch.no_grad():
                logits_B = predict_in_batches(self.model_B, Z_ctrl_t)
                A_DNN_data_t = torch.exp(torch.clamp(logits_B, -15, 15)).squeeze()
            
            w_train_B_gbdt = np.concatenate([w_s_prev_t.cpu().numpy(), w_b_cpu * A_DNN_data_t.cpu().numpy()])
            w_train_B_gbdt /= np.mean(w_train_B_gbdt)
            
            if self.gbdt_mode == 'XGB_GPU':
                self.gbdt_B = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, tree_method='hist', device='cuda')
                X_train_B_t = torch.tensor(X_train_B, dtype=torch.float32, device=self.device)
                y_train_B_t = torch.tensor(y_train_B, dtype=torch.float32, device=self.device)
                w_train_B_t = torch.tensor(w_train_B_gbdt, dtype=torch.float32, device=self.device)
                
                self.gbdt_B.fit(X_train_B_t, y_train_B_t, sample_weight=w_train_B_t)
                
                p_phys_raw = self.gbdt_B.predict_proba(Z_ctrl_t)[:, 1]
                if isinstance(p_phys_raw, np.ndarray): p_phys_raw = torch.tensor(p_phys_raw, device=self.device)
                p_phys_t = torch.clamp(p_phys_raw, 1e-7, 1-1e-7)
                A_GBDT_data_t = p_phys_t / (1.0 - p_phys_t)
                
            elif self.gbdt_mode == 'SKLEARN':
                self.gbdt_B = HistGradientBoostingClassifier(max_iter=50, max_depth=4, learning_rate=0.1)
                self.gbdt_B.fit(X_train_B, y_train_B, sample_weight=w_train_B_gbdt)
                
                p_phys = np.clip(self.gbdt_B.predict_proba(Z_ctrl)[:, 1], 1e-7, 1-1e-7)
                A_GBDT_data_t = torch.tensor(p_phys / (1.0 - p_phys), dtype=torch.float32, device=self.device)
            else:
                A_GBDT_data_t = torch.ones(N_tot, device=self.device)
            
            A_final_data_t = A_DNN_data_t * A_GBDT_data_t
            
            w_s_safe_t = torch.clamp(w_s_prev_t, min=1e-10)
            log_A_t = torch.log(torch.clamp(A_final_data_t, min=1e-8, max=1e8))
            kl_divergence = (torch.sum(w_s_safe_t * log_A_t) / torch.sum(w_s_safe_t)).item()
            kl_history.append(kl_divergence)
            
            # -------------------------------------------------------------
            # NEW IMPLEMENTATION: OPTIONAL YIELD BOOST CONSTRAINT
            # -------------------------------------------------------------
            if self.yield_boost > 0.0:
                fs_norm = fs_local_t / torch.clamp(torch.max(fs_local_t), min=1e-12)
                fb_effective = torch.clamp(fb_local_t * (1.0 - self.yield_boost * fs_norm), min=1e-12)
            else:
                fb_effective = fb_local_t

            w_s_bayes_t = (fs_local_t * A_final_data_t) / torch.clamp((fs_local_t * A_final_data_t) + fb_effective, min=1e-12)
            w_s_t = torch.clamp(w_s_prev_t + omega_curr * (w_s_bayes_t - w_s_prev_t), 0.0, 1.0)
            
            current_Ns = torch.sum(w_s_t).item()
            yield_history.append(current_Ns)
            delta_Ns = current_Ns - torch.sum(w_s_prev_t).item()
            
            # -------------------------------------------------------------
            # CORRECTION: MACROSCOPIC YIELD STABILIZATION (Yield Shift vs delta_w)
            # -------------------------------------------------------------
            fractional_yield_shift = abs(delta_Ns) / max(current_Ns, 1.0)
            
            if cooldown > 0:
                cooldown -= 1
            elif self.dynamic_alpha and len(kl_history) >= 3:
                kl_dropping = (kl_history[-1] < kl_history[-2]) and (kl_history[-2] < kl_history[-3])
                kl_rising = (kl_history[-1] > kl_history[-2]) and (kl_history[-2] > kl_history[-3])
                yield_bleeding = (yield_history[-1] < yield_history[-2])
                
                if kl_dropping and yield_bleeding:
                    current_alpha = min(1.0, current_alpha + 0.1)
                    cooldown = 3
                elif kl_rising:
                    current_alpha = max(0.0, current_alpha - 0.05)
                    cooldown = 2
            
            if validation_data is not None:
                from utils.diagnostics import get_chi2, plot_dynamic_iteration
                w_s_cpu = w_s_t.cpu().numpy()
                
                if validation_data.get('x_true') is not None:
                    chi2s = get_chi2(validation_data['x_raw'], w_s_cpu, validation_data['x_true'], 
                                     validation_data['bounds'], w_true=validation_data.get('w_true', None))
                    chi2_str = ", ".join([f"{c:.1f}" for c in chi2s])
                else:
                    chi2s = []
                    chi2_str = "N/A"
                    
                if validation_data.get('real_data_metric', False):
                    current_metric = fractional_yield_shift - (0.1 * kl_divergence)
                    logger.info(f"Iter {it:02d} [ω={omega_curr:.2f}, α={current_alpha:.2f}] | S: {current_Ns:.0f} (dNs: {delta_Ns:+.0f}) | "
                                f"Ref Chi2: [{chi2_str}] | <KL>: {kl_divergence:.4f} | Yield Shift: {fractional_yield_shift:.5f}")
                else:
                    current_metric = np.mean(chi2s) if len(chi2s) > 0 else 0.0
                    logger.info(f"Iter {it:02d} [ω={omega_curr:.2f}, α={current_alpha:.2f}] | S: {current_Ns:.0f} (dNs: {delta_Ns:+.0f}) | "
                                f"Chi2: [{chi2_str}] | <KL>: {kl_divergence:.4f} | Yield Shift: {fractional_yield_shift:.5f}")

                if validation_data.get('plot_iterations', False):
                    corr_grid_cpu = self.correction_grid_t.cpu().numpy() if self.correction_grid_t is not None else None
                    plot_dynamic_iteration(
                        x_raw=validation_data['x_raw'], 
                        w_s=w_s_cpu, 
                        x_true=validation_data.get('x_true'), 
                        bounds=validation_data['bounds'], 
                        labels=validation_data.get('labels', None), 
                        it=it, 
                        chi2s=chi2s, 
                        y_grid_cpu=self.y_grid_cpu, 
                        correction_grid=corr_grid_cpu,
                        y_raw=validation_data.get('y_raw', None), 
                        y_bounds=validation_data.get('y_bounds', None),
                        w_true=validation_data.get('w_true', None)
                    )
            else:
                current_metric = fractional_yield_shift - (0.1 * kl_divergence)
                logger.info(f"Iter {it:02d} [ω={omega_curr:.2f}, α={current_alpha:.2f}] | S: {current_Ns:.0f} (dNs: {delta_Ns:+.0f}) | "
                            f"<KL>: {kl_divergence:.4f} | Yield Shift: {fractional_yield_shift:.5f}")
            
            if it > 5:
                if current_metric < min_metric:
                    min_metric = current_metric
                    max_kl = kl_divergence
                    best_w_s = w_s_t.cpu().numpy()
                    best_correction_grid_t = self.correction_grid_t.clone() if self.correction_grid_t is not None else None
                    
                    best_model_A_state = copy.deepcopy(self.model_A.state_dict())
                    if getattr(self, 'gbdt_A', None) is not None:
                        best_gbdt_A = copy.deepcopy(self.gbdt_A)
                    if hasattr(f_sig_cond, 'get_state'):
                        best_sig_state = f_sig_cond.get_state()
                        
                    best_iter = it
            
            if fractional_yield_shift < tol and it > 5:
                stable_iters += 1
            else:
                stable_iters = 0
                
            if stable_iters >= required_stable_iters:
                logger.info(f"\n--> MACROSCOPIC YIELD STABILIZATION DETECTED.")
                break
                
            w_s_prev_t = w_s_t.clone()
            
            if self.gbdt_mode == 'XGB_GPU':
                del self.gbdt_A
                del self.gbdt_B
            
        logger.info(f"\n--> Restoring optimal physical weights from Iteration {best_iter} (Max KL: {max_kl:.4f})")
        self.w_s = best_w_s
        if best_correction_grid_t is not None:
            self.correction_grid_t = best_correction_grid_t
            
        self.model_A.load_state_dict(best_model_A_state)
        if best_gbdt_A is not None:
            self.gbdt_A = best_gbdt_A
        if hasattr(f_sig_cond, 'load_state') and best_sig_state is not None:
            f_sig_cond.load_state(best_sig_state)
        
        logger.info("Calculating mathematically rigorous N-Dimensional sWeights...")
        return self._calculate_nd_sweights(X_ctrl, y_disc, self.w_s, f_sig_cond, f_bkg_1d)
    
    def _calculate_nd_sweights(self, X_ctrl, y_disc, w_s_final, f_sig_cond, f_bkg_1d, batch_size=32768):
        N_tot = len(y_disc)
        N_s = np.sum(w_s_final)
        N_b = N_tot - N_s
        sWeights_MD = np.zeros(N_tot)
        
        self.model_A.eval()
        if self.model_B is not None:
            self.model_B.eval()
            
        y_grid_t, dy_weights_t = get_gauss_legendre_grid(self.y_min, self.y_max, n_nodes=self.n_nodes, device=self.device)
        dy_weights_cpu = dy_weights_t.cpu().numpy()
        
        with torch.no_grad():
            for start_idx in range(0, N_tot, batch_size):
                end_idx = min(start_idx + batch_size, N_tot)
                batch_len = end_idx - start_idx
                
                x_b = X_ctrl[start_idx:end_idx]
                y_b = y_disc[start_idx:end_idx]
                z_b = self.ml_transform(x_b) if self.ml_transform else x_b
                
                z_b_t = torch.tensor(z_b, dtype=torch.float32, device=self.device)
                
                Z_eval_obs_t = torch.cat([
                    z_b_t,
                    torch.tensor(y_b, dtype=torch.float32, device=self.device).unsqueeze(1)
                ], dim=1)
                
                fs_obs_raw = f_sig_cond(z_b, y_b) 
                fb_base_obs = f_bkg_1d(y_b)
                
                R_dnn_obs = torch.exp(torch.clamp(self.model_A(Z_eval_obs_t), -15, 15)).cpu().numpy().flatten()
                
                if getattr(self, 'gbdt_A', None) is not None:
                    if self.gbdt_mode == 'XGB_GPU':
                        p_gbdt_raw = self.gbdt_A.predict_proba(Z_eval_obs_t)[:, 1]
                        if isinstance(p_gbdt_raw, torch.Tensor): p_gbdt_raw = p_gbdt_raw.cpu().numpy()
                        R_gbdt_obs = np.clip(p_gbdt_raw, 1e-7, 1-1e-7) / (1.0 - np.clip(p_gbdt_raw, 1e-7, 1-1e-7))
                    elif self.gbdt_mode == 'SKLEARN':
                        p_gbdt = np.clip(self.gbdt_A.predict_proba(Z_eval_obs_t.cpu().numpy())[:, 1], 1e-7, 1-1e-7)
                        R_gbdt_obs = p_gbdt / (1.0 - p_gbdt)
                    else:
                        R_gbdt_obs = np.ones(batch_len)
                else:
                    R_gbdt_obs = np.ones(batch_len)
                    
                if self.correction_grid_t is not None:
                    correction_interp_obs_t = torch_interp1d(
                        torch.tensor(y_b, dtype=torch.float32, device=self.device), 
                        y_grid_t, 
                        self.correction_grid_t
                    )
                    correction_interp_obs = correction_interp_obs_t.cpu().numpy()
                else:
                    correction_interp_obs = np.ones(batch_len)
                    
                fb_obs_raw = fb_base_obs * R_dnn_obs * R_gbdt_obs * correction_interp_obs
                
                z_b_repeated = np.repeat(z_b, self.n_nodes, axis=0)
                y_tiled = np.tile(self.y_grid_cpu, batch_len)
                
                fs_grid_raw = f_sig_cond(z_b_repeated, y_tiled).reshape(batch_len, self.n_nodes)
                
                Z_eval_grid_t = torch.cat([
                    torch.tensor(z_b_repeated, dtype=torch.float32, device=self.device),
                    torch.tensor(y_tiled, dtype=torch.float32, device=self.device).unsqueeze(1)
                ], dim=1)
                
                fb_base_grid = f_bkg_1d(y_tiled)
                R_dnn_grid = torch.exp(torch.clamp(self.model_A(Z_eval_grid_t), -15, 15)).cpu().numpy().flatten()
                
                if getattr(self, 'gbdt_A', None) is not None:
                    if self.gbdt_mode == 'XGB_GPU':
                        p_gbdt_grid = np.clip(self.gbdt_A.predict_proba(Z_eval_grid_t)[:, 1], 1e-7, 1-1e-7)
                        if isinstance(p_gbdt_grid, torch.Tensor): p_gbdt_grid = p_gbdt_grid.cpu().numpy()
                        R_gbdt_grid = p_gbdt_grid / (1.0 - p_gbdt_grid)
                    elif self.gbdt_mode == 'SKLEARN':
                        p_gbdt_grid = np.clip(self.gbdt_A.predict_proba(Z_eval_grid_t.cpu().numpy())[:, 1], 1e-7, 1-1e-7)
                        R_gbdt_grid = p_gbdt_grid / (1.0 - p_gbdt_grid)
                    else:
                        R_gbdt_grid = np.ones(len(y_tiled))
                else:
                    R_gbdt_grid = np.ones(len(y_tiled))
                    
                if self.correction_grid_t is not None:
                    correction_interp_grid = np.tile(self.correction_grid_t.cpu().numpy(), batch_len)
                else:
                    correction_interp_grid = np.ones(len(y_tiled))
                    
                fb_grid_raw = (fb_base_grid * R_dnn_grid * R_gbdt_grid * correction_interp_grid).reshape(batch_len, self.n_nodes)
                
                # Strict Normalization
                Z_s_ev = np.sum(fs_grid_raw * dy_weights_cpu, axis=1)
                Z_b_ev = np.sum(fb_grid_raw * dy_weights_cpu, axis=1)
                
                fs_cond_obs = fs_obs_raw / np.maximum(Z_s_ev, 1e-12)
                fb_cond_obs = fb_obs_raw / np.maximum(Z_b_ev, 1e-12)
                
                fs_cond_grid = fs_grid_raw / np.maximum(Z_s_ev[:, None], 1e-12)
                fb_cond_grid = fb_grid_raw / np.maximum(Z_b_ev[:, None], 1e-12)
                
                # Local COWs Integration
                I_grid = N_s * fs_cond_grid + N_b * fb_cond_grid
                
                W_ss_local = np.sum((fs_cond_grid**2 / np.maximum(I_grid, 1e-12)) * dy_weights_cpu, axis=1)
                W_bb_local = np.sum((fb_cond_grid**2 / np.maximum(I_grid, 1e-12)) * dy_weights_cpu, axis=1)
                W_sb_local = np.sum((fs_cond_grid * fb_cond_grid / np.maximum(I_grid, 1e-12)) * dy_weights_cpu, axis=1)
                
                Det_local = np.maximum(W_ss_local * W_bb_local - W_sb_local**2, 1e-14)
                A_ss_local = W_bb_local / Det_local
                A_sb_local = -W_sb_local / Det_local
                
                I_obs = N_s * fs_cond_obs + N_b * fb_cond_obs
                sWeights_MD[start_idx:end_idx] = (A_ss_local * fs_cond_obs + A_sb_local * fb_cond_obs) / np.maximum(I_obs, 1e-12)

        return sWeights_MD
