import numpy as np
import torch
import time
import copy
import logging
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from .networks import PureDNN, train_dnn, predict_in_batches
from .quadrature import get_gauss_legendre_grid

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class MLSWeightExtractor:
    def __init__(self, disc_bounds, n_nodes=20, device='cuda', gbdt_mode='XGB_GPU', 
                 use_relu=True, reinstate_nn=False, 
                 marginal_alpha=1.0, dynamic_alpha=False, 
                 use_gbdt=True, dnn_epochs=50, ml_transform=None,
                 loss_fn='BCE', logit_l2_penalty=0.0): # <-- Added Configuration Flags
        
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
        
        self.gbdt_mode = gbdt_mode if self.use_gbdt else 'NONE'
        if self.gbdt_mode == 'XGB_GPU' and (xgb is None or not torch.cuda.is_available()):
            logger.warning("XGBoost GPU not available. Falling back to SKLEARN.")
            self.gbdt_mode = 'SKLEARN'
            
        self.model_A = None
        self.gbdt_A = None
        self.model_B = None
        self.gbdt_B = None
        self.w_s = None
        self.correction_grid = None
        self.y_grid_cpu = None

    def fit_transform(self, X_ctrl, y_disc, w_s_init, f_sig_cond, f_bkg_1d, bkg_sampler, 
                      max_iters=40, tol=0.002, validation_data=None, 
                      omega_init=1.5, omega_decay_iters=10.0):
        
        N_tot, n_ctrl_dims = X_ctrl.shape
        w_s_prev = w_s_init.copy()
        
        Z_ctrl = self.ml_transform(X_ctrl) if self.ml_transform else X_ctrl
        
        y_grid_t, dy_weights_t = get_gauss_legendre_grid(self.y_min, self.y_max, n_nodes=self.n_nodes, device=self.device)
        y_grid_cpu = y_grid_t.cpu().numpy()
        dy_weights_cpu = dy_weights_t.cpu().numpy()
        self.y_grid_cpu = y_grid_cpu
        
        fb_base_grid = f_bkg_1d(y_grid_cpu) 
        X_ctrl_repeated = np.repeat(X_ctrl, self.n_nodes, axis=0)
        y_tiled = np.tile(y_grid_cpu, N_tot)
        
        Z_ctrl_repeated = np.repeat(Z_ctrl, self.n_nodes, axis=0)
        Z_eval_Zb_cpu = np.column_stack([Z_ctrl_repeated, y_tiled])
        Z_eval_Zb_t = torch.tensor(Z_eval_Zb_cpu, dtype=torch.float32, device=self.device)
        
        Z_data_t = torch.tensor(Z_ctrl, dtype=torch.float32, device=self.device)
        fb_global = f_bkg_1d(y_disc)
        
        fs_local_raw = f_sig_cond(X_ctrl, y_disc)
        fs_grid_raw = f_sig_cond(X_ctrl_repeated, y_tiled)
        fs_grid_2d = fs_grid_raw.reshape(N_tot, self.n_nodes)
        
        Z_s_ev = np.sum(fs_grid_2d * dy_weights_cpu, axis=1)
        fs_local = fs_local_raw / Z_s_ev

        if self.model_A is None or self.reinstate_nn:
            self.model_A = PureDNN(input_dim=n_ctrl_dims + 1, use_relu=self.use_relu).to(self.device)
            self.model_B = PureDNN(input_dim=n_ctrl_dims, use_relu=self.use_relu).to(self.device)

        stable_iters = 0
        required_stable_iters = 3
        
        best_w_s = w_s_prev.copy()
        best_correction_grid = None
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
        logger.info(f"Launching Accelerated Physics-Informed EM Loop on {self.device}...")
        
        for it in range(1, max_iters + 1):
            t0 = time.time()
            w_b = 1.0 - w_s_prev
            
            decay_factor = max(0.0, (omega_decay_iters - it) / max(1.0, omega_decay_iters - 1.0))
            omega_curr = 1.0 + (omega_init - 1.0) * decay_factor
            
            if self.reinstate_nn:
                self.model_A = PureDNN(input_dim=n_ctrl_dims + 1, use_relu=self.use_relu).to(self.device)
                self.model_B = PureDNN(input_dim=n_ctrl_dims, use_relu=self.use_relu).to(self.device)
                
            if hasattr(f_sig_cond, 'update'):
                f_sig_cond.update(X_ctrl, y_disc, w_s_prev)
                fs_local_raw = f_sig_cond(X_ctrl, y_disc)
                fs_grid_raw = f_sig_cond(X_ctrl_repeated, y_tiled)
                fs_grid_2d = fs_grid_raw.reshape(N_tot, self.n_nodes)
                Z_s_ev = np.sum(fs_grid_2d * dy_weights_cpu, axis=1)
                fs_local = fs_local_raw / Z_s_ev
            
            X_targ_A = np.column_stack((Z_ctrl, y_disc))
            X_ref_A = np.column_stack((Z_ctrl, bkg_sampler(N_tot)))
            X_train_A = np.vstack([X_targ_A, X_ref_A])
            y_train_A = np.concatenate([np.ones(N_tot), np.zeros(N_tot)])
            
            w_train_A_dnn = np.concatenate([w_b, w_b]) 
            
            # --> MODIFIED: Passing configured loss parameters
            self.model_A = train_dnn(self.model_A, X_train_A, y_train_A, w_train_A_dnn, self.device, 
                                     epochs=self.dnn_epochs, loss_fn=self.loss_fn, logit_l2_penalty=self.logit_l2_penalty)
            
            with torch.no_grad():
                logits_ref_A = torch.clamp(predict_in_batches(self.model_A, torch.tensor(X_ref_A, dtype=torch.float32).to(self.device)), -15, 15)
                R_DNN_ref = torch.exp(logits_ref_A).cpu().numpy().flatten()
                logits_data_A = torch.clamp(predict_in_batches(self.model_A, torch.tensor(X_targ_A, dtype=torch.float32).to(self.device)), -15, 15)
                R_DNN_data = torch.exp(logits_data_A).cpu().numpy().flatten()
                logits_Zb_A = torch.clamp(predict_in_batches(self.model_A, Z_eval_Zb_t), -15, 15)
                R_DNN_grid = torch.exp(logits_Zb_A).cpu().numpy().reshape(N_tot, self.n_nodes)

            w_train_A_gbdt = np.concatenate([w_b, w_b * R_DNN_ref])
            w_train_A_gbdt /= np.mean(w_train_A_gbdt)
            
            if self.gbdt_mode == 'XGB_GPU':
                self.gbdt_A = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, tree_method='hist', device='cuda')
                X_train_A_t = torch.tensor(X_train_A, dtype=torch.float32, device=self.device)
                y_train_A_t = torch.tensor(y_train_A, dtype=torch.float32, device=self.device)
                w_train_A_t = torch.tensor(w_train_A_gbdt, dtype=torch.float32, device=self.device)
                X_targ_A_t = torch.tensor(X_targ_A, dtype=torch.float32, device=self.device)
                
                self.gbdt_A.fit(X_train_A_t, y_train_A_t, sample_weight=w_train_A_t)
                
                p_data_raw = self.gbdt_A.predict_proba(X_targ_A_t)[:, 1]
                p_data_raw = p_data_raw.cpu().numpy() if isinstance(p_data_raw, torch.Tensor) else p_data_raw
                p_data = np.clip(p_data_raw, 1e-7, 1-1e-7)
                R_GBDT_data = p_data / (1.0 - p_data)
                
                p_grid_raw = self.gbdt_A.predict_proba(Z_eval_Zb_t)[:, 1]
                p_grid_raw = p_grid_raw.cpu().numpy() if isinstance(p_grid_raw, torch.Tensor) else p_grid_raw
                p_grid = np.clip(p_grid_raw, 1e-7, 1-1e-7)
                R_GBDT_grid = (p_grid / (1.0 - p_grid)).reshape(N_tot, self.n_nodes)
                
            elif self.gbdt_mode == 'SKLEARN':
                self.gbdt_A = HistGradientBoostingClassifier(max_iter=50, max_depth=4, learning_rate=0.1)
                self.gbdt_A.fit(X_train_A, y_train_A, sample_weight=w_train_A_gbdt)
                p_data = np.clip(self.gbdt_A.predict_proba(X_targ_A)[:, 1], 1e-7, 1-1e-7)
                R_GBDT_data = p_data / (1.0 - p_data)
                p_grid = np.clip(self.gbdt_A.predict_proba(Z_eval_Zb_cpu)[:, 1], 1e-7, 1-1e-7)
                R_GBDT_grid = (p_grid / (1.0 - p_grid)).reshape(N_tot, self.n_nodes)
            else:
                R_GBDT_data = np.ones(N_tot)
                R_GBDT_grid = np.ones((N_tot, self.n_nodes))
            
            R_raw_data = np.clip(R_DNN_data * R_GBDT_data, 1e-4, 1e4)
            R_raw_grid = np.clip(R_DNN_grid * R_GBDT_grid, 1e-4, 1e4)
            
            R_y_margin = np.mean(R_raw_grid, axis=0)
            f_margin_grid = fb_base_grid * R_y_margin
            
            I_base = np.sum(fb_base_grid * dy_weights_cpu)
            I_margin = np.sum(f_margin_grid * dy_weights_cpu)
            
            correction_factor_grid = (I_margin / np.maximum(I_base, 1e-10)) / np.maximum(R_y_margin, 1e-10)
            self.correction_grid = np.power(correction_factor_grid, current_alpha)
            
            correction_interp_data = np.interp(y_disc, y_grid_cpu, self.correction_grid)
            R_final_grid = R_raw_grid * self.correction_grid
            R_final_data = R_raw_data * correction_interp_data
            
            Z_b_ev = np.sum(R_final_grid * fb_base_grid * dy_weights_cpu, axis=1)
            fb_local = (fb_global * R_final_data) / Z_b_ev
            
            X_train_B = np.vstack([Z_ctrl, Z_ctrl])
            y_train_B = np.concatenate([np.ones(N_tot), np.zeros(N_tot)])
            
            w_train_B_dnn = np.concatenate([w_s_prev, w_b])
            
            # --> MODIFIED: Passing configured loss parameters
            self.model_B = train_dnn(self.model_B, X_train_B, y_train_B, w_train_B_dnn, self.device, 
                                     epochs=self.dnn_epochs, loss_fn=self.loss_fn, logit_l2_penalty=self.logit_l2_penalty)
            
            with torch.no_grad():
                logits_B = torch.clamp(predict_in_batches(self.model_B, Z_data_t), -15, 15)
                A_DNN_data = torch.exp(logits_B).cpu().numpy().flatten()
            
            w_train_B_gbdt = np.concatenate([w_s_prev, w_b * A_DNN_data])
            w_train_B_gbdt /= np.mean(w_train_B_gbdt)
            
            if self.gbdt_mode == 'XGB_GPU':
                self.gbdt_B = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, tree_method='hist', device='cuda')
                X_train_B_t = torch.tensor(X_train_B, dtype=torch.float32, device=self.device)
                y_train_B_t = torch.tensor(y_train_B, dtype=torch.float32, device=self.device)
                w_train_B_t = torch.tensor(w_train_B_gbdt, dtype=torch.float32, device=self.device)
                
                self.gbdt_B.fit(X_train_B_t, y_train_B_t, sample_weight=w_train_B_t)
                
                p_phys_raw = self.gbdt_B.predict_proba(Z_data_t)[:, 1]
                p_phys_raw = p_phys_raw.cpu().numpy() if isinstance(p_phys_raw, torch.Tensor) else p_phys_raw
                p_phys = np.clip(p_phys_raw, 1e-7, 1-1e-7)
                A_GBDT_data = p_phys / (1.0 - p_phys)
                
            elif self.gbdt_mode == 'SKLEARN':
                self.gbdt_B = HistGradientBoostingClassifier(max_iter=50, max_depth=4, learning_rate=0.1)
                self.gbdt_B.fit(X_train_B, y_train_B, sample_weight=w_train_B_gbdt)
                p_phys = np.clip(self.gbdt_B.predict_proba(Z_ctrl)[:, 1], 1e-7, 1-1e-7)
                A_GBDT_data = p_phys / (1.0 - p_phys)
                
            else:
                A_GBDT_data = np.ones(N_tot)
            
            A_final_data = A_DNN_data * A_GBDT_data
            
            w_s_safe = np.maximum(w_s_prev, 1e-10)
            log_A = np.log(np.clip(A_final_data, 1e-8, 1e8))
            kl_divergence = np.sum(w_s_safe * log_A) / np.sum(w_s_safe)
            kl_history.append(kl_divergence)
            
            w_s_bayes = (fs_local * A_final_data) / np.maximum((fs_local * A_final_data) + fb_local, 1e-10)
            w_s = np.clip(w_s_prev + omega_curr * (w_s_bayes - w_s_prev), 0.0, 1.0)
            
            current_Ns = np.sum(w_s)
            yield_history.append(current_Ns)
            delta_Ns = current_Ns - np.sum(w_s_prev)
            delta_w = np.mean(np.abs(w_s - w_s_prev))
            
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
                from utils.diagnostics import get_chi2
                chi2s = get_chi2(validation_data['x_raw'], w_s, validation_data['x_true'], validation_data['bounds'])
                chi2_str = ", ".join([f"{c:.1f}" for c in chi2s])
                current_metric = np.mean(chi2s) 
                
                logger.info(f"Iter {it:02d} [ω={omega_curr:.2f}, α={current_alpha:.2f}] | S: {current_Ns:.0f} (dNs: {delta_Ns:+.0f}) | "
                            f"Chi2: [{chi2_str}] | <KL>: {kl_divergence:.4f} | Conv dW: {delta_w:.5f}")

                if validation_data.get('plot_iterations', False):
                    from utils.diagnostics import plot_dynamic_iteration
                    plot_dynamic_iteration(
                        validation_data['x_raw'], w_s, validation_data['x_true'], 
                        validation_data['bounds'], validation_data.get('labels', None), 
                        it, chi2s, y_grid_cpu, self.correction_grid,
                        validation_data.get('y_raw', None), validation_data.get('y_bounds', None) 
                    )
            else:
                current_metric = delta_w 
                logger.info(f"Iter {it:02d} [ω={omega_curr:.2f}, α={current_alpha:.2f}] | S: {current_Ns:.0f} (dNs: {delta_Ns:+.0f}) | "
                            f"<KL>: {kl_divergence:.4f} | Conv dW: {delta_w:.5f}")
            
            if it > 5:
                if current_metric < min_metric:
                    min_metric = current_metric
                    max_kl = kl_divergence
                    best_w_s = w_s.copy()
                    best_correction_grid = self.correction_grid.copy() if self.correction_grid is not None else None
                    
                    if self.model_A is not None:
                        best_model_A_state = copy.deepcopy(self.model_A.state_dict())
                    if getattr(self, 'gbdt_A', None) is not None:
                        best_gbdt_A = copy.deepcopy(self.gbdt_A)
                    if hasattr(f_sig_cond, 'get_state'):
                        best_sig_state = f_sig_cond.get_state()
                        
                    best_iter = it
            
            if delta_w < tol and it > 5:
                stable_iters += 1
            else:
                stable_iters = 0
                
            if stable_iters >= required_stable_iters:
                logger.info(f"\n--> KINEMATIC STABILIZATION DETECTED.")
                break
                
            w_s_prev = w_s.copy()
            
        logger.info(f"\n--> Restoring optimal physical weights from Iteration {best_iter} (Max KL: {max_kl:.4f}, Chi2/dW: {min_metric:.5f})")
        self.w_s = best_w_s
        if best_correction_grid is not None:
            self.correction_grid = best_correction_grid
            
        if best_model_A_state is not None:
            self.model_A.load_state_dict(best_model_A_state)
        if best_gbdt_A is not None:
            self.gbdt_A = best_gbdt_A
        if hasattr(f_sig_cond, 'load_state') and best_sig_state is not None:
            f_sig_cond.load_state(best_sig_state)
        
        logger.info("Calculating mathematically rigorous N-Dimensional sWeights...")
        return self._calculate_nd_sweights(X_ctrl, y_disc, self.w_s, f_sig_cond, f_bkg_1d)
    
    def _calculate_nd_sweights(self, X_ctrl, y_disc, w_s_final, f_sig_cond, f_bkg_1d, batch_size=10000):
        N_tot = len(y_disc)
        N_s = np.sum(w_s_final)
        N_b = N_tot - N_s
        
        sWeights_MD = np.zeros(N_tot)
        self.model_A.eval()
        
        y_grid_t, dy_weights_t = get_gauss_legendre_grid(self.y_min, self.y_max, n_nodes=self.n_nodes, device=self.device)
        dy_weights_cpu = dy_weights_t.cpu().numpy()
        
        with torch.no_grad():
            for start_idx in range(0, N_tot, batch_size):
                end_idx = min(start_idx + batch_size, N_tot)
                batch_len = end_idx - start_idx
                
                x_b = X_ctrl[start_idx:end_idx]
                y_b = y_disc[start_idx:end_idx]
                z_b = self.ml_transform(x_b) if self.ml_transform else x_b
                
                # =======================================================
                # 1. EVALUATE CONDITIONAL DENSITIES (Observations & Grid)
                # =======================================================
                Z_eval_obs_cpu = np.column_stack([z_b, y_b])
                Z_eval_obs_t = torch.tensor(Z_eval_obs_cpu, dtype=torch.float32, device=self.device)
                fs_obs_raw = f_sig_cond(x_b, y_b)
                fb_base_obs = f_bkg_1d(y_b)
                
                R_dnn_obs = torch.exp(torch.clamp(self.model_A(Z_eval_obs_t), -15, 15)).cpu().numpy().flatten()
                if getattr(self, 'gbdt_A', None) is not None:
                    if self.gbdt_mode == 'XGB_GPU':
                        p_gbdt_raw = self.gbdt_A.predict_proba(Z_eval_obs_t)[:, 1]
                        if isinstance(p_gbdt_raw, torch.Tensor): p_gbdt_raw = p_gbdt_raw.cpu().numpy()
                        R_gbdt_obs = np.clip(p_gbdt_raw, 1e-7, 1-1e-7) / (1.0 - np.clip(p_gbdt_raw, 1e-7, 1-1e-7))
                    elif self.gbdt_mode == 'SKLEARN':
                        p_gbdt = np.clip(self.gbdt_A.predict_proba(Z_eval_obs_cpu)[:, 1], 1e-7, 1-1e-7)
                        R_gbdt_obs = p_gbdt / (1.0 - p_gbdt)
                    else:
                        R_gbdt_obs = np.ones(batch_len)
                else:
                    R_gbdt_obs = np.ones(batch_len)
                    
                correction_interp_obs = np.interp(y_b, self.y_grid_cpu, self.correction_grid) if self.correction_grid is not None else np.ones(batch_len)
                fb_obs_raw = fb_base_obs * R_dnn_obs * R_gbdt_obs * correction_interp_obs
                
                x_b_repeated = np.repeat(x_b, self.n_nodes, axis=0)
                z_b_repeated = np.repeat(z_b, self.n_nodes, axis=0)
                y_tiled = np.tile(self.y_grid_cpu, batch_len)
                
                fs_grid_raw = f_sig_cond(x_b_repeated, y_tiled).reshape(batch_len, self.n_nodes)
                Z_eval_grid_cpu = np.column_stack([z_b_repeated, y_tiled])
                Z_eval_grid_t = torch.tensor(Z_eval_grid_cpu, dtype=torch.float32, device=self.device)
                fb_base_grid = f_bkg_1d(y_tiled)
                
                R_dnn_grid = torch.exp(torch.clamp(self.model_A(Z_eval_grid_t), -15, 15)).cpu().numpy().flatten()
                if getattr(self, 'gbdt_A', None) is not None:
                    if self.gbdt_mode == 'XGB_GPU':
                        p_gbdt_grid = np.clip(self.gbdt_A.predict_proba(Z_eval_grid_t)[:, 1], 1e-7, 1-1e-7)
                        if isinstance(p_gbdt_grid, torch.Tensor): p_gbdt_grid = p_gbdt_grid.cpu().numpy()
                        R_gbdt_grid = p_gbdt_grid / (1.0 - p_gbdt_grid)
                    elif self.gbdt_mode == 'SKLEARN':
                        p_gbdt_grid = np.clip(self.gbdt_A.predict_proba(Z_eval_grid_cpu)[:, 1], 1e-7, 1-1e-7)
                        R_gbdt_grid = p_gbdt_grid / (1.0 - p_gbdt_grid)
                    else:
                        R_gbdt_grid = np.ones(len(y_tiled))
                else:
                    R_gbdt_grid = np.ones(len(y_tiled))
                    
                correction_interp_grid = np.tile(self.correction_grid, batch_len) if self.correction_grid is not None else np.ones(len(y_tiled))
                fb_grid_raw = (fb_base_grid * R_dnn_grid * R_gbdt_grid * correction_interp_grid).reshape(batch_len, self.n_nodes)
                
                # =======================================================
                # 2. STRICT NORMALIZATION
                # =======================================================
                Z_s_ev = np.sum(fs_grid_raw * dy_weights_cpu, axis=1)
                Z_b_ev = np.sum(fb_grid_raw * dy_weights_cpu, axis=1)
                
                fs_cond_obs = fs_obs_raw / np.maximum(Z_s_ev, 1e-12)
                fb_cond_obs = fb_obs_raw / np.maximum(Z_b_ev, 1e-12)
                
                fs_cond_grid = fs_grid_raw / np.maximum(Z_s_ev[:, None], 1e-12)
                fb_cond_grid = fb_grid_raw / np.maximum(Z_b_ev[:, None], 1e-12)
                
                # =======================================================
                # 3. LOCAL COWs VIA EVENT-BY-EVENT QUADRATURE INTEGRATION
                # =======================================================
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
