import numpy as np

class OmegaToyGenerator:
    """
    Factory class for generating Omega photoproduction toy data.
    Strictly separates pure truth-level kinematics from detector-level distortions.
    """
    def __init__(self):
        # Base Physics Constants
        self.SDME = {
            'r0_00': 0.1, 'r0_1-1': 0.4, 'Re_r0_10': 0.05, 
            'r1_11': 0.4, 'r1_00': -0.05, 'r1_1-1': 0.4, 
            'Re_r1_10': 0.02, 'r2_1-1': 0.4, 'Im_r2_10': 0.02
        }
        self.P_gamma = 0.85
        self.m_omega = 0.782
        self.base_sigma = 0.010 

    # ====================================================================
    # 1. PURE TRUTH PHYSICS (Reusable Core)
    # ====================================================================
    def _schilling_wolf_pdf(self, cos_theta, phi, Phi):
        """Analytical SDME angular distribution."""
        sin_theta = np.sqrt(np.maximum(1 - cos_theta**2, 0))
        sin_2theta = 2 * sin_theta * cos_theta
        
        W0 = 0.5*(1 - self.SDME['r0_00']) + 0.5*(3*self.SDME['r0_00'] - 1)*cos_theta**2 \
             - np.sqrt(2)*self.SDME['Re_r0_10']*sin_2theta*np.cos(phi) - self.SDME['r0_1-1']*sin_theta**2*np.cos(2*phi)
        W1 = self.SDME['r1_11']*sin_theta**2 + self.SDME['r1_00']*cos_theta**2 \
             - np.sqrt(2)*self.SDME['Re_r1_10']*sin_2theta*np.cos(phi) - self.SDME['r1_1-1']*sin_theta**2*np.cos(2*phi)
        W2 = np.sqrt(2)*self.SDME['Im_r2_10']*sin_2theta*np.sin(phi) + self.SDME['r2_1-1']*sin_theta**2*np.sin(2*phi)
        
        W = (3/(4*np.pi)) * (W0 - self.P_gamma*np.cos(2*Phi)*W1 - self.P_gamma*np.sin(2*Phi)*W2)
        return np.maximum(W, 1e-6)

    def _generate_truth_kinematics(self, n_events, is_signal, t_slope):
        """Generates pure truth topological angles (x) and momentum transfer (z)."""
        acc_cos, acc_phi, acc_Phi = [], [], []
        
        while len(acc_cos) < n_events:
            n_gen = int((n_events - len(acc_cos)) * 4.0)
            c = np.random.uniform(-1, 1, n_gen)
            p = np.random.uniform(-np.pi, np.pi, n_gen)
            P = np.random.uniform(-np.pi, np.pi, n_gen)
            
            if is_signal:
                accept = np.random.uniform(0, 0.6, n_gen) < self._schilling_wolf_pdf(c, p, P)
            else:
                # SEVERE BACKGROUND: Carving out massive PDF == 0 valleys
                pdf_bkg = (1.2 + c) * (1.0 - 0.8 * np.cos(2 * p)) - 0.9
                pdf_bkg = np.maximum(pdf_bkg, 0.0)
                accept = np.random.uniform(0, 3.0, n_gen) < pdf_bkg
                
            acc_cos.extend(c[accept])
            acc_phi.extend(p[accept])
            acc_Phi.extend(P[accept])
            
        x_truth = np.column_stack([acc_cos[:n_events], acc_phi[:n_events], acc_Phi[:n_events]])
        
        # Exponential momentum transfer via inverse transform sampling
        u = np.random.uniform(0, 1, n_events)
        t_max = 2.0
        norm_t = 1.0 - np.exp(-t_slope * t_max)
        z_truth = - (1.0 / t_slope) * np.log(1.0 - u * norm_t)
        
        return x_truth, z_truth.reshape(-1, 1)

    # ====================================================================
    # 2. DETECTOR DISTORTIONS (Mode Selection)
    # ====================================================================
    def generate_signal(self, n_events, mode='correlated', is_data=False):
        """
        Generates signal events. 
        Mode 'factorizable': Fixed resolution, standard sPlot works perfectly.
        Mode 'correlated': Resolution depends on kinematics.
        """
        t_slope = 3.0 if is_data else 4.0 
        x, z = self._generate_truth_kinematics(n_events, is_signal=True, t_slope=t_slope)
        
        if mode == 'factorizable':
            # Pure Gaussian perfectly matching trunc_norm_pdf
            y_true = np.random.normal(self.m_omega, self.base_sigma, n_events)
            
        elif mode == 'correlated':
            # Dynamic resolution smearing based on cos(theta) and t
            sigma_m = 0.006 + 0.005 * (x[:, 0]**2) + 0.003 * z.flatten()
            tail_shift = np.random.exponential(0.002, n_events) * np.random.choice([-1, 1], n_events)
            
            if is_data:
                # Systemic data-only miscalibrations
                shift_mev = 0.003  
                smear_pct = 1.05   
                y_true = np.random.normal((self.m_omega + shift_mev), (sigma_m * smear_pct)) + tail_shift
            else:
                y_true = np.random.normal(self.m_omega, sigma_m) + tail_shift
        else:
            raise ValueError(f"Unknown signal mode: {mode}")
            
        return x, z, y_true

    def generate_background(self, n_events, mode='correlated'):
        """
        Generates background events.
        Mode 'factorizable': Flat uniform mass perfectly matching Bernstein polynomial fit.
        Mode 'correlated': Mass shape and thresholds depend on kinematics.
        """
        x_acc, z_acc, y_acc = [], [] , []
        alpha = 12.0 if mode == 'correlated' else 15.0
        
        while len(x_acc) < n_events:
            n_gen = int((n_events - len(x_acc)) * 1.5)
            x_truth, z_truth = self._generate_truth_kinematics(n_gen, is_signal=False, t_slope=1.5)
            
            u_mass = np.random.uniform(0, 1, n_gen)
            
            if mode == 'factorizable':
                # Perfectly uniform background mass, completely unlinked from kinematics.
                # This guarantees the 1D Bernstein polynomial fits it flawlessly.
                y_cand = np.random.uniform(0.73, 0.85, n_gen)
                valid = np.ones(n_gen, dtype=bool)
                
            elif mode == 'correlated':
                norm_const = 1.0 - np.exp(-alpha * 1.30)
                y_base = 0.20 - (1.0 / alpha) * np.log(1.0 - u_mass * norm_const)
                
                # Kinematic reflection pushing mass peak
                y_cand = y_base + 0.05 * x_truth[:, 0] * np.sin(x_truth[:, 1])
                
                # Geometric acceptance threshold dependent on cos(theta)
                threshold = 0.70 + 0.05 * x_truth[:, 0]
                valid = y_cand > threshold
            else:
                raise ValueError(f"Unknown background mode: {mode}")

            x_acc.extend(x_truth[valid])
            z_acc.extend(z_truth[valid])
            y_acc.extend(y_cand[valid])

        return np.array(x_acc[:n_events]), np.array(z_acc[:n_events]), np.array(y_acc[:n_events])
