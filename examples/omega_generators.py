# examples/omega_generators.py
import numpy as np


class OmegaToyGenerator:
    """
    Factory class for generating Omega photoproduction toy data.
    Supports Timing Sidebands, Missing Mass, and toggling 'factorizable' vs 'correlated' modes.
    In 'correlated' (or 'polynomial') mode, the background is an asymmetric, strictly
    non-peaking (convex, d^2 f_b / dy^2 >= 0 everywhere) Bernstein polynomial on [0.70, 0.88]
    whose slope and curvature vary strongly with x = (cos_theta, phi, Phi) and z,
    with a ~3.2:1 lower-to-upper sideband asymmetry inside [0.73, 0.84].
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
        self.base_sigma = 0.0085 
        
        # Missing Mass parameters
        self.mm_peak = 0.938  # Proton mass
        self.mm_sigma_l = 0.012
        self.mm_sigma_r = 0.025 # Radiative tail
        
        # Timing parameters
        self.time_center = 20.0
        self.time_sigma = 1.0

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
                # Smoothly modulated background angular shape with strictly positive support everywhere
                pdf_bkg = (1.2 + 0.8 * c) * (1.0 - 0.6 * np.cos(2 * p))
                accept = np.random.uniform(0, 3.2, n_gen) < pdf_bkg
                
            acc_cos.extend(c[accept])
            acc_phi.extend(p[accept])
            acc_Phi.extend(P[accept])
            
        x_truth = np.column_stack([acc_cos[:n_events], acc_phi[:n_events], acc_Phi[:n_events]])
        
        u = np.random.uniform(0, 1, n_events)
        t_max = 2.0
        norm_t = 1.0 - np.exp(-t_slope * t_max)
        z_truth = - (1.0 / t_slope) * np.log(1.0 - u * norm_t)
        
        return x_truth, z_truth.reshape(-1, 1)

    def _generate_asymmetric_gaussian(self, n_events, peak, sig_l, sig_r):
        """Generates an asymmetric Gaussian (Bifurcated Gaussian)."""
        scale_l = sig_l / (sig_l + sig_r)
        u = np.random.uniform(0, 1, n_events)
        
        left_mask = u < scale_l
        right_mask = ~left_mask
        
        y = np.zeros(n_events)
        y[left_mask] = np.random.normal(peak, sig_l, np.sum(left_mask))
        y[right_mask] = np.random.normal(peak, sig_r, np.sum(right_mask))
        
        y[left_mask] = peak - np.abs(y[left_mask] - peak)
        y[right_mask] = peak + np.abs(y[right_mask] - peak)
        return y

    @staticmethod
    def _sample_nonpeaking_polynomial(x_truth, z_truth, y_min, y_max):
        """
        Samples y from an x-dependent, asymmetric, strictly non-peaking (convex, d^2 f_b / dy^2 >= 0)
        degree-4 Bernstein polynomial on [y_min, y_max] = [0.70, 0.88]:
            f_b(u | x) = p0(x) * 5(1 - u)^4 + p1(x) * 3(1 - u)^2 + p2(x) * (0.40 + 1.20*u) + p3(x) * 1
        where u = (y - y_min) / (y_max - y_min) in [0, 1].
        Because [y_min, y_max] = [0.70, 0.88] extends beyond the [0.73, 0.84] analysis cut window
        (u in [0.167, 0.778]), 5*(1 - u)^4 has moderate density (2.31) at y=0.73 so the 1D signal
        peak at 0.784 is never obscured by a boundary wall, while maintaining a ~3.2:1 lower-to-upper
        sideband asymmetry and strong kinematic non-factorization.
        """
        n_gen = len(x_truth)
        c_th = x_truth[:, 0]
        phi_ang = x_truth[:, 1]
        Phi_ang = x_truth[:, 2]
        z_val = z_truth.flatten()

        # Comp 0: 5*(1 - u)^4 [Beta(1, 5)] -> convex lower-sideband decay at cos_th < 0, sin(phi) > 0
        w0 = (
            1.50 * ((0.5 * (1.0 - c_th)) ** 2) * (0.5 * (1.0 + np.sin(phi_ang)))
            + 0.40 * (0.5 * (1.0 - np.cos(Phi_ang)))
            + 0.25 * np.exp(-z_val)
        )
        # Comp 1: 3*(1 - u)^2 [Beta(1, 3)] -> quadratic downward decay at central cos_th, cos(phi) > 0
        w1 = 0.55 * (1.0 - c_th ** 2) * (0.5 * (1.0 + np.cos(phi_ang)))
        # Comp 2: (0.40 + 1.20*u) [40% Beta(1,1) + 60% Beta(2,1)] -> linear rising tilt (NO u^3 corner spike!)
        w2 = (
            0.95 * ((0.5 * (1.0 + c_th)) ** 2) * (0.5 * (1.0 - np.sin(phi_ang)))
            + 0.30 * (0.5 * (1.0 + np.cos(Phi_ang)))
        )
        # Comp 3: 1 [Beta(1, 1)] -> strictly positive flat floor everywhere
        w3 = np.full(n_gen, 0.30, dtype=np.float64)

        W_Stack = np.column_stack([w0, w1, w2, w3])
        P_Stack = W_Stack / np.sum(W_Stack, axis=1, keepdims=True)
        cdf = np.cumsum(P_Stack, axis=1)

        u_choice = np.random.uniform(0.0, 1.0, size=(n_gen, 1))
        comp_idx = np.argmax(u_choice <= cdf, axis=1)

        u_sample = np.empty(n_gen, dtype=np.float64)
        m0 = comp_idx == 0
        m1 = comp_idx == 1
        m2 = comp_idx == 2
        m3 = comp_idx == 3

        if np.any(m0):
            u_sample[m0] = np.random.beta(1.0, 5.0, size=int(np.sum(m0)))
        if np.any(m1):
            u_sample[m1] = np.random.beta(1.0, 3.0, size=int(np.sum(m1)))
        if np.any(m2):
            n_m2 = int(np.sum(m2))
            is_lin = np.random.uniform(0.0, 1.0, size=n_m2) < 0.60
            u_m2 = np.random.uniform(0.0, 1.0, size=n_m2)
            if np.any(is_lin):
                u_m2[is_lin] = np.random.beta(2.0, 1.0, size=int(np.sum(is_lin)))
            u_sample[m2] = u_m2
        if np.any(m3):
            u_sample[m3] = np.random.uniform(0.0, 1.0, size=int(np.sum(m3)))

        return y_min + (y_max - y_min) * u_sample

    def generate_signal(self, n_events, mode='factorizable', is_data=False, phase_space_cut=False):
        """
        Generates signal events. 
        If phase_space_cut=True, applies a kinematic boundary in the (mass, cos_theta) 
        plane so the distribution does not fill a full rectangle.
        """
        x_acc, z_acc, y_inv_acc, y_mm_acc, t_time_acc = [], [], [], [], []
        t_slope = 3.0 if is_data else 4.0 
        
        while len(x_acc) < n_events:
            n_gen = int((n_events - len(x_acc)) * 1.5)
            if n_gen < 500: n_gen = 500
            
            x_truth, z_truth = self._generate_truth_kinematics(n_gen, is_signal=True, t_slope=t_slope)
            
            if mode == 'factorizable':
                if is_data:
                    y_inv = np.random.normal(self.m_omega + 0.002, self.base_sigma * 1.05, n_gen)
                else:
                    y_inv = np.random.normal(self.m_omega, self.base_sigma, n_gen)
            elif mode == 'correlated':
                sigma_m = 0.006 + 0.005 * (x_truth[:, 0]**2) + 0.003 * z_truth.flatten()
                tail_shift = np.random.exponential(0.002, n_gen) * np.random.choice([-1, 1], n_gen)
                
                if is_data:
                    y_inv = np.random.normal((self.m_omega + 0.003), (sigma_m * 1.05)) + tail_shift
                else:
                    y_inv = np.random.normal(self.m_omega, sigma_m) + tail_shift
            else:
                raise ValueError(f"Unknown signal mode: {mode}")
                
            y_mm = self._generate_asymmetric_gaussian(n_gen, self.mm_peak, self.mm_sigma_l, self.mm_sigma_r)
            t_time = np.random.normal(self.time_center, self.time_sigma, n_gen)
            
            # Apply optional phase-space cut in (cos_theta, mass) on one side
            if phase_space_cut:
                cos_th = x_truth[:, 0]
                # Example boundary: restricts low-mass events at negative cos_theta (or similar edge)
                # Keeping events where cos_theta > threshold(mass)
                valid_mask = cos_th > (-1 + 8.5 * (y_inv - 0.73))
                x_truth = x_truth[valid_mask]
                z_truth = z_truth[valid_mask]
                y_inv = y_inv[valid_mask]
                y_mm = y_mm[valid_mask]
                t_time = t_time[valid_mask]

            x_acc.extend(x_truth)
            z_acc.extend(z_truth)
            y_inv_acc.extend(y_inv)
            y_mm_acc.extend(y_mm)
            t_time_acc.extend(t_time)
            
        return (
            np.array(x_acc[:n_events]), np.array(z_acc[:n_events]), 
            np.array(y_inv_acc[:n_events]), np.array(y_mm_acc[:n_events]), np.array(t_time_acc[:n_events])
        )

    def generate_background(self, n_events, bkg_type='combinatorial', mode='correlated', phase_space_cut=False):
        """
        Generates background events with an optional phase-space cut in (mass, cos_theta).
        """
        x_acc, z_acc, y_inv_acc, y_mm_acc, t_time_acc = [], [], [], [], []
        y_min, y_max = 0.70, 0.88
        
        while len(x_acc) < n_events:
            n_gen = int((n_events - len(x_acc)) * 1.5)
            if n_gen < 500: n_gen = 500
            
            x_truth, z_truth = self._generate_truth_kinematics(n_gen, is_signal=False, t_slope=1.5)
            
            if bkg_type == 'combinatorial':
                if mode == 'factorizable':
                    y_inv = np.random.uniform(y_min, y_max, n_gen)
                elif mode in ('correlated', 'polynomial'):
                    y_inv = self._sample_nonpeaking_polynomial(x_truth, z_truth, y_min, y_max)
                else:
                    raise ValueError(f"Unknown background mode: {mode}")
                    
                y_mm = self._generate_asymmetric_gaussian(n_gen, self.mm_peak, self.mm_sigma_l, self.mm_sigma_r)
                t_time = np.random.normal(self.time_center, self.time_sigma, n_gen)
                
            elif bkg_type == 'non_exclusive':
                if mode == 'factorizable':
                    y_inv = np.random.normal(self.m_omega, self.base_sigma, n_gen)
                    y_mm = np.random.uniform(0.70, 1.20, n_gen)
                elif mode in ('correlated', 'polynomial'):
                    sigma_m = 0.006 + 0.005 * (x_truth[:, 0]**2) + 0.003 * z_truth.flatten()
                    y_inv = np.random.normal(self.m_omega, sigma_m)
                    y_mm = self._sample_nonpeaking_polynomial(x_truth, z_truth, 0.75, 1.15)
                else:
                    raise ValueError(f"Unknown background mode: {mode}")
                    
                t_time = np.random.normal(self.time_center, self.time_sigma, n_gen)
                
            elif bkg_type == 'out_of_time':
                y_inv = np.random.uniform(y_min, y_max, n_gen)
                y_mm = np.random.uniform(0.70, 1.20, n_gen)
                t_time = np.random.uniform(0.0, 40.0, n_gen)
                
            else:
                raise ValueError(f"Unknown bkg_type: {bkg_type}")

            # Apply optional phase-space cut in (cos_theta, mass) on one side
            if phase_space_cut:
                cos_th = x_truth[:, 0]
                valid_mask = cos_th > (-1 + 8.5 * (y_inv - 0.73))
                x_truth = x_truth[valid_mask]
                z_truth = z_truth[valid_mask]
                y_inv = y_inv[valid_mask]
                y_mm = y_mm[valid_mask]
                t_time = t_time[valid_mask]

            x_acc.extend(x_truth)
            z_acc.extend(z_truth)
            y_inv_acc.extend(y_inv)
            y_mm_acc.extend(y_mm)
            t_time_acc.extend(t_time)

        return (
            np.array(x_acc[:n_events]), np.array(z_acc[:n_events]), 
            np.array(y_inv_acc[:n_events]), np.array(y_mm_acc[:n_events]), np.array(t_time_acc[:n_events])
        )
