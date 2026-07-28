import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.special
from scipy.stats import kendalltau
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL

from core import MLSWeightExtractor, build_signal_flow, calculate_1d_sweights
from utils.diagnostics import (plot_stacked_1d, plot_final_1d_projections, 
                               plot_2d_correlations, plot_1d_y_fit, plot_sliced_y_grids, get_chi2)

# ====================================================================
# 1. PURE TRUTH PHYSICS GENERATOR (Strict Exponential Background)
# ====================================================================
class OmegaExpToyGenerator:
    def __init__(self):
        self.SDME = {'r0_00': 0.1, 'r0_1-1': 0.4, 'Re_r0_10': 0.05, 
                     'r1_11': 0.4, 'r1_00': -0.05, 'r1_1-1': 0.4, 
                     'Re_r1_10': 0.02, 'r2_1-1': 0.4, 'Im_r2_10': 0.02}
        self.P_gamma = 0.85
        self.m_omega = 0.782
        self.base_sigma = 0.010 

    def _schilling_wolf_pdf(self, cos_theta, phi, Phi):
        sin_theta = np.sqrt(np.maximum(1 - cos_theta**2, 0))
        sin_2theta = 2 * sin_theta * cos_theta
        W0 = 0.5*(1 - self.SDME['r0_00']) + 0.5*(3*self.SDME['r0_00'] - 1)*cos_theta**2 \
             - np.sqrt(2)*self.SDME['Re_r0_10']*sin_2theta*np.cos(phi) - self.SDME['r0_1-1']*sin_theta**2*np.cos(2*phi)
        W1 = self.SDME['r1_11']*sin_theta**2 + self.SDME['r1_00']*cos_theta**2 \
             - np.sqrt(2)*self.SDME['Re_r1_10']*sin_2theta*np.cos(phi) - self.SDME['r1_1-1']*sin_theta**2*np.cos(2*phi)
        W2 = np.sqrt(2)*self.SDME['Im_r2_10']*sin_2theta*np.sin(phi) + self.SDME['r2_1-1']*sin_theta**2*np.sin(2*phi)
        W = (3/(4*np.pi)) * (W0 - self.P_gamma*np.cos(2*Phi)*W1 - self.P_gamma*np.sin(2*Phi)*W2)
        return np.maximum(W, 1e-6)

    def generate_signal(self, n_events):
        acc_cos, acc_phi, acc_Phi = [], [], []
        while len(acc_cos) < n_events:
            n_gen = int((n_events - len(acc_cos)) * 4.0)
            c = np.random.uniform(-1, 1, n_gen)
            p = np.random.uniform(-np.pi, np.pi, n_gen)
            P = np.random.uniform(-np.pi, np.pi, n_gen)
            accept = np.random.uniform(0, 0.6, n_gen) < self._schilling_wolf_pdf(c, p, P)
            acc_cos.extend(c[accept]); acc_phi.extend(p[accept]); acc_Phi.extend(P[accept])
            
        x_truth = np.column_stack([acc_cos[:n_events], acc_phi[:n_events], acc_Phi[:n_events]])
        
        # Pure factorizable mass (Gaussian)
        y_true = np.random.normal(self.m_omega, self.base_sigma, n_events)
        
        # Exponential momentum transfer (z)
        u = np.random.uniform(0, 1, n_events)
        norm_t = 1.0 - np.exp(-3.0 * 2.0)
        z_truth = - (1.0 / 3.0) * np.log(1.0 - u * norm_t)
        
        return x_truth, z_truth.reshape(-1, 1), y_true

    def generate_background(self, n_events, y_min):
        acc_cos, acc_phi, acc_Phi = [], [], []
        while len(acc_cos) < n_events:
            n_gen = int((n_events - len(acc_cos)) * 1.5)
            c = np.random.uniform(-1, 1, n_gen)
            p = np.random.uniform(-np.pi, np.pi, n_gen)
            P = np.random.uniform(-np.pi, np.pi, n_gen)
            
            # Acceptance shaping the kinematics
            pdf_bkg = np.maximum((1.2 + c) * (1.0 - 0.8 * np.cos(2 * p)) - 0.9, 0.0)
            accept = np.random.uniform(0, 3.0, n_gen) < pdf_bkg
            acc_cos.extend(c[accept]); acc_phi.extend(p[accept]); acc_Phi.extend(P[accept])
            
        x_truth = np.column_stack([acc_cos[:n_events], acc_phi[:n_events], acc_Phi[:n_events]])
        
        # Pure factorizable exponential mass (unlinked from kinematics)
        alpha = 15.0
        u_mass = np.random.uniform(0, 1, n_events)
        norm_const = 1.0 - np.exp(-alpha * 0.16)
        y_true = y_min - (1.0 / alpha) * np.log(1.0 - u_mass * norm_const)
        
        # Flat t-distribution for background
        z_truth = np.random.uniform(0, 2.0, n_events)
        
        return x_truth, z_truth.reshape(-1, 1), y_true

print("1. Generating Factorizable Exponential Toy Data...")
factory = OmegaExpToyGenerator()
Y_MIN, Y_MAX = 0.73, 0.85

x_mc, z_mc, y_mc = factory.generate_signal(100000)
x_sig, z_sig, y_sig = factory.generate_signal(10000)
x_bkg, z_bkg, y_bkg = factory.generate_background(50000, Y_MIN)

x_data_full = np.vstack([x_sig, x_bkg])
z_data_full = np.concatenate([z_sig, z_bkg])
y_data_full = np.concatenate([y_sig, y_bkg])

mask = (y_data_full >= Y_MIN) & (y_data_full <= Y_MAX)
x_data, z_data, y_data = x_data_full[mask], z_data_full[mask], y_data_full[mask]
x_sig_true = x_sig[(y_sig >= Y_MIN) & (y_sig <= Y_MAX)]
x_bkg_true = x_bkg[(y_bkg >= Y_MIN) & (y_bkg <= Y_MAX)]
N_tot = len(y_data)

def map_topology(x):
    cT, phi, Phi = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    sT = np.sqrt(np.maximum(1 - cT**2, 0.0))
    return np.hstack([cT, sT, np.cos(phi), np.sin(phi), np.cos(Phi), np.sin(Phi)])

X_ctrl_data = np.hstack([map_topology(x_data), z_data])
Z_ctx_data = np.column_stack([x_data[:, :3], z_data]) 
Z_ctx_mc = np.column_stack([x_mc[:, :3], z_mc])

# ====================================================================
# 1.5 FACTORIZATION TEST
# ====================================================================
print("\n--- Factorization Check (Kendall's Tau) ---")
sideband_mask = (y_data < 0.76) | (y_data > 0.81)
tau_stat_cos, p_value_cos = kendalltau(y_data[sideband_mask], x_data[sideband_mask, 0])
tau_stat_phi, p_value_phi = kendalltau(y_data[sideband_mask], x_data[sideband_mask, 1])

print(f"Mass vs cos(theta) -> Tau: {tau_stat_cos:.4f}, p-value: {p_value_cos:.2e}")
print(f"Mass vs phi        -> Tau: {tau_stat_phi:.4f}, p-value: {p_value_phi:.2e}")
print("-------------------------------------------\n")

# ====================================================================
# 2. HYBRID 1D ANCHOR FIT (Exponential * Bernstein)
# ====================================================================
print("Executing Hybrid 1D Anchor Fit (Exponential * Bernstein)...")

def trunc_exp_pdf(y, y_min, y_max, lam):
    # Strictly clamp evaluation to bounds to prevent out-of-domain negative densities
    y_c = np.clip(y, y_min, y_max)
    if abs(lam) < 1e-5: return np.ones_like(y_c) / (y_max - y_min)
    num = lam * np.exp(-lam * y_c)
    den = np.exp(-lam * y_min) - np.exp(-lam * y_max)
    return num / den

def bernstein_poly(y, weights, y_min, y_max):
    # Strictly clamp evaluation to prevent y_norm < 0 (which causes negative polynomials)
    y_c = np.clip(y, y_min, y_max)
    y_norm = (y_c - y_min) / (y_max - y_min)
    n = len(weights) - 1
    poly = np.zeros_like(y_norm)
    for i, w in enumerate(weights):
        poly += w * scipy.special.binom(n, i) * (y_norm**i) * ((1 - y_norm)**(n - i))
    return poly

def trunc_norm_pdf(y, mu, sigma, y_min, y_max):
    from scipy.stats import truncnorm
    a, b = (y_min - mu) / sigma, (y_max - mu) / sigma
    return truncnorm.pdf(y, a, b, loc=mu, scale=sigma)

# Pre-compute a normalization grid for speed
y_grid_norm = np.linspace(Y_MIN, Y_MAX, 500)
dy_norm = y_grid_norm[1] - y_grid_norm[0]

def hybrid_bkg_pdf(y, lam, w0, w1, w2, y_min, y_max):
    weights = [1.0, w0, w1, w2]  # Fix first weight to 1.0 to break scaling degeneracy
    raw_pdf = trunc_exp_pdf(y, y_min, y_max, lam) * bernstein_poly(y, weights, y_min, y_max)
    
    # Fast Riemann normalization
    grid_pdf = trunc_exp_pdf(y_grid_norm, y_min, y_max, lam) * bernstein_poly(y_grid_norm, weights, y_min, y_max)
    integral = np.sum(grid_pdf) * dy_norm
    return raw_pdf / integral

def nll_hybrid(f_sig, mu, sigma, lam, w0, w1, w2):
    fs = trunc_norm_pdf(y_data, mu, sigma, Y_MIN, Y_MAX)
    fb = hybrid_bkg_pdf(y_data, lam, w0, w1, w2, Y_MIN, Y_MAX)
    pdf = f_sig * fs + (1 - f_sig) * fb
    pdf = np.where(pdf > 1e-10, pdf, 1e-10)
    return -np.sum(np.log(pdf))

# Give the minimizer a good starting guess for an exponential decay
m_anchor = Minuit(nll_hybrid, f_sig=0.2, mu=0.782, sigma=0.010, lam=15.0, w0=1.0, w1=1.0, w2=1.0)
m_anchor.limits = [ (0.01, 0.99), (0.76, 0.80), (0.005, 0.03), (0.0, 50.0), (0.1, 10.0), (0.1, 10.0), (0.1, 10.0) ]
m_anchor.migrad()

print(f"  --> Extracted Data Peak: {m_anchor.values['mu']:.5f}")
print(f"  --> Extracted Decay Lambda: {m_anchor.values['lam']:.2f}")

N_s_fit = N_tot * m_anchor.values['f_sig']
N_b_fit = N_tot * (1 - m_anchor.values['f_sig'])

fs_1d_eval = trunc_norm_pdf(y_data, m_anchor.values['mu'], m_anchor.values['sigma'], Y_MIN, Y_MAX)
fb_1d_eval = hybrid_bkg_pdf(y_data, m_anchor.values['lam'], m_anchor.values['w0'], m_anchor.values['w1'], m_anchor.values['w2'], Y_MIN, Y_MAX)

w_splot_1d = calculate_1d_sweights(N_s_fit, N_b_fit, fs_1d_eval, fb_1d_eval)

bounds_list = [(-1.0, 1.0), (-np.pi, np.pi), (-np.pi, np.pi)]
labels_list = [r"$\cos\theta$", r"$\phi$", r"$\Phi$"]

chi2s_1d = get_chi2(x_data, w_splot_1d, x_sig_true, bounds_list)
chi2_str_1d = ", ".join([f"{c:.1f}" for c in chi2s_1d])
print(f"--> Baseline 1D sPlot Chi2 (cos_theta, phi, Phi): [{chi2_str_1d}]\n")

# Package Callables for the Extractor
def final_bkg_callable(y):
    return hybrid_bkg_pdf(y, m_anchor.values['lam'], m_anchor.values['w0'], m_anchor.values['w1'], m_anchor.values['w2'], Y_MIN, Y_MAX)

# Standard Rejection Sampler for GBDT
def bkg_sampler(n_samples):
    samples = []
    max_pdf = np.max(final_bkg_callable(y_grid_norm)) * 1.2
    while len(samples) < n_samples:
        batch = int((n_samples - len(samples)) * 2)
        y_cands = np.random.uniform(Y_MIN, Y_MAX, batch)
        u = np.random.uniform(0, max_pdf, batch)
        accept = u < final_bkg_callable(y_cands)
        samples.extend(y_cands[accept])
    return np.array(samples[:n_samples])

# ====================================================================
# 3. NORMALIZING FLOW TRAINING & INITIALIZATION
# ====================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Flow is initialized directly at the 1D anchor parameters.
sig_pdf_Z = build_signal_flow(
    Z_ctx_mc, y_mc, 
    factory.m_omega, factory.base_sigma, m_anchor.values['mu'], device=device
)

def user_signal_pdf(X_ctrl, y_array):
    cT, cp, sp, cP, sP, z = X_ctrl[:, 0], X_ctrl[:, 2], X_ctrl[:, 3], X_ctrl[:, 4], X_ctrl[:, 5], X_ctrl[:, 6]
    phi, np_Phi = np.arctan2(sp, cp), np.arctan2(sP, cP)
    Z_ctx = np.column_stack([cT, phi, np_Phi, z])
    return sig_pdf_Z(Z_ctx, y_array)
# ====================================================================
# 4. MULTI-DIMENSIONAL SWEIGHT EXTRACTION
# ====================================================================
plt.ion()
plot_1d_y_fit(y_data, lambda y: trunc_norm_pdf(y, m_anchor.values['mu'], m_anchor.values['sigma'], Y_MIN, Y_MAX), final_bkg_callable, m_anchor.values['f_sig'], Y_MIN, Y_MAX)

val_dict = {
    'x_raw': x_data, 
    'x_true': x_sig_true, 
    'bounds': bounds_list, 
    'labels': labels_list, 
    'y_raw': y_data, 
    'y_bounds': (Y_MIN, Y_MAX), 
    'plot_iterations': True
}

# Bayes Probabilities for Iteration 0 Initialization
w_s_init = (N_s_fit * fs_1d_eval) / np.maximum(N_s_fit * fs_1d_eval + N_b_fit * fb_1d_eval, 1e-12)

plot_sliced_y_grids(x_raw=x_data, X_ctrl=X_ctrl_data, y_data=y_data, w_sig=w_s_init, 
                    bounds_list=bounds_list, f_sig_cond=user_signal_pdf, f_bkg_1d=final_bkg_callable, 
                    Y_MIN=Y_MIN, Y_MAX=Y_MAX, device=device, is_iter_0=True, iter_label="Iter 0 Anchor")

extractor = MLSWeightExtractor(
    disc_bounds=(Y_MIN, Y_MAX), 
    n_nodes=40, 
    gbdt_mode='XGB_GPU',
    device=device,
    marginal_alpha=0.5,
    dynamic_alpha=True,
    use_gbdt=True,            # Disabled for purely factorizable check
    reinstate_nn=True,
    dnn_epochs=100,
    ml_transform=False,
    loss_fn='BCE',           
    logit_l2_penalty=0.0001      # High penalty suppresses noise-fitting 
)

sWeights_MD = extractor.fit_transform(
    X_ctrl=X_ctrl_data, 
    y_disc=y_data, 
    w_s_init=w_s_init, 
    f_sig_cond=user_signal_pdf, 
    f_bkg_1d=final_bkg_callable, 
    bkg_sampler=bkg_sampler,
    max_iters=40,
    validation_data=val_dict,
    omega_init=2.0,
    omega_decay_iters=20.0
)

# ====================================================================
# 5. FINAL DIAGNOSTICS & PLOTTING
# ====================================================================
plt.ioff()

plot_sliced_y_grids(x_raw=x_data, X_ctrl=X_ctrl_data, y_data=y_data, w_sig=sWeights_MD, 
                    bounds_list=bounds_list, f_sig_cond=user_signal_pdf, f_bkg_1d=final_bkg_callable, 
                    Y_MIN=Y_MIN, Y_MAX=Y_MAX, device=device, is_iter_0=False, iter_label="Final Extraction", extractor_obj=extractor)

plot_stacked_1d(x_data, sWeights_MD, bounds_list, labels_list)
plot_final_1d_projections(x_data, x_sig_true, x_bkg_true, w_splot_1d, sWeights_MD, bounds_list, labels_list, w_prob=extractor.w_s)

axes_pairs = [(0, 1, r"$\cos\theta$", r"$\phi$"), (0, 2, r"$\cos\theta$", r"$\Phi$"), (1, 2, r"$\phi$", r"$\Phi$")]
plot_2d_correlations(x_data, x_sig_true, w_splot_1d, sWeights_MD, axes_pairs, bounds_list, is_signal=True, w_prob=extractor.w_s)
plot_2d_correlations(x_data, x_bkg_true, (1-w_splot_1d), (1-sWeights_MD), axes_pairs, bounds_list, is_signal=False, w_prob=(1-extractor.w_s))

plt.show()
