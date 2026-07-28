
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import kendalltau, norm as scipy_norm, expon
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL, ExtendedBinnedNLL

# sweights imports
from sweights.testing import make_classic_toy
from sweights.util import make_bernstein_pdf
from sweights.experimental import Cows

# Core ML Imports
from core import MLSWeightExtractor, fit_1d_anchor

# ====================================================================
# 1. GENERATE NON-FACTORIZING TOY DATA
# ====================================================================
print("1. Generating Non-Factorizing Toy Data...")
mrange = (0.0, 1.0)
trange = (0.0, 0.5)

toy = make_classic_toy(
    1, s=40000, b=100000, mrange=mrange, trange=trange, 
    mb_mu=lambda t: 0.1 + 10 * (t / 1.5)
)

y_data = toy[0]  
t_data = toy[1]  
is_signal = toy[2]  

X_ctrl_data = t_data.reshape(-1, 1)

# ====================================================================
# 2. FACTORIZATION CHECK (Kendall's Tau)
# ====================================================================
sideband_mask = (y_data < 0.2) | (y_data > 0.8)
tau_stat, p_value = kendalltau(y_data[sideband_mask], t_data[sideband_mask])
print(f"\n2. Factorization Check -> Mass vs 't' Tau: {tau_stat:.4f}, p-value: {p_value:.2e}")

# ====================================================================
# 3. STANDARD 1D SPLOT & ML ANCHOR FIT
# ====================================================================
print("\n3. Fitting 1D Anchor (Standard sPlot Baseline)...")
anchor = fit_1d_anchor(y_data, mrange[0], mrange[1], n_poly=5)
sWeights_splot = anchor['w_s_init']

def user_signal_pdf(X_ctrl, y_array):
    return anchor['trunc_norm_pdf'](y_array, anchor['y_peak'], anchor['y_sigma'])

# ====================================================================
# 4. COWs (Custom Orthogonal Weight functions) EXTRACTION
# ====================================================================
print("\n4. Executing COWs Extraction...")
bern = make_bernstein_pdf(2, *mrange)

def m_density(x, s, b1, b2, b3, mu, sigma):
    ds = scipy_norm(mu, sigma)
    snorm = np.diff(ds.cdf(mrange))[0]
    pdf_val = s / snorm * ds.pdf(x) + b1 * bern[0](x) + b2 * bern[1](x) + b3 * bern[2](x)
    return s + b1 + b2 + b3, pdf_val

cost = ExtendedUnbinnedNLL(y_data, m_density)
mi = Minuit(cost, s=4000, b1=3333, b2=3333, b3=3333, mu=0.5, sigma=0.05)
mi.limits["s", "b1", "b2", "b3", "sigma"] = (0, None)
mi.migrad()

def spdf(x):
    ds = scipy_norm(mi.values['mu'], mi.values['sigma'])
    snorm = np.diff(ds.cdf(mrange))[0]
    return ds.pdf(x) / snorm

cows_obj = Cows(y_data, spdf, bern, range=mrange)
sWeights_COWs = cows_obj(y_data)

# ====================================================================
# 5. ML-SWEIGHT EXTRACTION
# ====================================================================
print("\n5. Executing Multi-Dimensional ML sWeight Extraction...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- THE FIX: Define the Logit Map for ML feature scaling ---
def logit_transform(x):
    # Safely scale physical t [0, 0.5] into an infinite manifold [-inf, inf]
    x_scaled = x / trange[1]
    x_clipped = np.clip(x_scaled, 1e-6, 1.0 - 1e-6)
    return np.log(x_clipped / (1.0 - x_clipped))

val_dict = {
    'x_raw': X_ctrl_data, 
    'x_true': X_ctrl_data[is_signal], 
    'bounds': [(trange[0], trange[1])], 
    'labels': ["Kinematic Var (t)"], 
    'y_raw': y_data,
    'y_bounds': mrange,
    'plot_iterations': True
}

extractor = MLSWeightExtractor(
    disc_bounds=mrange, 
    n_nodes=20, 
    gbdt_mode='XGB_GPU', 
    device=device,
    reinstate_nn=True,
    dnn_epochs=100,
    marginal_alpha=0.,
    dynamic_alpha=True,
    use_gbdt=True,
    ml_transform=False
)

sWeights_ML = extractor.fit_transform(
    X_ctrl=X_ctrl_data,          # <-- Safely pass the PHYSICAL data
    y_disc=y_data, 
    w_s_init=sWeights_splot, 
    f_sig_cond=user_signal_pdf, 
    f_bkg_1d=anchor['bkg_pdf'], 
    bkg_sampler=anchor['bkg_sampler'],
    max_iters=40,
    validation_data=val_dict,
    omega_init=1.5,
    omega_decay_iters=20
)

# ====================================================================
# 6. QUANTITATIVE FITS (Minuit Extended Binned NLL)
# ====================================================================
print("\n6. Performing Minuit Fits on Extracted 't' Distributions...")
results = [
    ("Standard sPlot", sWeights_splot),
    ("COWs", sWeights_COWs),
    ("MLSWeights", sWeights_ML)
]

for method, w in results:
    val, xe = np.histogram(t_data, bins=40, range=trange, weights=w)
    var = np.histogram(t_data, bins=xe, weights=w**2)[0]

    data = np.transpose((val, var))
    cost = ExtendedBinnedNLL(data, xe, lambda x, n, slope: n * expon.cdf(x, 0, slope))
    tmi = Minuit(cost, n=np.sum(w), slope=0.1)
    tmi.limits["n", "slope"] = (0.01, None)
    tmi.migrad()
    tmi.hesse()
    
    val_slope = tmi.values["slope"]
    err_slope = tmi.errors["slope"]
    print(f"{method:15}: slope = {val_slope:.4f} +/- {err_slope:.4f} "
          f"(chi2/ndof = {tmi.fmin.reduced_chi2:.1f})")

# ====================================================================
# 7. DIAGNOSTIC COMPARISON PLOTS
# ====================================================================
print("\n7. Plotting Final Comparisons...")

plt.ioff()
plt.figure(figsize=(15, 5))
plt.suptitle("Factorization Breaking: Extracted Kinematic Variable (t) Distributions", fontsize=16)

bins = np.linspace(trange[0], trange[1], 40)

# 1. Standard sPlot
plt.subplot(1, 3, 1)
plt.hist(t_data[is_signal], bins=bins, color='black', histtype='step', lw=2, label="True Signal")
plt.hist(t_data, bins=bins, weights=sWeights_splot, color='red', histtype='step', ls='--', lw=2, label="Standard sPlot")
plt.title("Standard 1D sPlot")
plt.xlabel("Kinematic Variable (t)")
plt.legend()

# 2. COWs
plt.subplot(1, 3, 2)
plt.hist(t_data[is_signal], bins=bins, color='black', histtype='step', lw=2, label="True Signal")
plt.hist(t_data, bins=bins, weights=sWeights_COWs, color='green', histtype='step', ls='--', lw=2, label="COWs (Non-Factorizing)")
plt.title("COWs Extraction")
plt.xlabel("Kinematic Variable (t)")
plt.legend()

# 3. MLSWeights
plt.subplot(1, 3, 3)
plt.hist(t_data[is_signal], bins=bins, color='black', histtype='step', lw=2, label="True Signal")
plt.hist(t_data, bins=bins, weights=sWeights_ML, color='blue', histtype='step', ls='--', lw=2, label="ML sWeights")
plt.title("MLSWeightExtractor")
plt.xlabel("Kinematic Variable (t)")
plt.legend()

plt.tight_layout()
plt.show()
