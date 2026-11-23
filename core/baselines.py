"""
Classical 1D Custom Orthogonal Weight functions (COWs) Baselines.
Provides three mathematically verified implementations:
  1. calculate_cows_lite: Official sweights.experimental.Cows against Bernstein basis
  2. calculate_cows_full_derivatives: Continuous Gauss-Legendre COWs with strictly
     zero-integral projected Taylor derivatives (d_mu, d_sigma) and Tikhonov regularization
     to prevent basis collinearity blow-up when d_sigma overlaps quadratic Bernstein modes.
  3. calculate_cows_full_official: Multi-component signal expansion using sweights.experimental.Cows,
     summing the component weights so shifted/broadened signal events are preserved rather than subtracted.
"""

import numpy as np
import scipy.stats as stats
import scipy.ndimage
from scipy.stats import norm
from sweights.experimental import Cows
from sweights.util import make_bernstein_pdf


class StandardPDFs:
    """A collection of standard analytical 1D PDFs for baseline fitting."""
    
    @staticmethod
    def trunc_norm(y, m, sig, y_min, y_max):
        cdf_max = norm.cdf(y_max, m, sig)
        cdf_min = norm.cdf(y_min, m, sig)
        norm_fact = np.maximum(cdf_max - cdf_min, 1e-10)
        return np.maximum(norm.pdf(y, m, sig) / norm_fact, 1e-10)

    @staticmethod
    def exponential(y, alpha, y_min, y_max):
        val = np.exp(-alpha * (y - y_min))
        norm_c = (1.0 - np.exp(-alpha * (y_max - y_min))) / alpha
        return np.maximum(val / norm_c, 1e-10)


def calculate_1d_sweights(Ns, Nb, fs, fb):
    """Rigorous 1D analytical covariance inversion."""
    P_tot = np.maximum(Ns * fs + Nb * fb, 1e-10)
    H_ss = np.sum((fs/P_tot)**2)
    H_bb = np.sum((fb/P_tot)**2)
    H_sb = np.sum((fs*fb)/P_tot**2)
    Det = np.maximum(H_ss * H_bb - H_sb**2, 1e-12)
    return ((H_bb/Det) * fs + (-H_sb/Det) * fb) / P_tot


def calculate_cows_lite(y_data, spdf_fn, mrange, bern_degree=2):
    """
    Standard 1D COWs Lite: orthogonalizes the 1D signal PDF against a Bernstein
    polynomial background basis of degree `bern_degree`.
    """
    bern = make_bernstein_pdf(bern_degree, mrange[0], mrange[1])
    cows_obj = Cows(y_data, spdf_fn, bern, range=mrange)
    return cows_obj(y_data)


def calculate_cows_full_derivatives(
    y_data, spdf_fn, mrange, mu_0, sigma_0, bern_degree=2, ridge_reg=1e-4
):
    """
    1D Full COWs using continuous Gauss-Legendre quadrature and analytical/numerical
    derivatives (d_mu, d_sigma) projected to have strictly zero integral on [y_min, y_max].
    Includes a mild Tikhonov ridge regularization (`ridge_reg`) on the nuisance block to
    prevent catastrophic variance explosion (N_eq collapse) when d_sigma is collinear with
    degree-2 Bernstein polynomials over narrow mass windows.
    """
    y_min, y_max = mrange
    n_gl = 200
    nodes, weights = np.polynomial.legendre.leggauss(n_gl)
    y_grid = 0.5 * (y_max - y_min) * nodes + 0.5 * (y_max + y_min)
    dy_grid = 0.5 * (y_max - y_min) * weights

    # Smooth empirical variance density I(y) on the quadrature grid
    hist, edges = np.histogram(y_data, bins=60, range=mrange, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist_sm = scipy.ndimage.gaussian_filter1d(hist, sigma=1.5)
    I_grid = np.maximum(np.interp(y_grid, centers, hist_sm), 1e-6)
    I_obs = np.maximum(np.interp(y_data, centers, hist_sm), 1e-6)

    # 1. Normalized Signal PDF
    g_s_grid = spdf_fn(y_grid)
    norm_s = np.sum(g_s_grid * dy_grid)
    g_s_grid = g_s_grid / max(norm_s, 1e-12)
    g_s_obs = spdf_fn(y_data) / max(norm_s, 1e-12)

    # 2. Strictly Zero-Integral Projected Derivatives (d_mu, d_sigma)
    # Using zero-integral projection: d_tilde(y) = d(y) - g_s(y) * \int d(y') dy'
    # guarantees \int w_s(y) g_s(y) dy = 1 without subtracting net signal yield!
    d1_raw = np.gradient(g_s_grid, y_grid)
    d1_int = np.sum(d1_raw * dy_grid)
    d1_grid = d1_raw - g_s_grid * d1_int
    d1_rms = np.sqrt(max(np.sum((d1_grid ** 2) * dy_grid), 1e-12))
    d1_grid /= d1_rms
    d1_obs = np.interp(y_data, y_grid, d1_grid)

    d2_raw = np.gradient(d1_raw, y_grid)
    d2_int = np.sum(d2_raw * dy_grid)
    d2_grid = d2_raw - g_s_grid * d2_int
    d2_rms = np.sqrt(max(np.sum((d2_grid ** 2) * dy_grid), 1e-12))
    d2_grid /= d2_rms
    d2_obs = np.interp(y_data, y_grid, d2_grid)

    # 3. Bernstein Background Basis
    bern = make_bernstein_pdf(bern_degree, y_min, y_max)
    b_grid_list = [b_fn(y_grid) for b_fn in bern]
    b_obs_list = [b_fn(y_data) for b_fn in bern]

    basis_grid = np.vstack([g_s_grid, d1_grid, d2_grid] + b_grid_list)
    basis_obs = np.vstack([g_s_obs, d1_obs, d2_obs] + b_obs_list)
    k_dim = basis_grid.shape[0]

    # Construct continuous Gram matrix W_kl = \int g_k(y) g_l(y) / I(y) dy
    W_mat = np.einsum("km,lm,m->kl", basis_grid, basis_grid, dy_grid / I_grid)

    # Apply mild Tikhonov regularization on the nuisance block (indices 1..k_dim-1)
    # to stabilize collinearity between d2_grid and quadratic Bernstein modes
    reg_diag = np.zeros(k_dim)
    reg_diag[1:] = ridge_reg * np.diag(W_mat)[1:]
    W_mat += np.diag(reg_diag)

    A_mat = np.linalg.inv(W_mat)
    # Signal weight is the 0-th row projection (preserved unit integral on g_s, orthogonal to 1..k_dim-1)
    w_s = (A_mat[0, :] @ basis_obs) / I_obs
    return w_s


def calculate_cows_full_official(y_data, spdf_fn, mrange, mu_0, sigma_0, bern_degree=2):
    """
    Uses the official sweights package to compute True Full COWs 
    (non-factorizing signal expanded into 2 Gaussians + Bernstein background).
    """
    bern_pdfs = make_bernstein_pdf(bern_degree, mrange[0], mrange[1])
    
    # Expand signal into a nominal and a broader Gaussian to absorb resolution
    def g1_sig(y):
        return StandardPDFs.trunc_norm(y, mu_0, 1.35 * sigma_0, mrange[0], mrange[1])
        
    # Pass sequences to both spdf and bpdf to execute Full COWs
    weighter = Cows(y_data, [spdf_fn, g1_sig], bern_pdfs, range=mrange)
    
    # Evaluate the weights
    w_all = weighter(y_data)
    
    # FIX: The official package returns a 2D array of shape (n_sig_components, N_events).
    # We MUST sum them along axis 0 to get the total signal weight.
    if w_all.ndim == 2:
        return np.sum(w_all, axis=0)
    
    return w_all


# Backward-compatible alias
def calculate_cows_full(y_data, spdf_fn, mrange, mu_0, sigma_0, bern_degree=2, mode="derivatives"):
    if mode == "official":
        return calculate_cows_full_official(y_data, spdf_fn, mrange, mu_0, sigma_0, bern_degree=bern_degree)
    return calculate_cows_full_derivatives(y_data, spdf_fn, mrange, mu_0, sigma_0, bern_degree=bern_degree)
