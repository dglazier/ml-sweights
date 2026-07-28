import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import comb

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
        
    @staticmethod
    def bernstein(y, p_arr, y_min, y_max, n_poly):
        x = (y - y_min) / (y_max - y_min)
        p_shifted = p_arr - np.max(p_arr)
        exps = np.exp(p_shifted)
        betas = exps / np.sum(exps)
        val = np.zeros_like(x)
        for v, beta in enumerate(betas):
            val += beta * comb(n_poly, v) * (x**v) * ((1-x)**(n_poly-v))
        return np.maximum(val * (n_poly + 1) / (y_max - y_min), 1e-10)

def calculate_1d_sweights(Ns, Nb, fs, fb):
    """Rigorous 1D analytical covariance inversion."""
    P_tot = np.maximum(Ns * fs + Nb * fb, 1e-10)
    H_ss = np.sum((fs/P_tot)**2)
    H_bb = np.sum((fb/P_tot)**2)
    H_sb = np.sum((fs*fb)/P_tot**2)
    Det = np.maximum(H_ss * H_bb - H_sb**2, 1e-12)
    return ((H_bb/Det) * fs + (-H_sb/Det) * fb) / P_tot

def fit_1d_anchor(nll_func, init_guess, bounds):
    """Executes the Scipy Minimizer for the 1D Baseline Fit."""
    res = minimize(nll_func, init_guess, bounds=bounds)
    if not res.success:
        print(f"Warning: 1D Fit did not converge perfectly: {res.message}")
    return res.x
