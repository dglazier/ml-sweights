import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import comb
import logging

logger = logging.getLogger(__name__)

def fit_1d_anchor(y_data, y_min, y_max, n_poly=5):
    logger.info("Executing 1D Anchor Fit (Bernstein + Truncated Normal)...")
    y_width = y_max - y_min

    def fb_pdf(y, p_arr):
        x_sc = (y - y_min) / y_width
        betas = np.exp(p_arr - np.max(p_arr)) / np.sum(np.exp(p_arr - np.max(p_arr)))
        val = np.zeros_like(x_sc)
        for v, beta in enumerate(betas):
            val += beta * comb(n_poly, v) * (x_sc**v) * ((1-x_sc)**(n_poly-v))
        return np.maximum(val * (n_poly + 1) / y_width, 1e-10)

    def fb_sampler(n_events, p_arr):
        samples = []
        pdf_max = np.max(np.exp(p_arr - np.max(p_arr)) / np.sum(np.exp(p_arr - np.max(p_arr)))) * (n_poly + 1) / y_width 
        while len(samples) < n_events:
            n_gen = int((n_events - len(samples)) * 1.5)
            y_cand = np.random.uniform(y_min, y_max, n_gen)
            u = np.random.uniform(0, pdf_max, n_gen)
            samples.extend(y_cand[u < fb_pdf(y_cand, p_arr)])
        return np.array(samples[:n_events])

    def trunc_norm_pdf(y, m, sig):
        norm_fact = np.maximum(norm.cdf(y_max, m, sig) - norm.cdf(y_min, m, sig), 1e-10)
        return norm.pdf(y, m, sig) / norm_fact

    init_guess = [0.2, np.mean(y_data), np.std(y_data)] + [0.0] * (n_poly + 1)
    bnds = [(0.01, 0.99), (y_min, y_max), (0.001, np.std(y_data)*2)] + [(-10.0, 10.0)] * (n_poly + 1)

    res = minimize(
        lambda p: -np.sum(np.log(np.maximum(p[0]*trunc_norm_pdf(y_data, p[1], p[2]) + (1-p[0])*fb_pdf(y_data, p[3:]), 1e-10))), 
        init_guess, bounds=bnds
    )
    
    f0_anchor, m_peak, sig_peak = res.x[:3]
    poly_params = res.x[3:]

    fs_1d = trunc_norm_pdf(y_data, m_peak, sig_peak)
    fb_1d = fb_pdf(y_data, poly_params)
    w_s_init = (fs_1d * (f0_anchor/(1-f0_anchor))) / (fs_1d * (f0_anchor/(1-f0_anchor)) + fb_1d)

    logger.info(f"  --> Extracted Data Peak: {m_peak:.5f}")
    logger.info(f"  --> Yields Initialized: Signal = {f0_anchor*len(y_data):.0f}, Bkg = {(1-f0_anchor)*len(y_data):.0f}")

    # Return pure mathematical closures mapped to the fitted parameters
    return {
        'f0': f0_anchor,
        'y_peak': m_peak,
        'y_sigma': sig_peak,
        'w_s_init': w_s_init,
        'poly_params': poly_params,
        'bkg_pdf': lambda y: fb_pdf(y, poly_params),
        'bkg_sampler': lambda n: fb_sampler(n, poly_params),
        'trunc_norm_pdf': trunc_norm_pdf
    }
