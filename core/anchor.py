# core/anchor.py
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, truncnorm
from scipy.special import comb
import scipy.ndimage
import logging

logger = logging.getLogger(__name__)


def fit_1d_anchor(y_data, y_min, y_max, n_poly=5, w_in=None):
    logger.info("Executing 1D Anchor Fit (Bernstein + Truncated Normal)...")
    y_width = y_max - y_min
    w_arr = np.ones_like(y_data, dtype=np.float64) if w_in is None else np.asarray(w_in, dtype=np.float64)

    def fb_pdf(y, p_arr):
        x_sc = np.clip((y - y_min) / y_width, 0.0, 1.0)
        betas = np.exp(p_arr - np.max(p_arr)) / np.sum(np.exp(p_arr - np.max(p_arr)))
        val = np.zeros_like(x_sc, dtype=np.float64)
        for v, beta in enumerate(betas):
            val += beta * comb(n_poly, v) * (x_sc**v) * ((1.0 - x_sc)**(n_poly - v))
        return np.maximum(val * (n_poly + 1.0) / y_width, 1e-10)

    def fb_sampler(n_events, p_arr):
        grid = np.linspace(y_min, y_max, 4096)
        probs = fb_pdf(grid, p_arr)
        probs /= np.sum(probs)
        idx = np.random.choice(len(grid), size=n_events, p=probs)
        jitter = np.random.uniform(-0.5, 0.5, size=n_events) * (grid[1] - grid[0])
        return np.clip(grid[idx] + jitter, y_min, y_max)

    def trunc_norm_pdf(y, m, sig):
        norm_fact = np.maximum(norm.cdf(y_max, m, sig) - norm.cdf(y_min, m, sig), 1e-10)
        return np.maximum(norm.pdf(y, m, sig) / norm_fact, 1e-10)

    # Seed peak position at the mode of the smoothed weighted histogram
    counts, edges = np.histogram(y_data, bins=60, range=(y_min, y_max), weights=w_arr)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sm_counts = scipy.ndimage.gaussian_filter1d(counts, sigma=1.5)
    inner_mask = (centers > y_min + 0.15 * y_width) & (centers < y_max - 0.15 * y_width)
    y_mean_init = float(centers[inner_mask][np.argmax(sm_counts[inner_mask])])
    y_std_init = float(0.08 * y_width)

    init_guess = [0.20, y_mean_init, y_std_init] + [0.0] * (n_poly + 1)
    bnds = [(0.005, 0.995), (y_min + 0.05 * y_width, y_max - 0.05 * y_width), (0.002, 0.25 * y_width)] + [(-10.0, 10.0)] * (n_poly + 1)

    def weighted_nll(p):
        mix = p[0] * trunc_norm_pdf(y_data, p[1], p[2]) + (1.0 - p[0]) * fb_pdf(y_data, p[3:])
        return -np.sum(w_arr * np.log(np.maximum(mix, 1e-10)))

    res = minimize(weighted_nll, init_guess, bounds=bnds, method='L-BFGS-B')
    f0_anchor, m_peak, sig_peak = res.x[:3]
    poly_params = res.x[3:]

    fs_1d = trunc_norm_pdf(y_data, m_peak, sig_peak)
    fb_1d = fb_pdf(y_data, poly_params)
    
    odds = f0_anchor / max(1.0 - f0_anchor, 1e-10)
    w_s_init = (fs_1d * odds) / np.maximum(fs_1d * odds + fb_1d, 1e-12)

    eff_tot = float(np.sum(w_arr))
    logger.info(f"  --> Extracted Data Peak: {m_peak:.5f} (sigma: {sig_peak:.5f})")
    logger.info(f"  --> Weighted Yields Initialized: Signal = {f0_anchor * eff_tot:.0f}, Bkg = {(1.0 - f0_anchor) * eff_tot:.0f}")

    return {
        'f0': f0_anchor,
        'y_peak': m_peak,
        'y_sigma': sig_peak,
        'w_s_init': np.clip(w_s_init, 0.0, 1.0),
        'poly_params': poly_params,
        'bkg_pdf': lambda y: fb_pdf(y, poly_params),
        'bkg_sampler': lambda n: fb_sampler(n, poly_params),
        'trunc_norm_pdf': trunc_norm_pdf
    }


def _sample_trunc_norm(m, sig, y_min, y_max, size):
    a = (y_min - m) / max(sig, 1e-10)
    b = (y_max - m) / max(sig, 1e-10)
    return truncnorm.rvs(a, b, loc=m, scale=sig, size=size)


def fit_2d_anchor(y_data_2d, y1_bounds, y2_bounds, n_poly=(5, 5), w_in=None):
    """
    Fits a 4-Component 2D Anchor Model:
        p(y1, y2) = f_SS * S1(y1) * S2(y2)
                  + f_BS * B1(y1) * S2(y2)   [Combinatorial Exclusive Ridge]
                  + f_SB * S1(y1) * B2(y2)   [Non-Exclusive Peaking-Mass Ridge]
                  + f_BB * B1(y1) * B2(y2)   [Smooth 2D Continuum]
    Prevents the 2D signal component from swallowing peaking background ridges.
    """
    logger.info("Executing 4-Component 2D Anchor Fit (S1*S2 + B1*S2 + S1*B2 + B1*B2)...")
    y1, y2 = y_data_2d[:, 0], y_data_2d[:, 1]
    w_arr = np.ones_like(y1, dtype=np.float64) if w_in is None else np.asarray(w_in, dtype=np.float64)

    # 1. Fit 1D marginals to obtain accurate shapes for (S1, B1) and (S2, B2)
    n1 = n_poly[0] if isinstance(n_poly, (list, tuple)) else int(n_poly)
    n2 = n_poly[1] if isinstance(n_poly, (list, tuple)) else int(n_poly)
    anc1 = fit_1d_anchor(y1, y1_bounds[0], y1_bounds[1], n_poly=n1, w_in=w_arr)
    anc2 = fit_1d_anchor(y2, y2_bounds[0], y2_bounds[1], n_poly=n2, w_in=w_arr)

    S1_func = lambda a1: anc1['trunc_norm_pdf'](a1, anc1['y_peak'], anc1['y_sigma'])
    B1_func = anc1['bkg_pdf']
    S2_func = lambda a2: anc2['trunc_norm_pdf'](a2, anc2['y_peak'], anc2['y_sigma'])
    B2_func = anc2['bkg_pdf']

    s1_ev, b1_ev = S1_func(y1), B1_func(y1)
    s2_ev, b2_ev = S2_func(y2), B2_func(y2)

    comp_SS = s1_ev * s2_ev
    comp_BS = b1_ev * s2_ev
    comp_SB = s1_ev * b2_ev
    comp_BB = b1_ev * b2_ev

    # 2. Fit the 4 component fractions via softmax logits
    def nll_fractions(logits):
        exps = np.exp(logits - np.max(logits))
        fracs = exps / np.sum(exps)
        mix = fracs[0] * comp_SS + fracs[1] * comp_BS + fracs[2] * comp_SB + fracs[3] * comp_BB
        return -np.sum(w_arr * np.log(np.maximum(mix, 1e-12)))

    res = minimize(nll_fractions, [0.0, 0.0, 0.0, 0.0], method='L-BFGS-B')
    exps = np.exp(res.x - np.max(res.x))
    f_SS, f_BS, f_SB, f_BB = exps / np.sum(exps)
    f_bkg_tot = max(1.0 - f_SS, 1e-10)

    # Normalized composite 2D background weights
    w_BS = f_BS / f_bkg_tot
    w_SB = f_SB / f_bkg_tot
    w_BB = f_BB / f_bkg_tot

    def dynamic_fb_func(y_arr):
        y_arr = np.atleast_2d(y_arr)
        y1_in, y2_in = y_arr[:, 0], y_arr[:, 1]
        val = (
            w_BS * B1_func(y1_in) * S2_func(y2_in)
            + w_SB * S1_func(y1_in) * B2_func(y2_in)
            + w_BB * B1_func(y1_in) * B2_func(y2_in)
        )
        return np.maximum(val, 1e-10)

    def bkg_sampler_2d(n_events):
        counts = np.random.multinomial(n_events, [w_BS, w_SB, w_BB])
        samps = []
        if counts[0] > 0:
            y1_s = anc1['bkg_sampler'](counts[0])
            y2_s = _sample_trunc_norm(anc2['y_peak'], anc2['y_sigma'], y2_bounds[0], y2_bounds[1], counts[0])
            samps.append(np.column_stack([y1_s, y2_s]))
        if counts[1] > 0:
            y1_s = _sample_trunc_norm(anc1['y_peak'], anc1['y_sigma'], y1_bounds[0], y1_bounds[1], counts[1])
            y2_s = anc2['bkg_sampler'](counts[1])
            samps.append(np.column_stack([y1_s, y2_s]))
        if counts[2] > 0:
            y1_s = anc1['bkg_sampler'](counts[2])
            y2_s = anc2['bkg_sampler'](counts[2])
            samps.append(np.column_stack([y1_s, y2_s]))
        out = np.vstack(samps)
        np.random.shuffle(out)
        return out

    fb_eval = dynamic_fb_func(y_data_2d)
    odds = f_SS / f_bkg_tot
    w_s_init = (comp_SS * odds) / np.maximum(comp_SS * odds + fb_eval, 1e-12)

    eff_tot = float(np.sum(w_arr))
    logger.info(f"  --> 2D Anchor Fractions: S1*S2={f_SS:.3f}, B1*S2={f_BS:.3f}, S1*B2={f_SB:.3f}, B1*B2={f_BB:.3f}")
    logger.info(f"  --> Weighted 2D Yields Initialized: Signal (S1*S2) = {f_SS * eff_tot:.0f}, Total Bkg = {f_bkg_tot * eff_tot:.0f}")

    return {
        'f0': float(f_SS),
        'fractions': (float(f_SS), float(f_BS), float(f_SB), float(f_BB)),
        'm1': anc1['y_peak'],
        's1': anc1['y_sigma'],
        'm2': anc2['y_peak'],
        's2': anc2['y_sigma'],
        'w_s_init': np.clip(w_s_init, 0.0, 1.0),
        'fs_func': lambda Z, y1_arr, y2_arr: S1_func(y1_arr) * S2_func(y2_arr),
        'fb_func': dynamic_fb_func,
        'bkg_sampler_2d': bkg_sampler_2d
    }
