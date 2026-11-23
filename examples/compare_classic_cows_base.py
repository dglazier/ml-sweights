"""
Classic 1D Non-Factorizing Toy Benchmark (make_classic_toy) comparing:
  1. Standard 1D sPlot
  2. Official 1D COWs Lite
  3. Custom 1D Full COWs (Derivative Expansion)
  4. Official 1D True Full COWs (Multi-Component)
  5. Multi-Dimensional Local ML-sWeights
"""

import sys, os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import kendalltau, norm as scipy_norm, expon
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL, ExtendedBinnedNLL

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sweights.testing import make_classic_toy
from sweights.util import make_bernstein_pdf

# --- PURE BASE EXTRACTOR IMPORT ---
from core.base_extractor import BaseMLSWeightExtractor
from core import calculate_1d_sweights
from core.extractor import Analytical1DGaussian
from core.baselines import (
    calculate_cows_lite, calculate_cows_full_official, calculate_cows_full_derivatives
)

from utils.diagnostics import (
    plot_stacked_1d, plot_final_1d_projections, plot_diagnostics_timelines
)


def fit_exponential_slope(t_data, w, trange):
    """Utility function to extract the exponential slope and Chi2 using Minuit."""
    val, xe = np.histogram(t_data, bins=40, range=trange, weights=w)
    var = np.maximum(np.histogram(t_data, bins=xe, weights=w ** 2)[0], 1.0)
    
    def model(x, n, slope):
        return n * expon.cdf(x, 0, slope)
        
    tmi = Minuit(ExtendedBinnedNLL(np.transpose((val, var)), xe, model), n=np.sum(w), slope=0.1)
    tmi.limits["n", "slope"] = (0.01, None)
    tmi.migrad()
    tmi.hesse()
    
    n_eq = np.sum(w)**2 / np.sum(w**2)
    return tmi.values['slope'], tmi.errors['slope'], tmi.fmin.reduced_chi2, n_eq


def main():
    parser = argparse.ArgumentParser(description="1D Classic COWs Benchmark")
    parser.add_argument("--runs", type=int, default=1, help="Number of independent toy datasets to generate and fit")
    parser.add_argument("--alpha", type=float, default=1.0, help="Fixed marginal_alpha")
    parser.add_argument("--n-nodes", type=int, default=35, help="Number of Legendre nodes for Model A background grid")
    parser.add_argument("--derivs", type=int, default=1, help="Extended COWs derivatives")
    parser.add_argument("--max-iters", type=int, default=120, help="Maximum EM iterations")
    parser.add_argument("--data-seed", type=int, default=None, help="Fixed random seed for dataset generation")
    args = parser.parse_args()

    mrange = (0.0, 1.0)
    trange = (0.0, 0.5)
    bounds_list = [trange]
    labels_list = ["Kinematic Var (t)"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trackers = {
        "A1. Standard 1D sPlot": {"slope": [], "err": [], "chi2": [], "neq": [], "time": []},
        "A2. Official COWs Lite": {"slope": [], "err": [], "chi2": [], "neq": [], "time": []},
        "A3. Custom Full COWs": {"slope": [], "err": [], "chi2": [], "neq": [], "time": []},
        "A4. Official Full COWs": {"slope": [], "err": [], "chi2": [], "neq": [], "time": []},
        "B1. ML-sWeights": {"slope": [], "err": [], "chi2": [], "neq": [], "time": []}
    }

    print("=======================================================================")
    print(f"Non-Factorizing Classic COWs Benchmark | Planned Runs: {args.runs}")
    print("=======================================================================")

    for run in range(args.runs):
        print(f"\n=======================================================================")
        print(f"--- RUN {run + 1}/{args.runs} ---")
        print(f"=======================================================================")
        
        run_seed = (args.data_seed + run) if args.data_seed is not None else run
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

        toy = make_classic_toy(
            run_seed, s=50000, b=200000, mrange=mrange, trange=trange,
            mb_mu=lambda t: 0.1 + 10 * (t / 1.5)
        )
        y_data, t_data, is_signal = toy[0], toy[1], toy[2]
        X_ctrl_data = t_data.reshape(-1, 1)
        w_in = np.ones(len(y_data), dtype=np.float64)

        if run == 0:
            sideband_mask = (y_data < 0.2) | (y_data > 0.8)
            tau_stat, p_value = kendalltau(y_data[sideband_mask], t_data[sideband_mask])
            print(f"Factorization Check -> Mass vs 't' Tau: {tau_stat:.4f}, p-value: {p_value:.2e}\n")

        print("Executing Robust 1D Minuit Fit...")
        t0_splot = time.time()
        bern = make_bernstein_pdf(2, *mrange)

        def m_density(x, s, b1, b2, b3, mu, sigma):
            ds = scipy_norm(mu, sigma)
            snorm = np.diff(ds.cdf(mrange))[0]
            sig_eval = (s / snorm) * ds.pdf(x)
            bkg_eval = b1 * bern[0](x) + b2 * bern[1](x) + b3 * bern[2](x)
            return s + b1 + b2 + b3, sig_eval + bkg_eval

        cost = ExtendedUnbinnedNLL(y_data, m_density)
        mi = Minuit(cost, s=4000, b1=3333, b2=3333, b3=3333, mu=0.5, sigma=0.05)
        mi.limits["s", "b1", "b2", "b3", "sigma"] = (0, None)
        mi.migrad()

        s_fit = mi.values["s"]
        b_fit = mi.values["b1"] + mi.values["b2"] + mi.values["b3"]
        
        def spdf(x):
            ds = scipy_norm(mi.values["mu"], mi.values["sigma"])
            return ds.pdf(x) / np.diff(ds.cdf(mrange))[0]
            
        def bpdf(x):
            b1, b2, b3 = mi.values["b1"], mi.values["b2"], mi.values["b3"]
            return (b1 * bern[0](x) + b2 * bern[1](x) + b3 * bern[2](x)) / max(b_fit, 1e-10)

        fs_1d_eval = spdf(y_data)
        fb_1d_eval = bpdf(y_data)
        
        w_splot_1d = calculate_1d_sweights(s_fit, b_fit, fs_1d_eval, fb_1d_eval)
        t1_splot = time.time()
        
        p_sig = s_fit * fs_1d_eval
        p_bkg = b_fit * fb_1d_eval
        w_prob_init = p_sig / (p_sig + p_bkg)
        
        m_grid = np.linspace(mrange[0], mrange[1], 5000)
        bpdf_grid = bpdf(m_grid)
        bpdf_grid /= np.sum(bpdf_grid)
        
        def exact_bkg_sampler(n):
            return np.random.choice(m_grid, size=n, p=bpdf_grid)

        t0_cows = time.time()
        w_cows_1d = calculate_cows_lite(y_data, spdf, (mrange[0], mrange[1]), bern_degree=2)
        t1_cows = time.time()
        
        t0_cows_deriv = time.time()
        w_cows_deriv = calculate_cows_full_derivatives(
            y_data=y_data, spdf_fn=spdf, mrange=(mrange[0], mrange[1]), 
            mu_0=mi.values["mu"], sigma_0=mi.values["sigma"], bern_degree=2
        )
        t1_cows_deriv = time.time()

        t0_cows_true = time.time()
        w_cows_true = calculate_cows_full_official(
            y_data=y_data, spdf_fn=spdf, mrange=(mrange[0], mrange[1]), 
            mu_0=mi.values["mu"], sigma_0=mi.values["sigma"], bern_degree=2
        )
        t1_cows_true = time.time()

        val_dict = {
            "x_raw": X_ctrl_data, "x_true": X_ctrl_data[is_signal],
            "bounds": bounds_list, "labels": labels_list,
            "y_raw": y_data, "y_bounds": mrange, 
            "plot_iterations": (run == args.runs - 1),
            "real_data_metric": True 
        }

        t0_ml = time.time()
        
        tunable_sig_pdf = Analytical1DGaussian(
            mu=mi.values["mu"], sigma=mi.values["sigma"], y_min=mrange[0], y_max=mrange[1], burn_in_iters=3, freeze_domain=True
        )
        
        # --- INSTANTIATE PURE BASE EXTRACTOR ---
        extractor = BaseMLSWeightExtractor(
            disc_bounds=mrange, 
            n_nodes=args.n_nodes, 
            gbdt_mode='XGB_GPU', 
            device=device,
            marginal_alpha=args.alpha,                 
            dnn_epochs=100, 
            extended_cows_derivatives=args.derivs, 
            max_iters=args.max_iters, 
            omega_init=2.0,
            omega_decay_iters=20.0, 
            tol=1e-4,
            n_folds=1
        )

        sWeights_ML = extractor.fit_transform(
            X_ctrl=X_ctrl_data, y_disc=y_data, w_s_init=w_prob_init, w_in=w_in,
            f_sig_cond=tunable_sig_pdf, f_bkg_1d=bpdf, bkg_sampler=exact_bkg_sampler, validation_data=val_dict
        )
        t1_ml = time.time()

        run_results = [
            ("A1. Standard 1D sPlot", w_splot_1d, t1_splot - t0_splot),
            ("A2. Official COWs Lite", w_cows_1d, t1_cows - t0_cows),
            ("A3. Custom Full COWs", w_cows_deriv, t1_cows_deriv - t0_cows_deriv),
            ("A4. Official Full COWs", w_cows_true, t1_cows_true - t0_cows_true),
            ("B1. ML-sWeights", sWeights_ML, t1_ml - t0_ml)
        ]
        
        for name, w, time_taken in run_results:
            slope, err, chi2, neq = fit_exponential_slope(t_data, w, trange)
            trackers[name]["slope"].append(slope)
            trackers[name]["err"].append(err)
            trackers[name]["chi2"].append(chi2)
            trackers[name]["neq"].append(neq)
            trackers[name]["time"].append(time_taken)

    print("\n======================================================================================")
    print(f"{'Configuration / Method':<30} | {'Slope':>7} | {'+/- Err':>7} | {'Chi2/ndof':>9} | {'N_eq':>8} | {'Time':>6}")
    print("-" * 86)
    
    for name in trackers.keys():
        mean_slope = np.mean(trackers[name]["slope"])
        mean_err = np.mean(trackers[name]["err"])
        mean_chi2 = np.mean(trackers[name]["chi2"])
        mean_neq = np.mean(trackers[name]["neq"])
        mean_time = np.mean(trackers[name]["time"])
        print(f"{name:<30} | {mean_slope:7.4f} | {mean_err:7.4f} | {mean_chi2:9.2f} | {mean_neq:8.0f} | {mean_time:5.1f}s")
    print("======================================================================================\n")

    print("\n7. Plotting Final Diagnostics (from final run)...")
    plt.ioff()
    
    plot_stacked_1d(X_ctrl_data, sWeights_ML, bounds_list, labels_list)
    
    plot_final_1d_projections(
        X_ctrl_data, X_ctrl_data[is_signal], X_ctrl_data[~is_signal], 
        w_splot_1d, sWeights_ML, 
        bounds_list, labels_list, w_prob=extractor.w_s,
        extra_methods=[
            ("Official COWs Lite", w_cows_1d, "green"),
            ("Custom Full COWs", w_cows_deriv, "orange"),
            ("Official Full COWs", w_cows_true, "purple")
        ]
    )

    plt.figure(figsize=(18, 10))
    bins = np.linspace(trange[0], trange[1], 40)
    
    final_run_weights = [w_splot_1d, w_cows_1d, w_cows_deriv, w_cows_true, sWeights_ML]
    colors = ["red", "green", "orange", "purple", "blue"]
    
    for idx, (name, metrics_dict) in enumerate(trackers.items()):
        plt.subplot(2, 3, idx + 1)
        plt.hist(t_data[is_signal], bins=bins, color="black", histtype="step", lw=2, label="True Signal")
        plt.hist(t_data, bins=bins, weights=final_run_weights[idx], color=colors[idx], histtype="step", ls="--", lw=2, label=name)
        plt.title(name)
        plt.xlabel("Kinematic Variable (t)")
        plt.legend()
        plt.grid(alpha=0.3)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
