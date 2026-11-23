"""
1D Development Benchmark with Exact Signed Timing Weights.
Isolates the effect of w_in (input weights) on the standard 1D EM Extractor
and benchmarks against Standard 1D sPlot, Official 1D COWs Lite,
Custom 1D Full COWs (Derivatives), and Official 1D Full COWs (Multi-Component).
"""

import sys, os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.omega_generators import OmegaToyGenerator
# --- IMPORT THE PURE BASE EXTRACTOR ---
from core.base_extractor import BaseMLSWeightExtractor
from core.signal_models import MCSpline1D
from core.flows import MCFlow1D
from core.anchor import fit_1d_anchor
from core.baselines import (
    calculate_1d_sweights,
    calculate_cows_lite,
    calculate_cows_full_official,
    calculate_cows_full_derivatives
)
from utils.diagnostics import (
    plot_stacked_1d, plot_final_1d_projections, plot_2d_correlations, 
    plot_1d_y_fit, plot_diagnostics_timelines, get_chi2, check_discriminant_orthogonality
)


def get_timing_weights(t_data):
    """
    Evaluates exact signed weights from the out-of-time distribution.
    Signal Region: 15-25 ns (Width = 10)
    Sideband Regions: 0-10 ns and 30-40 ns (Total Width = 20)
    Weight = 1.0 (Signal), -0.5 (Sidebands), 0.0 (Otherwise)
    """
    w_timing = np.zeros_like(t_data)
    sig_mask = (t_data >= 15.0) & (t_data <= 25.0)
    sb_mask = ((t_data >= 0.0) & (t_data < 10.0)) | ((t_data > 30.0) & (t_data <= 40.0))
    
    w_timing[sig_mask] = 1.0
    w_timing[sb_mask] = -0.5
    return w_timing


def plot_kinematic_mass_2d(x_data, y_disc, w_in, labels_list, y_bounds):
    """
    Plots a 2D histogram of the topological control variables vs the mass variable
    to visually diagnose kinematic thresholds, cliffs, and non-factorizing correlations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Pre-Fit Diagnostics: Kinematic Discontinuities & Phase Space Limits", fontsize=14, fontweight='bold')
    
    for i in range(3):
        x_range = [-1.0, 1.0] if i == 0 else [-np.pi, np.pi]
        h, xedges, yedges, im = axes[i].hist2d(
            x_data[:, i], y_disc, bins=[40, 40], 
            range=[x_range, y_bounds], 
            weights=w_in, cmap='viridis', cmin=0.1
        )
        axes[i].set_xlabel(labels_list[i], fontsize=12)
        axes[i].set_ylabel("Mass (y)", fontsize=12)
        axes[i].set_title(f"Mass vs {labels_list[i]}", fontsize=12)
        fig.colorbar(im, ax=axes[i], label="Effective Events")
        
        axes[i].axhline(0.782, color='red', linestyle='--', alpha=0.7, label=r'$\omega$ peak (0.782)')
        if i == 0:
            axes[i].legend(loc='upper right')

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def main():
    parser = argparse.ArgumentParser(description="1D Dev Benchmark with Signed Timing Weights")
    parser.add_argument("--data-seed", type=int, default=43, help="Fixed random seed")
    parser.add_argument("--n-runs", type=int, default=1, help="Number of independent ML runs to average")
    parser.add_argument("--max-iters", type=int, default=30, help="Maximum EM iterations")
    parser.add_argument("--n-nodes", type=int, default=64, help="Number of Gauss-Legendre quadrature nodes")
    parser.add_argument("--alpha", type=float, default=1.00, help="Fixed marginal alpha prior")
    parser.add_argument("--signal-model", type=str, default="flow_1d",
                        choices=["spline", "flow_1d", "cond_flow_1d"],
                        help="Signal model: spline (MCSpline1D), flow_1d (Unconditional 1D Flow), or cond_flow_1d (Conditional 1D Flow)")
    parser.add_argument("--derivs", type=int, default=2, help="Extended local COWs derivatives (0, 1, or 2)")
    parser.add_argument("--bern-degree", type=int, default=2, help="Bernstein polynomial degree for 1D COWs baselines")
    args = parser.parse_args()

    np.random.seed(args.data_seed)
    torch.manual_seed(args.data_seed)

    print("=======================================================================")
    print("1. Generating 1D Toy Data (Signal + Combinatorial + Out-of-Time)...")
    print("=======================================================================")
    factory = OmegaToyGenerator()
    Y_MIN, Y_MAX = 0.73, 0.84
    
    x_s, z_s, m_inv_s, m_mm_s, t_s = factory.generate_signal(8000, is_data=True, phase_space_cut=False)
    x_b1, z_b1, m_inv_b1, m_mm_b1, t_b1 = factory.generate_background(60000, bkg_type='combinatorial', phase_space_cut=False)
    x_b2, z_b2, m_inv_b2, m_mm_b2, t_b2 = factory.generate_background(30000, bkg_type='out_of_time', phase_space_cut=False)

    x_data_full = np.vstack([x_s, x_b1, x_b2])
    z_data_full = np.concatenate([z_s, z_b1, z_b2])
    m_inv_full = np.concatenate([m_inv_s, m_inv_b1, m_inv_b2])
    t_full = np.concatenate([t_s, t_b1, t_b2])

    print("\n=======================================================================")
    print("2. Applying Timing Sideband Weights & Mass Cuts...")
    print("=======================================================================")
    
    mass_mask = (m_inv_full >= Y_MIN) & (m_inv_full <= Y_MAX)
    x_cut = x_data_full[mass_mask]
    z_cut = z_data_full[mass_mask]
    m_inv_cut = m_inv_full[mass_mask]
    t_cut = t_full[mass_mask]

    w_timing_full = get_timing_weights(t_cut)
    
    time_mask = w_timing_full != 0.0
    x_data = x_cut[time_mask]
    z_data = z_cut[time_mask]
    Y_disc = m_inv_cut[time_mask]
    w_timing = w_timing_full[time_mask]

    s_mass_mask = (m_inv_s >= Y_MIN) & (m_inv_s <= Y_MAX)
    x_sig_true = x_s[s_mass_mask]
    t_sig_true = t_s[s_mass_mask]
    
    w_sig_true = get_timing_weights(t_sig_true)
    true_effective_yield = np.sum(w_sig_true)
    
    print(f"Total Physics Events Generated: {len(m_inv_full)}")
    print(f"Total Events Post-Cuts: {len(w_timing)}")
    print(f"Effective Weighted Data Yield (Sum w_in): {np.sum(w_timing):.1f}")
    print(f"--> TARGET TRUE SIGNAL YIELD (Sum w_in_sig): {true_effective_yield:.1f}")

    def map_topology(x):
        cT, phi, Phi = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        sT = np.sqrt(np.maximum(1 - cT**2, 0.0))
        return np.hstack([cT, sT, np.cos(phi), np.sin(phi), np.cos(Phi), np.sin(Phi)])

    X_ctrl = np.hstack([map_topology(x_data), z_data])
    bounds_list = [(-1.0, 1.0), (-np.pi, np.pi), (-np.pi, np.pi)]
    labels_list = [r"$\cos\theta$", r"$\phi$", r"$\Phi$"]
    
    plot_kinematic_mass_2d(x_data[:, :3], Y_disc, w_timing, labels_list, (Y_MIN, Y_MAX))

    print("\n=======================================================================")
    print("3. Generating Exact MC Signal Templates...")
    print("=======================================================================")
    x_mc, z_mc, m_inv_mc, _, _ = factory.generate_signal(500000, is_data=False, phase_space_cut=False)
    mc_mask = (m_inv_mc >= Y_MIN) & (m_inv_mc <= Y_MAX)
    x_mc = x_mc[mc_mask]
    z_mc = z_mc[mc_mask]
    m_inv_mc = m_inv_mc[mc_mask]
    X_ctrl_mc = np.hstack([map_topology(x_mc), z_mc])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    val_dict = {
        'x_raw': x_data[:, :3], 
        'x_true': x_sig_true[:, :3][w_sig_true != 0], 
        'bounds': bounds_list, 
        'labels': labels_list, 
        'y_raw': Y_disc, 
        'y_bounds': (Y_MIN, Y_MAX), 
        'w_true': w_sig_true[w_sig_true != 0]
    }

    plt.ion()

    print("\nExecuting 1D Anchor Fit (Bernstein + Truncated Normal) with Input Weights...")
    t0_splot = time.time()
    anc_1d = fit_1d_anchor(Y_disc, Y_MIN, Y_MAX, n_poly=6, w_in=w_timing)
    
    def spdf(y):
        return anc_1d['trunc_norm_pdf'](y, anc_1d['y_peak'], anc_1d['y_sigma'])

    plot_1d_y_fit(Y_disc, spdf, anc_1d['bkg_pdf'], anc_1d['f0'], Y_MIN, Y_MAX, w_in=w_timing)

    try:
        ortho_metrics = check_discriminant_orthogonality(
            y_min=Y_MIN, y_max=Y_MAX, f_sig_1d=spdf, f_bkg_1d=anc_1d['bkg_pdf'],
            f0=anc_1d['f0'], y_peak=anc_1d['y_peak'], y_sigma=anc_1d['y_sigma'],
            y_data=Y_disc, x_ctrl=x_data[:, :3], w_in=w_timing, labels=labels_list
        )
    except TypeError:
        ortho_metrics = check_discriminant_orthogonality(
            y_min=Y_MIN, y_max=Y_MAX, f_sig_1d=spdf, f_bkg_1d=anc_1d['bkg_pdf'],
            f0=anc_1d['f0'], y_peak=anc_1d['y_peak'], y_sigma=anc_1d['y_sigma']
        )
    
    N_tot = np.sum(w_timing)
    N_s_fit, N_b_fit = N_tot * anc_1d['f0'], N_tot * (1 - anc_1d['f0'])
    fs_1d_eval = spdf(Y_disc)
    fb_1d_eval = anc_1d['bkg_pdf'](Y_disc)
    w_splot_1d = w_timing * calculate_1d_sweights(N_s_fit, N_b_fit, fs_1d_eval, fb_1d_eval)
    t1_splot = time.time()

    print(f"\nExecuting Official 1D COWs Lite Extraction (deg={args.bern_degree})...")
    t0_cows = time.time()
    w_cows_1d = w_timing * calculate_cows_lite(
        Y_disc, spdf, (Y_MIN, Y_MAX), bern_degree=args.bern_degree
    )
    t1_cows = time.time()

    print(f"Executing Custom 1D Full COWs (Derivative Expansion, deg={args.bern_degree})...")
    t0_cows_deriv = time.time()
    w_cows_deriv = w_timing * calculate_cows_full_derivatives(
        y_data=Y_disc, spdf_fn=spdf, mrange=(Y_MIN, Y_MAX),
        mu_0=anc_1d['y_peak'], sigma_0=anc_1d['y_sigma'],
        bern_degree=args.bern_degree, ridge_reg=1e-3
    )
    t1_cows_deriv = time.time()

    print(f"Executing Official 1D True Full COWs (Multi-Component Signal, deg={args.bern_degree})...")
    t0_cows_true = time.time()
    w_cows_true = w_timing * calculate_cows_full_official(
        y_data=Y_disc, spdf_fn=spdf, mrange=(Y_MIN, Y_MAX),
        mu_0=anc_1d['y_peak'], sigma_0=anc_1d['y_sigma'],
        bern_degree=args.bern_degree
    )
    t1_cows_true = time.time()

    print(f"\n--- Extracting ML-sWeights (1D Pure Math Mode | Model: {args.signal_model} | Runs: {args.n_runs}) ---")
    
    all_sw = []
    all_ws = []
    t0_total_ml = time.time()
    
    extractor = None

    for run_idx in range(args.n_runs):
        if args.data_seed is not None:
            run_seed = args.data_seed + 1000 * (run_idx + 1)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)

        print(f"\n--- ML-sWeights Extraction (Run {run_idx + 1}/{args.n_runs}) ---")
        
        if args.signal_model == "spline":
            sig_1d = MCSpline1D(m_inv_mc, Y_MIN, Y_MAX, bins=100, smooth=1.0, burn_in_iters=1)
        elif args.signal_model == "flow_1d":
            sig_1d = MCFlow1D(m_inv_mc, Y_MIN, Y_MAX, mc_x=None, burn_in_iters=1, device=device)
        elif args.signal_model == "cond_flow_1d":
            sig_1d = MCFlow1D(m_inv_mc, Y_MIN, Y_MAX, mc_x=X_ctrl_mc, burn_in_iters=1, device=device)
        else:
            raise ValueError(f"Unknown signal_model: {args.signal_model}")

        # --- INSTANTIATE THE PURE BASE EXTRACTOR ---
        extractor = BaseMLSWeightExtractor(
            disc_bounds=(Y_MIN, Y_MAX),
            n_nodes=args.n_nodes,
            device=device,
            gbdt_mode='XGB_GPU',
            dnn_epochs=100,
            marginal_alpha=args.alpha,
            max_iters=args.max_iters,
            tol=1e-4,
            extended_cows_derivatives=args.derivs,
            omega_decay_iters=2.0,
            n_folds=1
        )
        
        
        # Enable plotting ONLY for the last run to avoid visual clutter
        val_run = val_dict.copy()
        val_run['plot_iterations'] = (run_idx == args.n_runs - 1)

        # --- PURE MATHEMATICAL EXECUTION ---
        sw_run = extractor.fit_transform(
            X_ctrl=X_ctrl, 
            y_disc=Y_disc, 
            w_s_init=anc_1d['w_s_init'],
            f_sig_cond=sig_1d, 
            f_bkg_1d=anc_1d['bkg_pdf'], 
            bkg_sampler=anc_1d['bkg_sampler'],
            w_in=w_timing,
            validation_data=val_run 
        )
       
        all_sw.append(sw_run)
        all_ws.append(extractor.w_s)

    t_elapsed = time.time() - t0_total_ml

    if args.n_runs > 1:
        sW_final = np.mean(all_sw, axis=0)
        w_prob_eff = np.mean(all_ws, axis=0) * w_timing
    else:
        sW_final = all_sw[0]
        w_prob_eff = all_ws[0] * w_timing

    label_prob = f"B1. 1D ML-Prob ({args.n_runs}-Run)" if args.n_runs > 1 else "B1. 1D ML-Prob (w_in*w_s)"
    label_ml = f"B2. 1D ML-COWs ({args.n_runs}-Run)" if args.n_runs > 1 else f"B2. 1D ML-sWeights ({args.signal_model})"

    results = [
        ("A1. Standard 1D sPlot", w_splot_1d, t1_splot - t0_splot),
        (f"A2. Official COWs Lite (deg={args.bern_degree})", w_cows_1d, t1_cows - t0_cows),
        (f"A3. Custom Full COWs (Derivs, deg={args.bern_degree})", w_cows_deriv, t1_cows_deriv - t0_cows_deriv),
        (f"A4. Official Full COWs (Multi-Comp)", w_cows_true, t1_cows_true - t0_cows_true),
        (label_prob, w_prob_eff, t_elapsed),
        (label_ml, sW_final, t_elapsed),
    ]

    print("\n=====================================================================================================================")
    print(f"{'Configuration / Method':<38} |  cos(th) |      phi |      Phi | Mean Chi2 |      N_eq | YieldBias |    Time")
    print("-" * 117)

    for name, w, time_taken in results:
        chi2s = get_chi2(x_data[:, :3], w, val_dict['x_true'], bounds_list, w_true=val_dict['w_true'])
        mean_chi2 = np.mean(chi2s)
        n_eq = (np.sum(w)**2) / max(np.sum(w**2), 1e-12)
        bias = (np.sum(w) - true_effective_yield) / max(true_effective_yield, 1.0) * 100
        print(f"{name:<38} | {chi2s[0]:8.2f} | {chi2s[1]:8.2f} | {chi2s[2]:8.2f} | {mean_chi2:9.2f} | {n_eq:9.1f} | {bias:8.2f}% | {time_taken:6.1f}s")
    print("=====================================================================================================================\n")

    print("\n4. Plotting Final Diagnostics...")
    plt.ioff()
    
    # Passing the minimal Base Extractor; timeline plots will be skipped since history isn't saved anymore,
    # but final projections and correlations will plot beautifully.
    plot_stacked_1d(x_data[:, :3], sW_final, bounds_list, labels_list, w_in=w_timing)
    
    plot_final_1d_projections(
        x_data[:, :3], val_dict['x_true'], val_dict['x_true'], w_splot_1d, sW_final, 
        bounds_list, labels_list, w_prob=w_prob_eff, w_true=val_dict['w_true'],
        extra_methods=[
            ("Official COWs Lite", w_cows_1d, "teal"),
            ("Official Full COWs", w_cows_true, "purple"),
            ("Custom Full COWs (Derivs)", w_cows_deriv, "orange"),
        ]
    )

    axes_pairs = [(0, 1, r"$\cos\theta$", r"$\phi$"), (0, 2, r"$\cos\theta$", r"$\Phi$"), (1, 2, r"$\phi$", r"$\Phi$")]
    plot_2d_correlations(x_data[:, :3], val_dict['x_true'], w_splot_1d, sW_final, axes_pairs, bounds_list, is_signal=True)

    plt.show()

if __name__ == "__main__":
    main()
