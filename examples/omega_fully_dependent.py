import sys, os
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import kendalltau

# Add parent directory to path to allow absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.omega_generators import OmegaToyGenerator
from core import MLSWeightExtractor, fit_1d_anchor, build_signal_flow
from core.baselines import calculate_1d_sweights
from utils.diagnostics import (plot_stacked_1d, plot_final_1d_projections, 
                               plot_2d_correlations, plot_1d_y_fit, plot_sliced_y_grids, get_chi2)

def main():
    # ====================================================================
    # 1. PHYSICS SETUP & DATA GENERATION
    # ====================================================================
    print("1. Generating Fully Correlated Toy Data...")
    factory = OmegaToyGenerator()
    Y_MIN, Y_MAX = 0.73, 0.85

    # INCREASED DATASET SIZE: Deep learning cascades require adequate statistics 
    # to map 7D non-factorizable phase spaces without overfitting to noise.
    x_mc, z_mc, y_mc = factory.generate_signal(100000, mode='correlated', is_data=False)
    x_sig, z_sig, y_sig = factory.generate_signal(5000, mode='correlated', is_data=True)
    x_bkg, z_bkg, y_bkg = factory.generate_background(40000, mode='correlated')

    x_data_full = np.vstack([x_sig, x_bkg])
    z_data_full = np.concatenate([z_sig, z_bkg])
    y_data_full = np.concatenate([y_sig, y_bkg])

    mask = (y_data_full >= Y_MIN) & (y_data_full <= Y_MAX)
    x_data, z_data, y_data = x_data_full[mask], z_data_full[mask], y_data_full[mask]
    x_sig_true = x_sig[(y_sig >= Y_MIN) & (y_sig <= Y_MAX)]
    x_bkg_true = x_bkg[(y_bkg >= Y_MIN) & (y_bkg <= Y_MAX)]

    N_tot = len(y_data)

    def map_topology(x):
        """Maps periodic angles into continuous sine/cosine components"""
        cT, phi, Phi = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        sT = np.sqrt(np.maximum(1 - cT**2, 0.0))
        return np.hstack([cT, sT, np.cos(phi), np.sin(phi), np.cos(Phi), np.sin(Phi)])

    X_ctrl_data = np.hstack([map_topology(x_data), z_data])
    Z_ctx_mc = np.column_stack([x_mc[:, :3], z_mc])

    bounds_list = [(-1.0, 1.0), (-np.pi, np.pi), (-np.pi, np.pi)]
    labels_list = [r"$\cos\theta$", r"$\phi$", r"$\Phi$"]

    # ====================================================================
    # 2. FACTORIZATION TEST (Kendall's Tau on Sidebands)
    # ====================================================================
    print("\n--- Factorization Check (Kendall's Tau) ---")

    sideband_mask = (y_data < 0.76) | (y_data > 0.81)

    tau_stat_cos, p_value_cos = kendalltau(y_data[sideband_mask], x_data[sideband_mask, 0])
    tau_stat_phi, p_value_phi = kendalltau(y_data[sideband_mask], x_data[sideband_mask, 1])

    print(f"Mass vs cos(theta) -> Tau: {tau_stat_cos:.4f}, p-value: {p_value_cos:.2e}")
    if p_value_cos < 0.05:
        print("  --> Result: SIGNIFICANT non-factorization detected in cos(theta).")

    print(f"Mass vs phi        -> Tau: {tau_stat_phi:.4f}, p-value: {p_value_phi:.2e}")
    if p_value_phi < 0.05:
        print("  --> Result: SIGNIFICANT non-factorization detected in phi.")
    print("-------------------------------------------\n")

    # ====================================================================
    # 3. 1D BASELINE ANCHOR FIT
    # ====================================================================
    anchor = fit_1d_anchor(y_data, Y_MIN, Y_MAX, n_poly=5)

    N_s_fit, N_b_fit = N_tot * anchor['f0'], N_tot * (1 - anchor['f0'])
    fs_1d_eval = anchor['trunc_norm_pdf'](y_data, anchor['y_peak'], anchor['y_sigma'])
    fb_1d_eval = anchor['bkg_pdf'](y_data)

    w_splot_1d = calculate_1d_sweights(N_s_fit, N_b_fit, fs_1d_eval, fb_1d_eval)

    chi2s_1d = get_chi2(x_data, w_splot_1d, x_sig_true, bounds=bounds_list)
    print(f"--> Baseline 1D sPlot Chi2 (cos_theta, phi, Phi): [{', '.join([f'{c:.1f}' for c in chi2s_1d])}]\n")

    # ====================================================================
    # 4. NORMALIZING FLOW TRAINING & INITIALIZATION
    # ====================================================================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sig_pdf_Z = build_signal_flow(
        Z_ctx_mc, y_mc, 
        factory.m_omega, factory.base_sigma, anchor['y_peak'], device=device
    )

    def user_signal_pdf(X_ctrl, y_array):
        """Inverse mapping from topological features back to angles for the Flow"""
        cT, cp, sp, cP, sP, z = X_ctrl[:, 0], X_ctrl[:, 2], X_ctrl[:, 3], X_ctrl[:, 4], X_ctrl[:, 5], X_ctrl[:, 6]
        phi, Phi = np.arctan2(sp, cp), np.arctan2(sP, cP)
        Z_ctx = np.column_stack([cT, phi, Phi, z])
        return sig_pdf_Z(Z_ctx, y_array)

    # EXPLICIT STATE HOOKS: Allows the EM loop to dynamically warp the CNF 
    # to account for detector resolution smearing, and rollback if cannibalization occurs.
    def user_signal_update(X_ctrl, y_array, w_s_arr):
        if hasattr(sig_pdf_Z, 'update'):
            cT, cp, sp, cP, sP, z = X_ctrl[:, 0], X_ctrl[:, 2], X_ctrl[:, 3], X_ctrl[:, 4], X_ctrl[:, 5], X_ctrl[:, 6]
            phi, Phi = np.arctan2(sp, cp), np.arctan2(sP, cP)
            Z_ctx = np.column_stack([cT, phi, Phi, z])
            sig_pdf_Z.update(Z_ctx, y_array, w_s_arr)

    def user_signal_get_state():
        if hasattr(sig_pdf_Z, 'calib_net'):
            return {k: v.cpu().clone() for k, v in sig_pdf_Z.calib_net.state_dict().items()}
        return None

    def user_signal_load_state(state):
        if hasattr(sig_pdf_Z, 'calib_net') and state is not None:
            sig_pdf_Z.calib_net.load_state_dict({k: v.to(device) for k, v in state.items()})

    user_signal_pdf.update = user_signal_update
    user_signal_pdf.get_state = user_signal_get_state
    user_signal_pdf.load_state = user_signal_load_state

    # ====================================================================
    # 5. INITIAL DIAGNOSTIC PLOTS
    # ====================================================================
    plt.ion()
    plot_1d_y_fit(y_data, lambda y: anchor['trunc_norm_pdf'](y, anchor['y_peak'], anchor['y_sigma']), 
                     anchor['bkg_pdf'], anchor['f0'], Y_MIN, Y_MAX)

    plot_sliced_y_grids(
        x_raw=x_data, X_ctrl=X_ctrl_data, y_data=y_data, w_sig=anchor['w_s_init'], 
        bounds_list=bounds_list, f_sig_cond=user_signal_pdf, f_bkg_1d=anchor['bkg_pdf'], 
        Y_MIN=Y_MIN, Y_MAX=Y_MAX, device=device, is_iter_0=True, iter_label="Iter 0 Anchor"
    )

    # ====================================================================
    # 6. MULTI-DIMENSIONAL SWEIGHT EXTRACTION
    # ====================================================================
    val_dict = {
        'x_raw': x_data, 'x_true': x_sig_true, 'bounds': bounds_list, 
        'labels': labels_list, 'y_raw': y_data, 'y_bounds': (Y_MIN, Y_MAX),
        'plot_iterations': True
    }

    extractor = MLSWeightExtractor(
        disc_bounds=(Y_MIN, Y_MAX), 
        n_nodes=40, 
        gbdt_mode='XGB_GPU',
        device=device,
        marginal_alpha=0.5,
        dynamic_alpha=True,
        use_gbdt=True,
        reinstate_nn=True,
        dnn_epochs=100,
        ml_transform=None,
        loss_fn='BCE',           
        logit_l2_penalty=0.0001     
    )

    sWeights_MD = extractor.fit_transform(
        X_ctrl=X_ctrl_data, y_disc=y_data, w_s_init=anchor['w_s_init'], 
        f_sig_cond=user_signal_pdf, f_bkg_1d=anchor['bkg_pdf'], bkg_sampler=anchor['bkg_sampler'],
        max_iters=40, validation_data=val_dict, omega_init=2.0, omega_decay_iters=20.0
    )

    # ====================================================================
    # 7. FINAL DIAGNOSTICS & PLOTTING
    # ====================================================================
    plt.ioff()

    plot_sliced_y_grids(
        x_raw=x_data, X_ctrl=X_ctrl_data, y_data=y_data, w_sig=sWeights_MD, 
        bounds_list=bounds_list, f_sig_cond=user_signal_pdf, f_bkg_1d=anchor['bkg_pdf'], 
        Y_MIN=Y_MIN, Y_MAX=Y_MAX, device=device, is_iter_0=False, 
        iter_label="Final Extraction", extractor_obj=extractor
    )

    plot_stacked_1d(x_data, sWeights_MD, bounds_list, labels_list)
    plot_final_1d_projections(x_data, x_sig_true, x_bkg_true, w_splot_1d, sWeights_MD, bounds_list, labels_list, w_prob=extractor.w_s)

    axes_pairs = [(0, 1, r"$\cos\theta$", r"$\phi$"), (0, 2, r"$\cos\theta$", r"$\Phi$"), (1, 2, r"$\phi$", r"$\Phi$")]
    plot_2d_correlations(x_data, x_sig_true, w_splot_1d, sWeights_MD, axes_pairs, bounds_list, is_signal=True, w_prob=extractor.w_s)
    plot_2d_correlations(x_data, x_bkg_true, (1-w_splot_1d), (1-sWeights_MD), axes_pairs, bounds_list, is_signal=False, w_prob=(1-extractor.w_s))

    plt.show()

if __name__ == "__main__":
    main()
