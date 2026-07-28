import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_2d_hist(ax, fig, x_arr, y_arr, w_arr, range_bounds, title, xlabel, ylabel, vmin=None, vmax=None, cmap='viridis'):
    counts, _, _ = np.histogram2d(x_arr, y_arr, bins=30, range=range_bounds, weights=w_arr)
    
    # Auto-scale symmetric diverging colormaps if limits aren't provided
    if vmin is None and vmax is None:
        max_abs = np.max(np.abs(counts))
        if max_abs == 0: max_abs = 1e-6
        vmin, vmax = -max_abs, max_abs
        cmap = 'coolwarm'
        
    h = ax.hist2d(x_arr, y_arr, bins=30, range=range_bounds, weights=w_arr, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(h[3], ax=ax)
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_1d_y_fit(y_data, fs_func, fb_func, f0, y_min, y_max):
    fig0, ax0 = plt.subplots(figsize=(10, 6))
    ax0.hist(y_data, bins=60, color='gray', alpha=0.5, label='Experimental Data')
    y_plot = np.linspace(y_min, y_max, 200)
    bin_w = (y_max - y_min) / 60
    N_tot = len(y_data)
    
    fs_1d_plot = fs_func(y_plot)
    fb_1d_plot = fb_func(y_plot)
    
    ax0.plot(y_plot, N_tot * bin_w * (1-f0) * fb_1d_plot, 'gray', linestyle=':', lw=2, label='1D Background PDF')
    ax0.plot(y_plot, N_tot * bin_w * (f0 * fs_1d_plot + (1-f0) * fb_1d_plot), 'r--', lw=3, label='Standard 1D Anchor Fit')
    ax0.set_title("1D Analytical Anchor Mass Fit")
    ax0.legend()
    plt.pause(1.5)

def plot_stacked_1d(x_data, w_s, bounds_list, labels_list):
    fig1, axs1 = plt.subplots(1, 3, figsize=(18, 5))
    fig1.suptitle("Sanity Check: Total Data vs. Stacked ML Extraction (Signal + Background)", fontsize=16)
    for i in range(3):
        bins = np.linspace(bounds_list[i][0], bounds_list[i][1], 30)
        axs1[i].hist(x_data[:, i], bins=bins, histtype='step', color='black', lw=2, label="Total Data")
        axs1[i].hist([x_data[:, i], x_data[:, i]], bins=bins, weights=[1-w_s, w_s], 
                     stacked=True, color=['gray', 'green'], alpha=0.5, label=['ML Background', 'ML Signal'])
        axs1[i].set_title(f"Stacked {labels_list[i]}")
        if i==0: axs1[i].legend()
    plt.tight_layout()

def plot_final_1d_projections(x_data, x_sig_true, x_bkg_true, sWeights_1D, sWeights_MD, bounds_list, labels_list, w_prob=None):
    fig2, axs2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle(r"Final 1D Projection: Standard 1D $sPlot$ vs Rigorous ND $sWeights$", fontsize=16)
    
    for i in range(3):
        bins = np.linspace(bounds_list[i][0], bounds_list[i][1], 30)
        
        # Signal Row
        axs2[0, i].hist(x_sig_true[:, i], bins=bins, histtype='step', color='black', lw=2, label="True Signal")
        axs2[0, i].hist(x_data[:, i], bins=bins, weights=sWeights_1D, histtype='step', color='orange', linestyle='--', lw=3, alpha=0.8, label="1D sPlot")
        axs2[0, i].hist(x_data[:, i], bins=bins, weights=sWeights_MD, histtype='step', color='blue', lw=2, label="ND sWeights")
        if w_prob is not None:
            axs2[0, i].hist(x_data[:, i], bins=bins, weights=w_prob, histtype='step', color='magenta', lw=1.5, alpha=0.9, label="ML Prob Weights")
        
        axs2[0, i].set_title(f"Extracted Signal {labels_list[i]}")
        if i==0: axs2[0, i].legend(loc='lower center')
        
        # Background Row
        axs2[1, i].hist(x_bkg_true[:, i], bins=bins, histtype='step', color='gray', lw=2, label="True Background")
        axs2[1, i].hist(x_data[:, i], bins=bins, weights=(1-sWeights_1D), histtype='step', color='lightgreen', linestyle='--', lw=2, alpha=0.8, label="1D sPlot")
        axs2[1, i].hist(x_data[:, i], bins=bins, weights=(1-sWeights_MD), histtype='step', color='purple', lw=2, label="ND sWeights")
        if w_prob is not None:
            axs2[1, i].hist(x_data[:, i], bins=bins, weights=(1-w_prob), histtype='step', color='magenta', lw=1.5, alpha=0.9, label="ML Prob Weights")
        
        axs2[1, i].set_title(f"Extracted Background {labels_list[i]}")
        if i==0: axs2[1, i].legend(loc='lower center')
        
    plt.tight_layout()

def plot_2d_correlations(x_data, x_true, w_1d, w_md, axes_pairs, bounds_list, is_signal=True, w_prob=None):
    fig_title = r"2D SIGNAL Correlations" if is_signal else r"2D BACKGROUND Correlations"
    title_prefix = "True Signal" if is_signal else "True Background"
    
    sources = [
        (x_true, np.ones(len(x_true)), title_prefix), 
        (x_data, w_1d, "1D sPlot"), 
        (x_data, w_md, "ND sWeights")
    ]
    if w_prob is not None:
        sources.append((x_data, w_prob, "ML Prob Weights"))
        diff_weights = w_prob - w_md
        sources.append((x_data, diff_weights, "Diff: Prob - ND sWeights"))
        
    n_rows = len(sources)
    fig3, axs3 = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    fig3.suptitle(f"{fig_title}: Multi-method Comparison", fontsize=18)
        
    for c in range(3):
        idx_x, idx_y, lbl_x, lbl_y = axes_pairs[c]
        range_x = bounds_list[idx_x]
        range_y = bounds_list[idx_y]
        
        # Globally lock the Z-axis limits for this column (variable pair)
        c_vmin, c_vmax = 0, 0
        for r in range(n_rows):
            curr_data, curr_weights, row_title = sources[r]
            if "Diff" in row_title: continue
            counts, _, _ = np.histogram2d(curr_data[:, idx_x], curr_data[:, idx_y], bins=30, range=[range_x, range_y], weights=curr_weights)
            c_vmin = min(c_vmin, np.min(counts))
            c_vmax = max(c_vmax, np.max(counts))
            
        for r in range(n_rows):
            curr_data, curr_weights, row_title = sources[r]
            
            if "Diff" in row_title:
                # Differences are forced to diverge symmetrically around zero
                counts, _, _ = np.histogram2d(curr_data[:, idx_x], curr_data[:, idx_y], bins=30, range=[range_x, range_y], weights=curr_weights)
                diff_max = np.max(np.abs(counts))
                if diff_max == 0: diff_max = 1e-6
                plot_2d_hist(axs3[r, c], fig3, curr_data[:, idx_x], curr_data[:, idx_y], curr_weights, 
                             [range_x, range_y], row_title if c==0 else "", lbl_x, lbl_y, vmin=-diff_max, vmax=diff_max, cmap='coolwarm')
            else:
                # Densities all uniformly share the exact same scale and color map
                plot_2d_hist(axs3[r, c], fig3, curr_data[:, idx_x], curr_data[:, idx_y], curr_weights, 
                             [range_x, range_y], row_title if c==0 else "", lbl_x, lbl_y, vmin=c_vmin, vmax=c_vmax, cmap='viridis')
                
    plt.tight_layout()

def plot_sliced_y_grids(x_raw, X_ctrl, y_data, w_sig, bounds_list, 
                        f_sig_cond, f_bkg_1d, Y_MIN, Y_MAX, device, 
                        is_iter_0=False, iter_label="Iter 0", extractor_obj=None):
    
    configs = [(0, r"$\cos\theta$", bounds_list[0]), 
               (1, r"$\phi$", bounds_list[1]), 
               (2, r"$\Phi$", bounds_list[2])]
               
    y_grid_diag = np.linspace(Y_MIN, Y_MAX, 60)
    fb_base = f_bkg_1d(y_grid_diag)
    
    y_gl_cpu = None
    if getattr(extractor_obj, 'correction_grid', None) is not None:
        nodes, _ = np.polynomial.legendre.leggauss(extractor_obj.n_nodes)
        y_gl_cpu = 0.5 * (extractor_obj.y_max - extractor_obj.y_min) * nodes + 0.5 * (extractor_obj.y_max + extractor_obj.y_min)
    
    for var_idx, var_name, var_bounds in configs:
        fig, axes = plt.subplots(5, 2, figsize=(14, 20))
        fig.suptitle(f"Multidimensional Validation ({iter_label}): Slices in {var_name}", fontsize=18)
        axes = axes.flatten()
        slice_bins = np.linspace(var_bounds[0], var_bounds[1], 11)
        
        for i in range(10):
            ax = axes[i]
            mask = (x_raw[:, var_idx] >= slice_bins[i]) & (x_raw[:, var_idx] < slice_bins[i+1])
            n_events = np.sum(mask)
            if n_events == 0:
                ax.set_title(rf"{slice_bins[i]:.2f} $\leq$ {var_name} < {slice_bins[i+1]:.2f} (Empty)")
                continue
                
            counts, hist_bins, _ = ax.hist(y_data[mask], bins=30, histtype='step', color='black', lw=2, label='Data')
            bin_width = hist_bins[1] - hist_bins[0]
            idx_slice = np.random.choice(np.where(mask)[0], min(500, n_events), replace=False)
            
            X_ctrl_slice = X_ctrl[idx_slice]
            
            fs_slice = np.zeros_like(y_grid_diag)
            fb_slice = np.zeros_like(y_grid_diag)
            
            for j, yg in enumerate(y_grid_diag):
                fs_slice[j] = np.mean(f_sig_cond(X_ctrl_slice, np.full(len(idx_slice), yg)))
                
                if is_iter_0 or extractor_obj is None:
                    fb_slice[j] = fb_base[j]
                else:
                    X_slice_eval = np.column_stack([X_ctrl_slice, np.full((len(idx_slice), 1), yg)])
                    X_slice_t = torch.tensor(X_slice_eval, dtype=torch.float32).to(device)
                    
                    with torch.no_grad():
                        R_dnn = torch.exp(torch.clamp(extractor_obj.model_A(X_slice_t), -15, 15)).cpu().numpy().flatten()
                    
                    if getattr(extractor_obj, 'gbdt_A', None) is not None:
                        if extractor_obj.gbdt_mode == 'XGB_GPU':
                            p_g_raw = extractor_obj.gbdt_A.predict_proba(X_slice_t)[:, 1]
                            if isinstance(p_g_raw, torch.Tensor): p_g_raw = p_g_raw.cpu().numpy()
                            p_gbdt = np.clip(p_g_raw, 1e-7, 1-1e-7)
                            R_gbdt = p_gbdt / (1 - p_gbdt)
                        elif extractor_obj.gbdt_mode == 'SKLEARN':
                            p_gbdt = np.clip(extractor_obj.gbdt_A.predict_proba(X_slice_eval)[:, 1], 1e-7, 1-1e-7)
                            R_gbdt = p_gbdt / (1 - p_gbdt)
                        else:
                            R_gbdt = np.ones(len(idx_slice))
                    else:
                        R_gbdt = np.ones(len(idx_slice))

                    C_y = 1.0
                    if y_gl_cpu is not None:
                        C_y = np.interp(yg, y_gl_cpu, extractor_obj.correction_grid)

                    Z_b_slice = np.zeros(len(idx_slice))
                    for yg_int in y_grid_diag:
                        X_int_eval = np.column_stack([X_ctrl_slice, np.full((len(idx_slice), 1), yg_int)])
                        X_int_t = torch.tensor(X_int_eval, dtype=torch.float32).to(device)
                        
                        with torch.no_grad():
                            R_d_int = torch.exp(torch.clamp(extractor_obj.model_A(X_int_t), -15, 15)).cpu().numpy().flatten()
                        
                        if getattr(extractor_obj, 'gbdt_A', None) is not None:
                            if extractor_obj.gbdt_mode in ['XGB_GPU', 'SKLEARN']:
                                p_int_raw = extractor_obj.gbdt_A.predict_proba(X_int_t)[:, 1] if extractor_obj.gbdt_mode == 'XGB_GPU' else extractor_obj.gbdt_A.predict_proba(X_int_eval)[:, 1]
                                if isinstance(p_int_raw, torch.Tensor): p_int_raw = p_int_raw.cpu().numpy()
                                p_g_int = np.clip(p_int_raw, 1e-7, 1-1e-7)
                                R_g_int = p_g_int / (1 - p_g_int)
                            else:
                                R_g_int = np.ones(len(idx_slice))
                        else:
                            R_g_int = np.ones(len(idx_slice))
                            
                        C_y_int = 1.0
                        if y_gl_cpu is not None:
                            C_y_int = np.interp(yg_int, y_gl_cpu, extractor_obj.correction_grid)

                        Z_b_slice += (f_bkg_1d(np.full(len(idx_slice), yg_int)) * R_d_int * R_g_int * C_y_int) * (y_grid_diag[1] - y_grid_diag[0])

                    fb_slice[j] = np.mean((fb_base[j] * R_dnn * R_gbdt * C_y) / np.maximum(Z_b_slice, 1e-10))

            N_s_slice, N_b_slice = np.sum(w_sig[mask]), np.sum(1 - w_sig[mask])
            
            sig_pdf_scaled = N_s_slice * fs_slice * bin_width
            bkg_pdf_scaled = N_b_slice * fb_slice * bin_width
            tot_slice = sig_pdf_scaled + bkg_pdf_scaled
            
            ax.plot(y_grid_diag, tot_slice, 'r--', lw=2, label='Total PDF')
            ax.plot(y_grid_diag, bkg_pdf_scaled, 'b:', lw=2, label='Bkg PDF')
            ax.plot(y_grid_diag, sig_pdf_scaled, 'g-.', lw=1, alpha=0.8, label='Sig PDF')
            
            ax.set_title(rf"{slice_bins[i]:.2f} $\leq$ {var_name} < {slice_bins[i+1]:.2f} (N={n_events})")
            if i == 0: ax.legend(loc='best', fontsize='small')
        
        plt.tight_layout()
        plt.pause(0.5)

def get_chi2(x_raw, w_s, x_true, bounds=None, bins=50):
    chi2s = []
    n_dims = x_raw.shape[1] if len(x_raw.shape) > 1 else 1
    x_r = x_raw.reshape(-1, n_dims)
    x_t = x_true.reshape(-1, n_dims)
    
    for i in range(n_dims):
        b_min, b_max = bounds[i] if (bounds and i < len(bounds)) else (np.min(x_r[:, i]), np.max(x_r[:, i]))
        
        hist_ext, bin_edges = np.histogram(x_r[:, i], bins=bins, range=(b_min, b_max), weights=w_s)
        hist_true, _ = np.histogram(x_t[:, i], bins=bin_edges)
        
        if np.sum(hist_ext) > 0:
            hist_ext = hist_ext * (np.sum(hist_true) / np.sum(hist_ext))
            
        mask = hist_true > 0
        if np.sum(mask) == 0:
            chi2s.append(0.0)
        else:
            chi2 = np.sum((hist_ext[mask] - hist_true[mask])**2 / hist_true[mask]) / np.sum(mask)
            chi2s.append(chi2)
    return chi2s

def plot_dynamic_iteration(x_raw, w_s, x_true, bounds, labels, it, chi2s, y_grid_cpu=None, correction_grid=None, y_raw=None, y_bounds=None):
    plt.ion() 
    try:
        from IPython.display import clear_output
        clear_output(wait=True)
    except ImportError:
        pass
    
    n_dims = x_raw.shape[1] if len(x_raw.shape) > 1 else 1
    x_r = x_raw.reshape(-1, n_dims)
    x_t = x_true.reshape(-1, n_dims)
    
    n_cols = n_dims + (1 if y_raw is not None else 0) + (1 if correction_grid is not None else 0)
        
    plt.clf() 
    fig = plt.gcf()
    fig.set_size_inches(5.0 * n_cols, 4.5)
    col_idx = 1
    
    for i in range(n_dims):
        plt.subplot(1, n_cols, col_idx)
        col_idx += 1
        b_min, b_max = bounds[i] if (bounds and i < len(bounds)) else (np.min(x_r[:, i]), np.max(x_r[:, i]))
        lbl = labels[i] if (labels and i < len(labels)) else f"Var {i}"
        
        plt.hist(x_r[:, i], bins=40, range=(b_min, b_max), color='gray', alpha=0.3, label='Total Data')
        plt.hist(x_t[:, i], bins=40, range=(b_min, b_max), histtype='step', color='black', lw=2, label='True Signal')
        plt.hist(x_r[:, i], bins=40, range=(b_min, b_max), weights=w_s, histtype='step', color='blue', lw=2, label='ND sWeights')
        plt.hist(x_r[:, i], bins=40, range=(b_min, b_max), weights=(1.0 - w_s), histtype='step', color='red', ls='--', lw=2, label='ND sWeights Bkg')
        
        plt.title(f"{lbl} | Iter {it} | $\chi^2$: {chi2s[i]:.1f}")
        plt.legend(loc='upper right', fontsize='small')
        
    if y_raw is not None and y_bounds is not None:
        plt.subplot(1, n_cols, col_idx)
        col_idx += 1
        plt.hist(y_raw, bins=50, range=y_bounds, color='gray', alpha=0.3, label='Total Data')
        plt.hist(y_raw, bins=50, range=y_bounds, weights=w_s, histtype='step', color='blue', lw=2, label='ND sWeights')
        plt.hist(y_raw, bins=50, range=y_bounds, weights=(1.0 - w_s), histtype='step', color='red', ls='--', lw=2, label='ND sWeights Bkg')
        
        plt.title("Discriminant Variable (y)")
        plt.legend(loc='upper right', fontsize='small')
        
    if correction_grid is not None and y_grid_cpu is not None:
        plt.subplot(1, n_cols, col_idx)
        plt.plot(y_grid_cpu, correction_grid, marker='o', color='red', lw=2)
        plt.axhline(1.0, color='black', linestyle='--')
        plt.title("Background Correction R(y)")
        plt.xlabel("y")
        
    plt.tight_layout()
    plt.pause(0.1)
