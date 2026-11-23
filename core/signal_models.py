# core/signal_models.py
import numpy as np
import scipy.ndimage
import scipy.interpolate
import torch
import logging

try:
    import zuko
except ImportError:
    zuko = None

from .flows import _detect_tiled_period_1d, _detect_tiled_period_2d

logger = logging.getLogger(__name__)


class MCSpline1D:
    """
    Fast 1D Cubic MC Spline with two-way (shift, scale) domain adaptation
    using scale-covariant Gaussian moment matching (zero arbitrary subtraction constants).
    """
    def __init__(self, mc_y, y_min, y_max, bins=100, smooth=1.0, burn_in_iters=1, freeze_domain=False):
        hist, edges = np.histogram(mc_y, bins=bins, range=(y_min, y_max), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        hist_sm = scipy.ndimage.gaussian_filter1d(hist, sigma=smooth)
        self.spline = scipy.interpolate.interp1d(
            centers, hist_sm, kind='cubic', fill_value=(hist_sm[0], hist_sm[-1]), bounds_error=False
        )
        self.mc_mu = float(np.mean(mc_y))
        self.mc_sig = float(max(np.std(mc_y), 1e-6))
        self.mc_var = float(np.var(mc_y))

        # Windowed reference variance on MC (standard 2.5*sigma window)
        win_mc = np.exp(-0.5 * ((mc_y - self.mc_mu) / max(2.5 * self.mc_sig, 1e-4)) ** 2)
        self.mc_var_win = float(np.sum(win_mc * (mc_y - self.mc_mu) ** 2) / np.maximum(np.sum(win_mc), 1e-12))

        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.shift = 0.0
        self.scale = 1.0
        self.smear = 0.0
        self.burn_in_iters = burn_in_iters
        self.freeze_domain = freeze_domain
        self.iters = 0

    def _eval_core(self, y_vec):
        y_eq = self.mc_mu + (y_vec - (self.mc_mu + self.shift)) / self.scale
        return np.clip(self.spline(y_eq) / self.scale, 1e-10, None)

    def __call__(self, x, y):
        y_flat = np.asarray(y, dtype=np.float64).reshape(-1)
        m_grid = _detect_tiled_period_1d(y_flat)
        if m_grid is not None:
            return np.tile(self._eval_core(y_flat[:m_grid]), len(y_flat) // m_grid)
        return self._eval_core(y_flat)

    def update(self, x, y, w_eff):
        if self.freeze_domain:
            return
        self.iters += 1
        if self.iters >= self.burn_in_iters:
            y_flat = np.asarray(y, dtype=np.float64).reshape(-1)
            w_flat = np.asarray(w_eff, dtype=np.float64).reshape(-1)

            tune_steps = self.iters - self.burn_in_iters
            anneal = 1.0 if tune_steps == 0 else max(0.05, 0.5 * (0.85 ** tune_steps))

            peak_loc = self.mc_mu + self.shift
            # Scale-covariant window width: scales with self.scale so d_var / mc_var_win == scale^2
            win_width = max(2.5 * self.scale * self.mc_sig, 1e-4)

            for _ in range(3):
                window = np.exp(-0.5 * ((y_flat - peak_loc) / win_width) ** 2)
                w_win = w_flat * window
                w_sum = float(np.sum(w_win))
                if w_sum > 1e-10:
                    peak_loc = float(np.sum(w_win * y_flat) / w_sum)

            if w_sum > 1e-10:
                d_mu = peak_loc
                target_shift = d_mu - self.mc_mu
                self.shift = float(self.shift + anneal * (target_shift - self.shift))

                # Update scale only after Iteration 1 (tune_steps >= 1) once w_s has been
                # evaluated with the actual MC template rather than the crude Gaussian anchor
                if tune_steps >= 1:
                    d_var = float(np.sum(w_win * (y_flat - d_mu) ** 2) / w_sum)
                    if d_var > 1e-10:
                        target_scale = float(np.clip(np.sqrt(d_var / max(self.mc_var_win, 1e-12)), 0.50, 2.00))
                        self.scale = float(np.clip(self.scale + 0.5 * anneal * (target_scale - self.scale), 0.50, 2.00))
                        self.smear = float(self.mc_sig * np.sqrt(max(0.0, self.scale ** 2 - 1.0)))

                logger.info(
                    f"  [Domain Tune] Spline1D updated -> Shift: {self.shift:+.5f} | "
                    f"Scale: {self.scale:.4f} (Smear eq: {self.smear:.5f})"
                )

    def get_state(self):
        return {'shift': self.shift, 'scale': self.scale, 'smear': self.smear, 'iters': self.iters}

    def load_state(self, s):
        if s is not None:
            self.shift = float(s['shift'])
            self.scale = float(s.get('scale', 1.0))
            self.smear = float(s.get('smear', 0.0))
            self.iters = int(s['iters'])

class MCSpline2D:
    """
    Fast Bivariate Bicubic MC Spline with closed-form Gaussian resolution smearing.
    Convolves a 64x64 grid via scipy.ndimage.gaussian_filter (<0.1ms) and evaluates
    with automatic 2D quadrature grid tiling detection.
    Incorporates dynamic annealing to prevent late-stage positive feedback loops.
    """
    def __init__(self, mc_y1, mc_y2, y1_bounds, y2_bounds, bins=64, smooth=1.0, burn_in_iters=3, freeze_domain=False):
        self.y1_b, self.y2_b = y1_bounds, y2_bounds
        self.bins = bins
        hist, e1, e2 = np.histogram2d(mc_y1, mc_y2, bins=bins, range=[y1_bounds, y2_bounds], density=True)
        self.c1 = 0.5 * (e1[:-1] + e1[1:])
        self.c2 = 0.5 * (e2[:-1] + e2[1:])
        self.base_hist = scipy.ndimage.gaussian_filter(hist, sigma=smooth)
        self.spline = scipy.interpolate.RectBivariateSpline(self.c1, self.c2, self.base_hist)
        
        self.mc_mu1, self.mc_mu2 = float(np.mean(mc_y1)), float(np.mean(mc_y2))
        self.mc_sig1, self.mc_sig2 = float(np.std(mc_y1)), float(np.std(mc_y2))
        self.mc_var1, self.mc_var2 = float(np.var(mc_y1)), float(np.var(mc_y2))

        win_mc = np.exp(
            -0.5 * ((mc_y1 - self.mc_mu1) / max(2.5 * self.mc_sig1, 1e-4))**2
            - 0.5 * ((mc_y2 - self.mc_mu2) / max(2.5 * self.mc_sig2, 1e-4))**2
        )
        win_mc_sum = max(float(np.sum(win_mc)), 1e-12)
        self.mc_var1_win = float(np.sum(win_mc * (mc_y1 - self.mc_mu1)**2) / win_mc_sum)
        self.mc_var2_win = float(np.sum(win_mc * (mc_y2 - self.mc_mu2)**2) / win_mc_sum)

        self.d_mu1, self.d_mu2, self.smear1, self.smear2 = 0.0, 0.0, 0.0, 0.0
        self.burn_in_iters, self.freeze_domain, self.iters = burn_in_iters, freeze_domain, 0

    def _rebuild_spline(self):
        bw1 = (self.y1_b[1] - self.y1_b[0]) / self.bins
        bw2 = (self.y2_b[1] - self.y2_b[0]) / self.bins
        s1_pix = self.smear1 / bw1 if self.smear1 > 1e-5 else 0.0
        s2_pix = self.smear2 / bw2 if self.smear2 > 1e-5 else 0.0
        if s1_pix > 0.0 or s2_pix > 0.0:
            smeared_grid = scipy.ndimage.gaussian_filter(self.base_hist, sigma=[s1_pix, s2_pix])
        else:
            smeared_grid = self.base_hist
        self.spline = scipy.interpolate.RectBivariateSpline(self.c1, self.c2, smeared_grid)

    def _eval_core(self, y1_vec, y2_vec):
        val = self.spline(y1_vec - self.d_mu1, y2_vec - self.d_mu2, grid=False)
        return np.clip(val, 1e-10, None)

    def __call__(self, x, y1, y2):
        y1_flat = np.asarray(y1, dtype=np.float64).reshape(-1)
        y2_flat = np.asarray(y2, dtype=np.float64).reshape(-1)
        m_grid = _detect_tiled_period_2d(y1_flat, y2_flat)
        if m_grid is not None:
            return np.tile(self._eval_core(y1_flat[:m_grid], y2_flat[:m_grid]), len(y1_flat) // m_grid)
        return self._eval_core(y1_flat, y2_flat)

    def update(self, x, y1, y2, w_eff):
        if self.freeze_domain:
            return
        self.iters += 1
        if self.iters >= self.burn_in_iters:
            tune_steps = self.iters - self.burn_in_iters
            anneal = max(0.01, 0.5 * (0.90 ** tune_steps))

            p1 = self.mc_mu1 + self.d_mu1
            p2 = self.mc_mu2 + self.d_mu2
            w1_width = max(2.5 * self.mc_sig1, 1e-4)
            w2_width = max(2.5 * self.mc_sig2, 1e-4)
            window = np.exp(-0.5 * ((y1 - p1) / w1_width)**2 - 0.5 * ((y2 - p2) / w2_width)**2)
            w_win = w_eff * window
            w_sum = float(np.sum(w_win))
            if w_sum > 1e-10:
                d_mu1 = float(np.sum(w_win * y1) / w_sum)
                d_mu2 = float(np.sum(w_win * y2) / w_sum)
                d_v1 = float(np.sum(w_win * (y1 - d_mu1)**2) / w_sum)
                d_v2 = float(np.sum(w_win * (y2 - d_mu2)**2) / w_sum)
                
                self.d_mu1 = float(self.d_mu1 + anneal * ((d_mu1 - self.mc_mu1) - self.d_mu1))
                self.d_mu2 = float(self.d_mu2 + anneal * ((d_mu2 - self.mc_mu2) - self.d_mu2))

                new_sm1 = float(np.sqrt(max(0.0, d_v1 - self.mc_var1_win)))
                new_sm2 = float(np.sqrt(max(0.0, d_v2 - self.mc_var2_win)))
                self.smear1 = float(np.clip(self.smear1 + anneal * (new_sm1 - self.smear1), 0.0, 0.5 * self.mc_sig1))
                self.smear2 = float(np.clip(self.smear2 + anneal * (new_sm2 - self.smear2), 0.0, 0.5 * self.mc_sig2))
                self._rebuild_spline()
                logger.info(
                    f"  [2D Domain Tune] Spline2D -> Sh1: {self.d_mu1:+.5f}, Sm1: {self.smear1:.5f} | "
                    f"Sh2: {self.d_mu2:+.5f}, Sm2: {self.smear2:.5f}"
                )

    def get_state(self):
        return {'d_mu1': self.d_mu1, 'd_mu2': self.d_mu2, 'smear1': self.smear1, 'smear2': self.smear2, 'iters': self.iters}

    def load_state(self, s):
        self.d_mu1, self.d_mu2 = s['d_mu1'], s['d_mu2']
        self.smear1, self.smear2 = s['smear1'], s['smear2']
        self.iters = s['iters']
        self._rebuild_spline()


class UnconditionalNDFlow:
    """
    Unconditional Normalizing Flow trained on Cholesky-whitened y_MC in R^K (context=0),
    avoiding MC angular covariate shift. Adapts to data via w_eff-weighted affine
    transformation and supports both (x, y_nd) and (x, y1, y2) call signatures.
    Incorporates dynamic annealing to prevent late-stage positive feedback loops.
    """
    def __init__(self, mc_y_nd, device='cuda', epochs=40, burn_in_iters=3, freeze_domain=False):
        mc_y_nd = np.column_stack([mc_y_nd]) if mc_y_nd.ndim == 1 else np.asarray(mc_y_nd, dtype=np.float64)
        self.K = mc_y_nd.shape[1]
        self.device = device
        self.mc_mu = np.mean(mc_y_nd, axis=0)
        self.mc_sig = np.std(mc_y_nd, axis=0) + 1e-8
        self.mc_cov = np.atleast_2d(np.cov(mc_y_nd, rowvar=False)) + np.eye(self.K) * 1e-8
        self.L_mc = np.linalg.cholesky(self.mc_cov)
        self.L_mc_inv = np.linalg.inv(self.L_mc)

        # Peak-windowed reference covariance on MC
        z_mc_win = (mc_y_nd - self.mc_mu) / (2.5 * self.mc_sig)
        win_mc = np.exp(-0.5 * np.sum(z_mc_win**2, axis=1))
        win_sum = max(float(np.sum(win_mc)), 1e-12)
        diff_mc = mc_y_nd - self.mc_mu
        self.mc_cov_win = (diff_mc.T @ (win_mc[:, None] * diff_mc)) / win_sum + np.eye(self.K) * 1e-8
        
        if zuko is not None:
            # Train NSF in whitened O(1) coordinates so spline knots in [-5, 5] have full resolution
            u_mc = (mc_y_nd - self.mc_mu) @ self.L_mc_inv.T
            self.flow = zuko.flows.NSF(features=self.K, context=0, hidden_features=[64, 64], bins=8).to(device)
            loader = torch.utils.data.DataLoader(torch.tensor(u_mc, dtype=torch.float32), batch_size=16384, shuffle=True)
            opt = torch.optim.Adam(self.flow.parameters(), lr=1e-3)
            self.flow.train()
            for _ in range(epochs):
                for b in loader:
                    opt.zero_grad(set_to_none=True)
                    loss = -self.flow().log_prob(b.to(device)).mean()
                    loss.backward()
                    opt.step()
            self.flow.eval()
        else:
            self.flow = None

        self.A_mat = np.eye(self.K)
        self.b_vec = np.zeros(self.K)
        self.burn_in_iters = burn_in_iters
        self.freeze_domain = freeze_domain
        self.iters = 0

    @staticmethod
    def _pack_y(y_args):
        if len(y_args) == 1:
            arr = np.asarray(y_args[0], dtype=np.float64)
            return arr.reshape(-1, 1) if arr.ndim == 1 else arr
        return np.column_stack([np.asarray(a, dtype=np.float64).reshape(-1) for a in y_args])

    def _eval_core(self, y_nd):
        y_eq = (y_nd - self.b_vec) @ self.A_mat.T
        u_eq = (y_eq - self.mc_mu) @ self.L_mc_inv.T
        det_j = abs(float(np.linalg.det(self.A_mat) * np.linalg.det(self.L_mc_inv)))
        if self.flow is not None:
            with torch.no_grad():
                u_t = torch.tensor(u_eq, dtype=torch.float32, device=self.device)
                log_p = self.flow().log_prob(u_t).cpu().numpy()
            return np.clip(np.exp(log_p) * det_j, 1e-10, None)
        else:
            quad = np.sum(u_eq**2, axis=1)
            norm_c = (2.0 * np.pi) ** (0.5 * self.K)
            return np.clip(np.exp(-0.5 * quad) / norm_c * det_j, 1e-10, None)

    def __call__(self, x, *y_args):
        y_nd = self._pack_y(y_args)
        if self.K == 1:
            m_grid = _detect_tiled_period_1d(y_nd[:, 0])
            if m_grid is not None:
                return np.tile(self._eval_core(y_nd[:m_grid]), len(y_nd) // m_grid)
        elif self.K == 2:
            m_grid = _detect_tiled_period_2d(y_nd[:, 0], y_nd[:, 1])
            if m_grid is not None:
                return np.tile(self._eval_core(y_nd[:m_grid]), len(y_nd) // m_grid)
        return self._eval_core(y_nd)

    def update(self, x, *args):
        if self.freeze_domain:
            return
        w_eff = np.asarray(args[-1], dtype=np.float64).reshape(-1)
        y_nd = self._pack_y(args[:-1])
        self.iters += 1
        if self.iters >= self.burn_in_iters:
            tune_steps = self.iters - self.burn_in_iters
            anneal = max(0.01, 0.5 * (0.90 ** tune_steps))

            peak_loc = self.mc_mu + self.b_vec
            z_win = (y_nd - peak_loc) / (2.5 * self.mc_sig)
            window = np.exp(-0.5 * np.sum(z_win**2, axis=1))
            w_win = w_eff * window
            w_sum = float(np.sum(w_win))
            if w_sum > 1e-10:
                d_mu = np.sum(w_win[:, None] * y_nd, axis=0) / w_sum
                diff = y_nd - d_mu
                d_cov_win = (diff.T @ (w_win[:, None] * diff)) / w_sum
                delta_cov = d_cov_win - self.mc_cov_win
                # Project delta_cov onto PSD cone so resolution smearing is non-negative
                eigvals, eigvecs = np.linalg.eigh(delta_cov)
                eigvals_pos = np.clip(eigvals, 0.0, 0.25 * (self.mc_sig**2))
                target_cov = self.mc_cov + (eigvecs * eigvals_pos) @ eigvecs.T
                L_data = np.linalg.cholesky(target_cov)
                target_A = self.L_mc @ np.linalg.inv(L_data)
                target_b = d_mu - self.mc_mu

                self.A_mat = self.A_mat + anneal * (target_A - self.A_mat)
                self.b_vec = self.b_vec + anneal * (target_b - self.b_vec)

    def get_state(self):
        return {'A': self.A_mat.copy(), 'b': self.b_vec.copy(), 'iters': self.iters}

    def load_state(self, s):
        self.A_mat, self.b_vec, self.iters = s['A'].copy(), s['b'].copy(), s['iters']
