# core/quadrature.py
import numpy as np
import torch

def get_gauss_legendre_grid(y_min, y_max, n_nodes=20, device='cuda'):
    """
    Generates deterministic Gauss-Legendre quadrature nodes and weights 
    mapped exactly to the physical mass window.
    """
    x_standard, w_standard = np.polynomial.legendre.leggauss(n_nodes)
    
    y_nodes = 0.5 * (x_standard + 1.0) * (y_max - y_min) + y_min
    dy_weights = 0.5 * (y_max - y_min) * w_standard
    
    return (torch.tensor(y_nodes, dtype=torch.float32, device=device),
            torch.tensor(dy_weights, dtype=torch.float32, device=device))


def get_gauss_legendre_grid_2d(y1_bounds, y2_bounds, n_nodes=(10, 10), device='cuda'):
    """
    Generates a 2D tensor-product Gauss-Legendre quadrature grid 
    and corresponding area weights mapped to the physical 2D window.
    """
    if isinstance(n_nodes, int):
        n1, n2 = n_nodes, n_nodes
    else:
        n1, n2 = int(n_nodes[0]), int(n_nodes[1])

    y1_t, dy1_t = get_gauss_legendre_grid(y1_bounds[0], y1_bounds[1], n_nodes=n1, device=device)
    y2_t, dy2_t = get_gauss_legendre_grid(y2_bounds[0], y2_bounds[1], n_nodes=n2, device=device)

    # Use 'ij' indexing to maintain matrix-style Cartesian coordinates
    Y1, Y2 = torch.meshgrid(y1_t, y2_t, indexing='ij')
    W1, W2 = torch.meshgrid(dy1_t, dy2_t, indexing='ij')

    y_grid_2d_t = torch.stack([Y1.reshape(-1), Y2.reshape(-1)], dim=1)
    dA_weights_t = (W1 * W2).reshape(-1)

    return y1_t, y2_t, y_grid_2d_t, dA_weights_t


def torch_interp1d(x, xp, yp):
    """Fast GPU linear 1D interpolation mapping."""
    x = x.contiguous()
    idx = torch.searchsorted(xp, x)
    idx = torch.clamp(idx, 1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = yp[idx - 1], yp[idx]
    
    w_interp = (x - x0) / torch.clamp(x1 - x0, min=1e-12)
    return y0 + w_interp * (y1 - y0)


def torch_interp2d(y1, y2, y1_grid_t, y2_grid_t, V_grid_t):
    """
    Fast GPU bilinear interpolation on a 2D tensor-product grid.
    Handles both shared 2D surfaces (M1, M2) and batched per-event surfaces (N, M1, M2).
    """
    M1 = len(y1_grid_t)
    M2 = len(y2_grid_t)

    # Protect against searchsorted contiguous warnings when slicing data
    y1 = y1.contiguous()
    y2 = y2.contiguous()

    idx1 = torch.clamp(torch.searchsorted(y1_grid_t, y1), 1, M1 - 1)
    idx2 = torch.clamp(torch.searchsorted(y2_grid_t, y2), 1, M2 - 1)

    y1_0, y1_1 = y1_grid_t[idx1 - 1], y1_grid_t[idx1]
    y2_0, y2_1 = y2_grid_t[idx2 - 1], y2_grid_t[idx2]

    t1 = torch.clamp((y1 - y1_0) / torch.clamp(y1_1 - y1_0, min=1e-12), 0.0, 1.0)
    t2 = torch.clamp((y2 - y2_0) / torch.clamp(y2_1 - y2_0, min=1e-12), 0.0, 1.0)

    if V_grid_t.ndim == 2:
        v00 = V_grid_t[idx1 - 1, idx2 - 1]
        v01 = V_grid_t[idx1 - 1, idx2]
        v10 = V_grid_t[idx1, idx2 - 1]
        v11 = V_grid_t[idx1, idx2]
    else:
        row = torch.arange(len(y1), device=V_grid_t.device)
        v00 = V_grid_t[row, idx1 - 1, idx2 - 1]
        v01 = V_grid_t[row, idx1 - 1, idx2]
        v10 = V_grid_t[row, idx1, idx2 - 1]
        v11 = V_grid_t[row, idx1, idx2]

    return (
        (1.0 - t1) * (1.0 - t2) * v00
        + (1.0 - t1) * t2 * v01
        + t1 * (1.0 - t2) * v10
        + t1 * t2 * v11
    )
