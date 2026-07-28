import numpy as np
import torch

def get_gauss_legendre_grid(y_min, y_max, n_nodes=20, device='cuda'):
    """
    Generates deterministic Gauss-Legendre quadrature nodes and weights 
    mapped exactly to the physical mass window.
    """
    x_standard, w_standard = np.polynomial.legendre.leggauss(n_nodes)
    
    y_nodes = 0.5 * (x_standard + 1) * (y_max - y_min) + y_min
    dy_weights = 0.5 * (y_max - y_min) * w_standard
    
    return (torch.tensor(y_nodes, dtype=torch.float32, device=device),
            torch.tensor(dy_weights, dtype=torch.float32, device=device))
