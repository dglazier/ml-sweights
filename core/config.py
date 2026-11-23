# core/config.py

class MLSWeightConfig:
    def __init__(self,
                 disc_bounds=None,
                 n_nodes=50,
                 device='cuda',
                 gbdt_mode='XGB_GPU',
                 use_relu=True,
                 reinstate_nn=False,
                 marginal_alpha=1.0,
                 dynamic_alpha=True,
                 alpha_mode="inner_opt",
                 alpha_bounds=None,
                 alpha_pullback=False,
                 correction_mode="penalized_spline",
                 peak_curv_penalty=3.0,
                 tune_ry_smoothing=False,
                 inner_est_iters=1,
                 use_gbdt=True,
                 dnn_epochs=50,
                 ml_transform=None,
                 loss_fn='BCE',
                 logit_l2_penalty=0.0,
                 yield_boost=0.0,
                 ry_smoothing=1.5,
                 n_folds=1,
                 extended_cows_derivatives=1,
                 convergence_mode="self_consistency",
                 burn_in_ema_shift=0.005,
                 explicit_burn_in_iters=-1,
                 adaptive_neq_thermostat=True,
                 factorization_p_value=1.0,
                 trust_valid_1d_anchor=True,
                 dnn_optimizer='Adam',
                 max_iters=50,
                 tol=0.002,
                 omega_init=2.0,
                 omega_decay_iters=30.0):
        
        self.disc_bounds = disc_bounds if disc_bounds else (0.73, 0.85)
        self.n_nodes = n_nodes
        self.device = device
        self.gbdt_mode = gbdt_mode
        self.use_relu = use_relu
        self.reinstate_nn = reinstate_nn
        self.marginal_alpha = marginal_alpha
        self.dynamic_alpha = dynamic_alpha
        self.alpha_mode = alpha_mode
        self.alpha_bounds = alpha_bounds
        self.alpha_pullback = alpha_pullback
        self.correction_mode = correction_mode
        self.peak_curv_penalty = peak_curv_penalty
        self.tune_ry_smoothing = tune_ry_smoothing
        self.inner_est_iters = inner_est_iters
        self.use_gbdt = use_gbdt
        self.dnn_epochs = dnn_epochs
        self.ml_transform = ml_transform
        self.loss_fn = loss_fn
        self.logit_l2_penalty = logit_l2_penalty
        self.yield_boost = yield_boost
        self.ry_smoothing = ry_smoothing
        self.n_folds = n_folds
        self.extended_cows_derivatives = extended_cows_derivatives
        self.convergence_mode = convergence_mode
        self.burn_in_ema_shift = burn_in_ema_shift
        self.explicit_burn_in_iters = explicit_burn_in_iters
        self.adaptive_neq_thermostat = adaptive_neq_thermostat
        self.factorization_p_value = factorization_p_value
        self.trust_valid_1d_anchor = trust_valid_1d_anchor
        self.dnn_optimizer = dnn_optimizer
        self.max_iters = max_iters
        self.tol = tol
        self.omega_init = omega_init
        self.omega_decay_iters = omega_decay_iters# core/config.py
from dataclasses import dataclass
from typing import Tuple, Any, Optional, Union, List




@dataclass
class FlowConfig:
    """Configuration for the Deep Generative Normalizing Flow."""
    flow_mode: str = "conditional"
    smear_mode: str = "mc"
    init_shift: float = 0.0
    init_smear: float = 0.0
    init_scale: float = 1.0
    burn_in_iters: int = 3
    tune_interval: int = 3
    ema_alpha: float = 0.3
    gh_nodes: int = 9
    k_samples_opt: int = 5
    k_samples_eval: int = 15
    prior_shift: float = 5.0
    prior_scale: float = 5.0
