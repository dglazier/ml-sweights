# core/__init__.py
from .config import FlowConfig, MLSWeightConfig
from .quadrature import (
    get_gauss_legendre_grid, 
    get_gauss_legendre_grid_2d, 
    torch_interp1d, 
    torch_interp2d
)
from .anchor import fit_1d_anchor, fit_2d_anchor
from .baselines import (
    calculate_1d_sweights,
    calculate_cows_lite,
    calculate_cows_full_official,
    calculate_cows_full_derivatives
)
from .networks import (
    PureDNN, 
    train_dnn, 
    predict_in_batches, 
    map_trig_topology,
    SignedXGBClassifier  # <--- Add this if you want it exposed at the package level
)
from .signal_models import (
    MCSpline1D, 
    MCSpline2D, 
    UnconditionalNDFlow
)
#from .extractor_2d import MLSWeightExtractor2D
#from .sequential import SequentialMLSWeightExtractor
from .base_extractor import BaseMLSWeightExtractor

__all__ = [
    "FlowConfig",
    "MLSWeightConfig",
    "get_gauss_legendre_grid",
    "get_gauss_legendre_grid_2d",
    "fit_1d_anchor",
    "fit_2d_anchor",
    "calculate_1d_sweights",
    "calculate_cows_lite",
    "calculate_cows_full_official",
    "calculate_cows_full_derivatives",
    "PureDNN",
    "train_dnn",
    "predict_in_batches",
    "map_trig_topology",
    "SignedXGBClassifier", # <--- Added here
    "train_flow_model",
    "DynamicFlowWrapper",
    "calibrate_flow_model",
    "build_signal_flow",
    "build_domain_adapted_signal_flow",
    "build_2d_domain_adapted_signal_flow",
    "MCSpline1D",
    "MCSpline2D",
    "UnconditionalNDFlow",
    "BaseMLSWeightExtractor",
    "torch_interp1d",
    "torch_interp2d",
]
