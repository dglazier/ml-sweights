# Multidimensional ML-sWeights Extractor

## Overview

This repository provides a multidimensional, non-parametric Machine Learning framework designed to separate physical signal amplitudes from background processes in unbinned kinematic datasets. 

Standard analytical background subtraction tools—such as the classic 1D sPlot method—depend strictly on the assumption of **factorization** between the discriminating variable (typically invariant mass `y`) and the control variables (angular kinematics `x`). In modern high-precision experimental environments, physical acceptance boundaries, detector resolution stretching, and moving kinematic thresholds create severe, non-factorizable correlations.

To decouple these complex correlations without relying on rigid, global parametric assumptions, this algorithm combines three core components:
1. **Dynamic Signal Modeling (Splines & Flows):** Models dynamic, event-by-event signal resolution stretching and shape morphing initialized directly from 1D anchor fits, using either scale-covariant Splines or Conditional Normalizing Flows[cite: 13, 14].
2. **Sequential ML Cascades (PyTorch DNN + XGBoost):** Learns the continuous multidimensional background density ratio directly from data via the Density Ratio Trick, separating spatial priors from mass physics[cite: 22].
3. **Local Covariance Inversion (Local COWs):** Computes event-by-event local covariance matrices using Gauss-Legendre Quadrature, translating deep-learning probabilities back into mathematically pristine, multi-dimensional sWeights[cite: 22].

The resulting multi-dimensional sWeights naturally incorporate valid negative sideband weights, eliminating spatial over-subtraction while remaining fully compatible with standard sum-of-squares (`Sum W^2`) error propagation in flat trees or `RDataFrame` workflows.

---

## Directory Structure

```text
ml_sweights/
│
├── core/
│   ├── __init__.py
│   ├── base_extractor.py   # Core Expectation-Maximization Loop & Local COWs Inversion
│   ├── config.py           # Configuration classes (MLSWeightConfig, FlowConfig)
│   ├── flows.py            # Zuko-based Normalizing Flows (MCFlow1D, MCFlowND, CNFs)
│   ├── networks.py         # PyTorch DNNs, SignedXGBClassifier, and trigonometric mappers
│   ├── signal_models.py    # Fast exact representations (MCSpline1D, MCSpline2D, NDFlow)
│   └── quadrature.py       # Gauss-Legendre quadrature grids for local matrix integration
│
├── utils/
│   ├── __init__.py
│   └── diagnostics.py      # Chi2 evaluation, 1D/2D kinematic projection plotting
│
└── examples/
    ├── example_base.py         # Unified CLI executable for running all physics scenarios
    ├── omega_generators.py     # Central physics toy generator (Schilling-Wolf SDMEs)
    └── compare_cows_ml.py      # Direct benchmark: Standard 1D sPlot vs COWs vs ML-sWeights
```

---

## Installation & Requirements

The framework requires a Python 3.9+ environment with CUDA acceleration. Batched numerical quadrature and neural network updates rely heavily on GPU VRAM for high-throughput processing.

### Dependencies
```bash
pip install numpy scipy matplotlib torch scikit-learn xgboost zuko iminuit cupy-cuda12x
```

> **Note on GPU Execution:** Ensure `xgboost` is installed with CUDA support. The `BaseMLSWeightExtractor` defaults to `gbdt_mode='XGB_GPU'` to avoid PCIe bus bottlenecking during the Expectation-Maximization (EM) loop[cite: 22].

---

## The Unified Example Suite

The framework relies on a unified executable, `example_base.py`, to configure and deploy the EM loop against simulated photoproduction datasets. 

### Usage

```bash
python examples/example_base.py --signal-model <MODEL_TYPE> [OPTIONS]
```

### Signal Model Options (`--signal-model`)
The framework supports scalable models depending on the dimensionality and complexity of the resolution smearing[cite: 13, 14, 20]:
*   `spline_1d`: Fast 1D Cubic MC Spline with scale-covariant Gaussian moment matching for dynamic shifting and smearing[cite: 14].
*   `spline_2d`: Bivariate Bicubic MC Spline with closed-form Gaussian resolution smearing[cite: 14].
*   `flow_1d` / `flow_nd`: Unconditional Normalizing Flow trained on Cholesky-whitened coordinates[cite: 14].
*   `cond_flow_1d` / `cond_flow_nd`: Fully Conditional Normalizing Flows (CNF) adapting dynamically to event-by-event control kinematics `x` using a Neural Spline Flow[cite: 13].

---

## Configuration Guide (`BaseMLSWeightExtractor`)

The `BaseMLSWeightExtractor` class in `core/base_extractor.py` houses the primary Expectation-Maximization logic[cite: 22].

```python
from core import BaseMLSWeightExtractor

extractor = BaseMLSWeightExtractor(
    disc_bounds=(0.73, 0.85),
    n_nodes=64,
    device='cuda',
    gbdt_mode='XGB_GPU',
    dnn_epochs=50,
    warm_start=True,
    lock_yield=True,
    ry_smoothing=5.0,
    extended_cows_derivatives=1,
    omega_init=2.0,
    omega_decay_iters=25.0
)
```

### Key Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `disc_bounds` | `tuple` | *(required)* | `(Y_MIN, Y_MAX)` of the discriminating variable (e.g., invariant mass window)[cite: 22]. |
| `n_nodes` | `int` | `64` | Number of Gauss-Legendre quadrature nodes used for Local COWs matrix integrations[cite: 22]. |
| `lock_yield` | `bool` | `True` | Synchronizes the ML density ratio spatial odds to an algebraic yield target ($N_{s,\mathrm{cows}}$) using fixed-point root finding. Set to `False` to allow organic equilibrium drift[cite: 22]. |
| `ry_smoothing` | `float` | `5.0` | Controls the 2D Row-Wise Whittaker-Eilers Log-Spline Smoothing applied to the raw density ratio. Prevents collinearity hallucinations under the mass peak[cite: 22]. |
| `extended_cows_derivatives` | `int` | `1` | Configures the local polynomial basis expansion for Exact Derivative COWs Inversion (1 = $3 \times 3$ matrix, 2 = $4 \times 4$ matrix). Essential for eliminating spatial bias[cite: 22]. |
| `omega_init` | `float` | `2.0` | Initial momentum parameter for the Over-Relaxed EM Update, accelerating early phase-space exploration[cite: 22]. |
| `warm_start` | `bool` | `True` | Carries PyTorch DNN weights across EM iterations rather than cold-starting[cite: 22]. |

---

## How to Integrate Custom Experimental Data

To apply this framework to your own unbinned experimental datasets:

### Step 1: Topology Mapping
Map periodic angular variables to continuous trigonometric features to prevent neural networks from struggling at phase-space boundaries (`-pi` to `+pi`). The framework provides a built-in mapper[cite: 18]:

```python
from core.networks import map_trig_topology

# Explicitly map cos(theta) and periodic phi/Phi angles
X_ctrl_mapped = map_trig_topology(
    x_raw, 
    cos_theta_indices=(0,), 
    periodic_phi_indices=(1, 2)
)
```

### Step 2: Initialize Domain-Adapted Signal Model
Build a continuous, normalized representation of your idealized Monte Carlo signal. The framework handles dynamic shifts and scales internally[cite: 13, 20]:

```python
from core import build_domain_adapted_signal_flow

# Builds a Conditional Normalizing Flow (MCFlow1D) or Unconditional Flow
sig_pdf_cond = build_domain_adapted_signal_flow(
    Z_ctx_mc=X_ctrl_mc, 
    y_mc=y_mass_mc, 
    base_loc=0.782, 
    peak_loc=fitted_data_peak_y, 
    device='cuda'
)
```

### Step 3: Execute the Local COWs EM Loop
Pass the mapped control variables and continuous PDFs to the `BaseMLSWeightExtractor`[cite: 22]:

```python
from core import BaseMLSWeightExtractor

extractor = BaseMLSWeightExtractor(disc_bounds=(Y_MIN, Y_MAX), device='cuda')

# Returns publication-ready statistical sWeights array
sWeights_ND = extractor.fit_transform(
    X_ctrl=X_ctrl_mapped,
    y_disc=y_data_array,
    w_s_init=initial_bayes_prob_guess,
    f_sig_cond=sig_pdf_cond,
    f_bkg_1d=f_bkg_1d_callable,      # Your baseline anchor background fit
    bkg_sampler=bkg_rejection_func,  # Fast sampler for the 1D anchor baseline
    max_iters=30
)
```