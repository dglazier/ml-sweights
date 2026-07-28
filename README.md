# Multidimensional ML-sWeights Extractor

## Overview

This repository provides a multidimensional, non-parametric Machine Learning framework designed to separate physical signal amplitudes from background processes in unbinned kinematic datasets. 

Standard analytical background subtraction tools—such as the classic 1D sPlot method—depend strictly on the assumption of **factorization** between the discriminating variable (typically invariant mass `y`) and the control variables (angular kinematics `x`). In modern high-precision experimental environments, physical acceptance boundaries, detector resolution stretching, and moving kinematic thresholds create severe, non-factorizable correlations.

To decouple these complex correlations without relying on rigid, global parametric assumptions, this algorithm combines three core components:
1. **Conditional Normalizing Flows (CNFs):** Model dynamic, event-by-event signal resolution stretching and shape morphing initialized directly from 1D anchor fits.
2. **Sequential ML Cascades (PyTorch DNN + XGBoost):** Learn the continuous multidimensional background density ratio directly from data via the Density Ratio Trick.
3. **Local Covariance Inversion (Gauss-Legendre Quadrature / COWs Formalism):** Computes event-by-event local covariance matrices, translating deep-learning probabilities back into mathematically pristine, multi-dimensional sWeights.

The resulting multi-dimensional sWeights naturally incorporate valid negative sideband weights, eliminating spatial over-subtraction while remaining fully compatible with standard sum-of-squares (`Sum W^2`) error propagation in flat trees or `RDataFrame` workflows.

---

## Directory Structure

```text
ml_sweights/
│
├── core/
│   ├── __init__.py
│   ├── extractor.py        # Main MLSWeightExtractor Scikit-Learn style API
│   ├── flow.py             # Conditional Normalizing Flow (zuko) & DynamicFlowWrapper
│   ├── networks.py         # PyTorch DNN architectures, loss functions (uLSIF/RuLSIF/BCE)
│   └── quadrature.py       # Gauss-Legendre quadrature for local covariance integration
│
├── utils/
│   ├── __init__.py
│   └── diagnostics.py      # Chi2 evaluation, 1D/2D kinematic projection plotting
│
└── examples/
    ├── omega_generators.py     # Central physics toy generator (Schilling-Wolf SDMEs)
    ├── omega_factorizable.py   # Baseline test: Uniform flat background & fixed signal
    ├── omega_factorise_exp.py  # Factorizable test: Exponential background & iMinuit anchor
    ├── omega_fully_dependent.py# Complex test: Correlated resolution, reflections & thresholds
    └── compare_cows_ml.py      # Direct benchmark: Standard 1D sPlot vs COWs vs ML-sWeights
```

---

## Installation & Requirements

The framework requires a Python 3.9+ environment with CUDA acceleration. Batched numerical quadrature and neural network updates rely heavily on GPU VRAM for high-throughput processing.

### Dependencies
```bash
pip install numpy scipy matplotlib torch scikit-learn xgboost zuko iminuit cupy-cuda12x
```

> **Note on GPU Execution:** Ensure `xgboost` is installed with CUDA support. The `MLSWeightExtractor` defaults to `gbdt_mode='XGB_GPU'` (`tree_method='hist'`, `device='cuda'`) to avoid PCIe bus bottlenecking during the Expectation-Maximization (EM) loop.

---

## The Example Suite

The `examples/` directory contains five scripts designed to validate the mathematics, test edge cases, and demonstrate performance across varying degrees of phase-space complexity.

### 1. `omega_generators.py` (Central Physics Engine)
Contains the `OmegaToyGenerator` class, which simulates omega meson photoproduction using Schilling-Wolf Spin Density Matrix Elements (SDMEs).
* **Signal Physics:** Generates topological angles (`cos(theta)`, `phi`, `Phi`) according to linear photon polarization (`P_gamma = 0.85`) and exponential momentum transfer (`|t|`).
* **Modes Supported:**
  * `'factorizable'`: Fixed Gaussian mass resolution (`sigma_m = 10 MeV`) and uniform or exponential background mass.
  * `'correlated'`: Dynamic mass smearing dependent on `cos(theta)` and `|t|`, plus kinematic reflections and `cos(theta)`-dependent geometric acceptance thresholds.

### 2. `omega_factorizable.py`
* **Scenario:** Flat uniform background with a constant-resolution signal peak.
* **Key Focus:** Serves as the fundamental baseline. Because the data is strictly factorizable, the multidimensional density ratio is flat (`R = 1.0`). 
* **Execution:**
  ```bash
  python examples/omega_factorizable.py
  ```

### 3. `omega_factorise_exp.py`
* **Scenario:** Strictly factorizable kinematics, but with a steep **exponential background** mass shape.
* **Key Focus:** Demonstrates the hybrid 1D anchor fit using `iMinuit` (Truncated Exponential x Bernstein polynomial). It shows how initializing the Conditional Normalizing Flow directly at the 1D anchor (`build_signal_flow`) prevents "downhill drift" into dense exponential sidebands.
* **Execution:**
  ```bash
  python examples/omega_factorise_exp.py
  ```

### 4. `omega_fully_dependent.py`
* **Scenario:** The ultimate non-factorizable stress test. Signal mass resolution stretches dynamically with kinematics, background features moving mass cutoffs, and kinematic reflections distort the mass shape across phase space.
* **Key Focus:** Exercises the full power of the ML cascade. The CNF dynamically tracks signal resolution shifts, while the DNN + XGBoost cascade maps the non-linear background density ratio.
* **Execution:**
  ```bash
  python examples/omega_fully_dependent.py
  ```

### 5. `compare_cows_ml.py`
* **Scenario:** Direct side-by-side performance benchmark comparing three methods:
  1. Standard 1D sPlot (Global Covariance Matrix).
  2. Classical COWs (Custom Orthogonal Weight functions with fixed 1D models).
  3. Multidimensional ML-sWeights (Dynamic Density Ratio + Local Quadrature Inversion).
* **Key Focus:** Illustrates how standard 1D sPlot fails (carving unphysical negative pits in forward angles) under correlated resolutions, whereas ML-sWeights successfully un-biases the extracted cross-sections.
* **Execution:**
  ```bash
  python examples/compare_cows_ml.py
  ```

---

## Configuration Guide (`MLSWeightExtractor`)

The `MLSWeightExtractor` class in `core/extractor.py` provides a unified Scikit-Learn style interface (`fit_transform`). Tweaking its hyperparameters allows you to tune performance between strict factorizable stability and high-capacity non-linear extraction.

```python
from core import MLSWeightExtractor

extractor = MLSWeightExtractor(
    disc_bounds=(0.73, 0.85),
    n_nodes=40,
    gbdt_mode='XGB_GPU',
    device='cuda',
    marginal_alpha=0.5,
    dynamic_alpha=True,
    use_gbdt=False,
    reinstate_nn=True,
    dnn_epochs=100,
    ml_transform=False,
    loss_fn='uLSIF',
    logit_l2_penalty=0.05
)
```

### Parameter Reference & Recommendations

| Parameter | Type | Default | Description & Tuning Advice |
| :--- | :--- | :--- | :--- |
| `disc_bounds` | `tuple` | *(required)* | `(Y_MIN, Y_MAX)` of the discriminating variable (e.g., invariant mass window). |
| `loss_fn` | `str` | `'uLSIF'` | Loss function for density ratio estimation: <br>• `'uLSIF'`: Unconstrained Least-Squares Importance Fitting (recommended). <br>• `'RuLSIF'`: Relative uLSIF (bounded variance). <br>• `'BCE'`: Binary Cross-Entropy (standard classification logit). |
| `use_gbdt` | `bool` | `True` | Toggles the sequential XGBoost residual booster: <br>• `False`: Set for **factorizable data** to prevent trees from over-fitting Poisson noise. <br>• `True`: Set for **fully dependent data** to capture sharp local residual boundaries. |
| `logit_l2_penalty` | `float` | `0.001` | L2 regularization applied to classifier logits (z-score penalty): <br>• Increase to `0.05` for **factorizable setups** to force R(x) -> 1.0 unless strong gradients exist. <br>• Lower to `0.001` for **complex correlated setups** to allow sharp boundary learning. |
| `marginal_alpha` | `float` | `0.5` | Initial geometric shape prior weighting (`alpha` between 0 and 1) for the marginal background correction factor. |
| `dynamic_alpha` | `bool` | `True` | Enables the **Active Suspension Thermostat**. Automatically stiffens `alpha` to 0 if signal cannibalization is detected, or relaxes `alpha` to 1 when shapes converge stably. |
| `reinstate_nn` | `bool` | `True` | Retrains the smooth PyTorch Deep Neural Network at the start of each EM iteration before passing residuals to XGBoost. |
| `dnn_epochs` | `int` | `100` | Maximum training epochs per EM iteration for the PyTorch MLP (governed by Autonomous Early Stopping). |
| `n_nodes` | `int` | `40` | Number of Gauss-Legendre numerical quadrature nodes used for local covariance matrix integrations. Higher values increase VRAM usage but improve numerical precision. |

---

## How to Integrate Custom Experimental Data

To apply this framework to your own unbinned experimental datasets:

### Step 1: Topology Mapping
Map periodic angular variables to continuous trigonometric features to prevent neural networks from struggling at phase-space boundaries (`-pi` to `+pi`).

```python
import numpy as np

def map_topology(x_raw):
    # x_raw: [cos_theta, phi, Phi]
    cT, phi, Phi = x_raw[:, 0:1], x_raw[:, 1:2], x_raw[:, 2:3]
    sT = np.sqrt(np.maximum(1 - cT**2, 0.0))
    return np.hstack([cT, sT, np.cos(phi), np.sin(phi), np.cos(Phi), np.sin(Phi)])

# Combine topological features with auxiliary kinematics (e.g., momentum transfer z)
X_ctrl_data = np.hstack([map_topology(x_data), z_data])
```

### Step 2: Initialize Signal Flow at 1D Anchor
Fit your 1D mass distribution (e.g., using `iMinuit` or standard ROOT fits) to obtain the signal peak position (`y_peak`), signal fraction (`f0`), and baseline background function (`f_bkg_1d`). Then instantiate the signal flow:

```python
from core import build_signal_flow

# Instantiate flow trained on pure Monte Carlo signal, locked to 1D anchor
sig_pdf_Z = build_signal_flow(
    Z_ctx_mc=Z_ctx_mc, 
    y_mc=y_mc, 
    base_loc=0.782, 
    base_scale=0.010, 
    peak_loc=fitted_y_peak, 
    device='cuda'
)

def user_signal_pdf(X_ctrl, y_array):
    # Unpack X_ctrl and pass to CNF
    cT, cp, sp, cP, sP, z = X_ctrl[:, 0], X_ctrl[:, 2], X_ctrl[:, 3], X_ctrl[:, 4], X_ctrl[:, 5], X_ctrl[:, 6]
    phi, Phi = np.arctan2(sp, cp), np.arctan2(sP, cP)
    Z_ctx = np.column_stack([cT, phi, Phi, z])
    return sig_pdf_Z(Z_ctx, y_array)
```

### Step 3: Run Extraction
Pass the controls and mass arrays to `MLSWeightExtractor`:

```python
extractor = MLSWeightExtractor(disc_bounds=(Y_MIN, Y_MAX), device='cuda')

sWeights_ND = extractor.fit_transform(
    X_ctrl=X_ctrl_data,
    y_disc=y_data,
    w_s_init=w_s_bayes_init,
    f_sig_cond=user_signal_pdf,
    f_bkg_1d=f_bkg_1d_callable,
    bkg_sampler=bkg_rejection_sampler_func,
    max_iters=30
)

# Extract event weights and store in ROOT RDataFrame / Pandas
# sWeights_ND contains publication-ready event weights!
```

---

## Summary of Configuration Setups

| Scenario | `use_gbdt` | `logit_l2_penalty` | Primary Goal |
| :--- | :--- | :--- | :--- |
| **Factorizable Check** (`omega_factorizable.py`, `omega_factorise_exp.py`) | `False` | `0.05` | Suppresses noise fitting, ensuring R -> 1.0 and matching classic 1D sPlot. |
| **Fully Dependent Physics** (`omega_fully_dependent.py`) | `True` | `0.001` | Maximizes model capacity to capture dynamic smearing and kinematic reflections. |