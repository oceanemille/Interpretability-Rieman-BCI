# Interpretable Riemannian BCI

Research code for computing and comparing channel-level attribution methods in
motor-imagery brain-computer interfaces (BCIs). The repository focuses on
Riemannian pipelines operating on symmetric positive-definite (SPD) covariance
matrices and also supports models that consume raw EEG trials.

The implementation includes:

- permutation feature importance (PFI) for raw EEG and SPD matrices;
- exact channel-wise Shapley values for SPD matrices;
- KernelSHAP approximations for raw EEG and SPD matrices;
- pyRiemann classifiers such as MDM and tangent-space logistic regression;
- deep SPD models based on SPDNet;
- topographic maps, trial-level Shapley distributions, and ranking-agreement metrics.

## Reproducing the experiments

[`Reproducing_and_Exploring_Results.ipynb`](Reproducing_and_Exploring_Results.ipynb)
is the main entry point. It presents the methods in the following order:

1. PFI on raw EEG and SPD representations, on classical and deep classifiers;
2. Exact Shapley values;
3. KernelSHAP approximations;
4. Agreement between attribution rankings;
5. Trial-level Shapley distributions.

The notebook contains a controlled synthetic example as well as examples based on MOABB datasets. Expensive exact Shapley computations are disabled by default:

```python
RUN_EXACT_SHAPLEY = False
```

Set the flag to `True` to recompute them. Otherwise, the notebook loads saved results from `PRECOMPUTED_RESULTS_DIR`. Update this path near the beginning of the notebook if the artifacts are stored elsewhere.

## Installation

Linux and Python 3.12 are recommended. A CUDA-capable GPU is optional, but can substantially reduce the runtime of compatible exact Shapley and deep-learning experiments.

```bash
git clone https://github.com/oceanemille/Interpretability-Rieman-BCI.git
cd Interpretability-Rieman-BCI

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the notebook with:

```bash
jupyter lab Reproducing_and_Exploring_Results.ipynb
```

The supplied `requirements.txt` is a snapshot of the Linux research
environment and includes CUDA-related packages. A CPU-only installation may require installing the appropriate PyTorch build separately and omitting the CUDA-specific packages.

MOABB downloads datasets when they are requested for the first time. The download location is managed by MOABB/MNE and is not part of this repository.

## Data conventions

Single-subject functions expect one of the following representations:

| Representation | Shape |
| --- | --- |
| Raw EEG | `(n_trials, n_channels, n_times)` |
| SPD covariance matrices | `(n_trials, n_channels, n_channels)` |
| Labels | `(n_trials,)` |

Multi-subject functions accept a dictionary mapping each subject name to its
data and labels:

```python
data = {
    "Subject 1": (X_subject_1, y_subject_1),
    "Subject 2": (X_subject_2, y_subject_2),
}
```

They return a dictionary with a common result schema:

```python
results = {
    "Subject 1": {
        "importance": importance_values,
        "accuracy": classification_scores,
    }
}
```

PFI importance generally has shape `(n_splits, n_channels)`. Shapley results
retain trial-level information and generally have shape
`(n_splits, n_test_trials, n_channels)`.

## Loading MOABB data

The repository provides configurations for `BNCI2014_001`, `Dreyer2023C`, and
`Beetl2021_A`.

```python
from src.data.moabb_adapter import load_from_moabb

# Raw EEG
data_eeg, sensors = load_from_moabb("BNCI2014_001")

# Trial-wise covariance matrices
data_spd, sensors = load_from_moabb("BNCI2014_001", covs=True)
```

Other MOABB datasets can be supplied as dataset objects together with their
sensor names. By default, the adapter selects left- and right-hand motor
imagery and applies a 7–35 Hz filter bank.

## Quick start

### Permutation feature importance

```python
from pyriemann.classification import MDM
from src.Permutation.multi_subjects import compute_pfi_spd_for_subjects

pfi_results = compute_pfi_spd_for_subjects(
    data_spd,
    classifier=MDM,
    n_splits=10,
    n_jobs=-1,
)
```

For SPD inputs, PFI removes the cross-covariances associated with one channel
while preserving its variance, then measures the resulting decrease in
classification accuracy.

### KernelSHAP

```python
from pyriemann.classification import MDM
from src.Shapley.multi_subjects import compute_kernel_shap_spd_for_subjects

kernel_results = compute_kernel_shap_spd_for_subjects(
    data_spd,
    classifier=MDM(),
    n_splits=1,
)
```

KernelSHAP is an approximation. Its runtime and stability depend on the number
of samples used to estimate the Shapley values.

### Exact Shapley values

```python
from src.Shapley.Exact.multi_subjects import compute_exact_shapley_for_subjects

exact_results = compute_exact_shapley_for_subjects(
    data_spd,
    clf="MDM",
    n_splits=10,
    input_is_covariance=True,
)
```

Exact Shapley computation enumerates channel coalitions and therefore scales
exponentially with the number of channels. For the full datasets, prefer the
saved artifacts for exploration and reserve recomputation for a suitable
machine.

## Visualizing and comparing attributions

### Topographic map

```python
import numpy as np
from src.Visualization.topomap import plot_topomap

subject_importance = np.mean(
    pfi_results["Subject 1"]["importance"],
    axis=0,
)

plot_topomap(
    subject_importance,
    sensors,
    title="PFI — Subject 1",
    cbar_type="Permutation",
    savefile_name="pfi_subject_1.pdf",
)
```

### Trial-level Shapley distribution

```python
import numpy as np
from src.Visualization.beeswarm import shap_beeswarm

subject_shapley = np.asarray(
    kernel_results["Subject 1"]["importance"]
)[0]

fig, ax = shap_beeswarm(
    subject_shapley,
    sensor_names=sensors,
    subject="Subject 1",
    max_channels=10,
)
```

This is a beeswarm-style distribution plot: each point represents one trial.
Unlike the standard SHAP beeswarm, point color does not encode the original feature value.

### Ranking agreement

```python
import numpy as np
from src.Interpretability import compare_rankings

pfi_importance = np.mean(
    pfi_results["Subject 1"]["importance"],
    axis=0,
)
shapley_importance = np.mean(
    np.asarray(kernel_results["Subject 1"]["importance"]),
    axis=(0, 1),
)

agreement = compare_rankings(
    pfi_importance,
    shapley_importance,
    channel_names=sensors,
    k=5,
    rank_by="magnitude",
)
```

The returned dictionary contains Spearman correlation, top-k overlap, Kendall Tay and Weighted Kendall Tau. Use `rank_by="value"` to preserve the direction of signed attributions, or `rank_by="magnitude"` to compare attribution strength
regardless of direction. The latter is useful to compare Shapley and PFI.

## Repository structure

```text
.
├── Reproducing_and_Exploring_Results.ipynb
├── requirements.txt
└── src
    ├── data/                  # Dataset configuration, MOABB adapter, utilities
    ├── Permutation/           # Raw-EEG and SPD permutation importance
    ├── Shapley/
    │   └── Exact/             # Exact coalition-based Shapley computation
    ├── SPDNet/                # Deep SPD model and training utilities
    ├── Interpretability/      # Ranking-agreement metrics
    └── Visualization/         # Topomaps, beeswarm plots, and related tools
```

## Reproducibility notes

- Repeated train/test splits use fixed integer seeds where supported.
- Classification accuracy and attribution values are reported per split.
- Shapley signs refer to the explained output class; verify the class ordering of a custom classifier before interpreting direction.
- Precomputed arrays should be kept with their classifier, dataset, split, and target-class metadata.
- The code is intended for research reproducibility. Validate preprocessing, class labels, baselines, and compute settings before applying it to a new dataset.
