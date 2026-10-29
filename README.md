# pVEP-EZNet

**Mechanism-Informed Explainable Graph Ranking of Seizure-Onset-Zone Candidates Using Virtual Epileptic Patient Pretraining and Human Intracranial EEG**

This repository contains the implementation and reproducibility support files for **pVEP-EZNet**, an explainable graph-ranking framework for prioritizing seizure-onset-zone (SOZ) contacts. The framework combines mechanistic **Virtual Epileptic Patient (VEP)** pretraining with patient-disjoint evaluation on real human intracranial EEG (**HUP iEEG**).

The main implementation is:

```text
pvep_eznet_v6.py
```

## Overview

The pipeline contains two scientific stages:

1. **VEP mechanistic pretraining**
   - Builds graph instances from the Virtual Epileptic Patient cohort.
   - Uses simulated source time series to construct neural input descriptors.
   - Uses the VEP excitability parameter only for EZ-label construction/audit, not as a learned input feature.

2. **HUP supervised SOZ ranking**
   - Loads ictal HUP iEEG recordings.
   - Constructs contact-level physiological features and functional/spatial graphs.
   - Initializes the graph model from the VEP checkpoint.
   - Fine-tunes on a patient-disjoint HUP training set.
   - Evaluates SOZ candidate ranking on a frozen held-out HUP test cohort.

The repository also supports same-split baselines, controlled VEP/onset sensitivity analysis, patient-level and recording-level statistics, retrained feature-group ablations, and multi-level explainability.

## Main model

Each contact is represented using seven signal/graph descriptors plus four Laplacian positional-encoding dimensions, for **11 neural inputs per node**.

The pVEP-EZNet architecture contains:

- 160-dimensional node projection
- learned feature gate
- two residual graph-convolution layers
- four-head weighted graph attention
- node-local multilayer perceptron
- learned node-wise fusion gate
- contact-level SOZ ranking logit

Graph edges combine functional and spatial information:

```text
A = 0.70 * A_functional + 0.30 * A_spatial
```

with a spatial Gaussian scale of 20 mm, up to 16 retained neighbors per node, and minimum edge weight 0.02.

The ranking objective combines focal, Dice, hard-pair, listwise, and differentiable soft-rank losses.

## Datasets

### HUP iEEG Epilepsy Dataset

- OpenNeuro accession: `ds004100`
- Version used in the manuscript: `1.1.3`
- DOI: https://doi.org/10.18112/openneuro.ds004100.v1.1.3

The reported analysis retained:

- 56 subjects
- 108 ictal recording graphs
- 35 training subjects / 9 validation subjects / 12 held-out test subjects

### Virtual Epileptic Patient cohort

- EBRAINS DOI: https://doi.org/10.25493/XWG2-S8X

The reported analysis retained:

- 30 virtual subjects
- 184 graph instances
- 19 training subjects / 5 validation subjects / 6 held-out test subjects

## Installation

Create and activate a Python virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

For GPU execution, install a PyTorch build compatible with your CUDA environment if needed.

## Configure local paths

Before running the script, edit the three path variables near the top of `pvep_eznet_v6.py`:

```python
HUP_ROOT = r"..."
VEP_ROOT = r"..."
OUTPUT_DIR = r"..."
```

The current implementation intentionally keeps these as explicit local paths rather than command-line arguments.

## Run

```bash
python pvep_eznet_v6.py
```

The script automatically uses CUDA when available:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

## Suggested repository structure

```text
Seizure-Onset-Zone-/
├── pvep_eznet_v6.py
├── README.md
├── requirements.txt
├── CITATION.cff
├── DATA.md
├── REPRODUCIBILITY.md
├── RESULTS.md
├── LICENSE_OPTIONS.md
├── .gitignore
├── figures/
│   └── README.md
└── results/
    └── README.md
```




