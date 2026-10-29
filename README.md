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

**Raw HUP and VEP data are not redistributed in this repository.** Download them from their original public sources and comply with their respective terms of use.

See [`DATA.md`](DATA.md) for setup details.

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

## Key experiment settings

| Setting | Value |
|---|---:|
| Random seed | 42 |
| HUP band-pass | 1-120 Hz |
| Implemented notch frequency | 50 Hz |
| RMS onset window | 1 s |
| Onset z threshold | 3.0 |
| Minimum onset duration | 2 s |
| Laplacian PE dimensions | 4 |
| Hidden dimension | 160 |
| Attention heads | 4 |
| Dropout | 0.20 |
| Graph top-k neighbors | 16 |
| Minimum graph edge | 0.02 |
| Learning rate | 4e-4 |
| Weight decay | 8e-5 |
| VEP maximum epochs | 300 |
| HUP maximum epochs | 500 |
| Effective early stopping | 45 non-improving epochs |
| MMD weight | 0.025 |
| Maximum MMD nodes/domain | 64 |

The controlled VEP/onset sensitivity experiment uses the same HUP split and isolates onset-feature and VEP-initialization effects. MMD is disabled for controlled configurations 1-4; the fifth configuration is the complete ensemble setting.

## Reported held-out HUP results

### Primary learned-transfer model

| Metric | Value |
|---|---:|
| AUC | 0.678 |
| AUPRC | 0.199 |
| MRR | 0.490 |
| Hit@5 | 0.583 |
| Mean first-SOZ rank | 9.42 |

### Supplementary validation-tuned ensemble

| Metric | Value |
|---|---:|
| AUC | 0.686 |
| AUPRC | 0.194 |
| MRR | 0.433 |
| Hit@5 | 0.667 |
| Mean first-SOZ rank | 9.00 |

The same-split spatio-temporal graph comparator was strongest on several endpoints. The reported sensitivity experiments therefore support VEP information as a **context-dependent mechanistic prior**, not as a uniformly beneficial source of predictive information.

See [`RESULTS.md`](RESULTS.md) for a compact result summary and interpretation.

## Reproducibility scope

The manuscript reports one fixed patient-disjoint split generated with random seed 42. The implementation contains optional grouped repeated cross-validation support, but this routine was disabled for the archived result set and no repeated-CV performance is claimed.

For repeated HUP seizures, predictions are aggregated by normalized contact name within each patient before primary subject-level evaluation.

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Important generated outputs

The pipeline writes results to `OUTPUT_DIR`. Important files include:

```text
VEP_loader_audit.csv
VEP_test_metrics.csv
HUP_test_metrics.csv
HUP_same_split_manifest.csv
HUP_benchmark_results.csv
HUP_benchmark_heatmap.png
HUP_VEP_onset_controlled_sensitivity.csv
HUP_VEP_contribution_sensitivity_table.csv
HUP_VEP_onset_controlled_paired_bootstrap.csv
HUP_patient_bootstrap_all_methods_v6.csv
HUP_recording_level_metrics_v6.csv
HUP_recording_level_cluster_bootstrap_v6.csv
HUP_recording_level_topk_summary_v6.csv
HUP_paired_subject_tests_v6.csv
HUP_retrained_feature_ablation_v6.csv
HUP_explainability_global_feature_importance_v6.csv
HUP_explainability_feature_attributions_v6.csv
HUP_explainability_neighbor_attributions_v6.csv
HUP_explainability_stability_v6.csv
HUP_explainability_summary_v6.csv
summary_report.txt
hybrid_pvep_eznet_model.pt
```

See [`results/README.md`](results/README.md) for guidance on what to archive publicly.

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




