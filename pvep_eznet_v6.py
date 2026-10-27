#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pvep_eznet_v6.py

Unified explainable SOZ candidate-ranking pipeline:
    Stage 1: VEP-based mechanistic graph learning
    Stage 2: HUP-based supervised SOZ validation

This V6 version extends the leakage-safe candidate-ranking pipeline with publication-oriented statistics and multi-level graph explainability:
    - VEP mechanistic pretraining
    - HUP patient-level train/validation/test split
    - Structure-aware adjacency: functional + electrode spatial graph when available
    - Laplacian positional encodings
    - Residual feature-gated graph encoder with weighted multi-head attention
    - Focal + Dice + hard-pair + listwise + soft-first-rank objective
    - VEP-to-HUP MMD representation alignment during fine-tuning
    - Validation-tuned HUP/transfer model ensemble (supplementary comparison)
    - Ranking metrics: Top-k hit, Recall@k, nDCG@k, MRR, mean first-SOZ rank
    - Controlled five-configuration VEP/onset sensitivity on the exact same HUP split
    - Subject-cluster bootstrap for recording-level Top-k and subject-level 95% CIs
    - Paired subject-level bootstrap + Wilcoxon tests versus HUP-only and onset-only controls
    - Multi-level explanation audit: integrated gradients, neighbor influence, edge occlusion, feature occlusion, stability
    - Retrained clinical feature-group ablations for onset, energy, HFA, gamma, centrality and Laplacian PE

Edit PATHS below before running.
"""

import os
import re
import glob
import zipfile
import warnings
warnings.filterwarnings("ignore")

# Use a non-interactive Matplotlib backend. This prevents Tk/Tcl crashes
# during long headless/GPU runs on Windows when figures are only saved to disk.
import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import mne
except Exception as exc:
    mne = None
    print("WARNING: mne could not be imported. EDF/HUP loading will fail unless mne is installed.")

from scipy.signal import welch
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import cdist

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    brier_score_loss,
)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PATHS: EDIT THESE
# ============================================================

HUP_ROOT = r"D:\Code\DATASET\hup_data"
VEP_ROOT = r"D:\Code\DATASET\d-c1702a5b-bc8f-486e-a0ee-4758a707ff00\VirtualEpilepticCohort"
OUTPUT_DIR = r"D:\Code\pvep_eznet_v6_results"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOW_FREQ = 1.0
HIGH_FREQ = 120.0
NOTCH_FREQ = 50.0
BASELINE_DURATION_SEC = 60.0
ANALYSIS_DURATION_SEC = 300.0
RMS_WINDOW_SEC = 1.0
ONSET_Z_THRESHOLD = 3.0
MIN_ONSET_DURATION_SEC = 2.0

MAX_HUP_PATIENTS = None
MAX_FILES_PER_HUP_PATIENT = 2
MAX_CHANNELS_PER_RECORDING = None
HUP_ICTAL_ONLY = True

USE_SPATIAL_ADJACENCY = True
SPATIAL_SIGMA_MM = 20.0
FUNCTIONAL_ADJ_WEIGHT = 0.70
SPATIAL_ADJ_WEIGHT = 0.30
N_PSEUDO_BRAIN_REGIONS = 12
REGION_RESULT_TOP_K = 15

# Publication-safe selection objective: "ranking", "auc", or "auprc"
PRIMARY_OBJECTIVE = "ranking"
HYBRID_GRID_STEP = 0.05

USE_LAPLACIAN_PE = True
LAPLACIAN_PE_DIM = 4

HIDDEN_DIM = 160
DROPOUT = 0.20
LR = 4e-4
WEIGHT_DECAY = 8e-5
PRETRAIN_EPOCHS = 300
FINETUNE_EPOCHS = 500
PATIENCE = 90

FOCAL_GAMMA = 2.0
DICE_LOSS_WEIGHT = 0.45
RANKING_LOSS_WEIGHT = 0.30
MAX_POS_WEIGHT = 10.0
CALIBRATION_BINS = 10
BOOTSTRAP_N = 1000
TOP_K_LIST = [1, 3, 5, 10]

# VEP/Epileptor label safeguards. The loader first looks for an explicit EZ
# annotation. Only when none can be found does it use an x0-based fallback.
# Higher (less negative) x0 indicates greater excitability in Epileptor-style
# parameterizations. The fallback is audited and capped to avoid all-positive
# graphs such as 162/162 EZ nodes.
VEP_X0_EZ_THRESHOLD = -2.05
VEP_FALLBACK_PERCENTILE = 90.0
VEP_MAX_EZ_FRACTION = 0.30
VEP_MIN_EZ_NODES = 1
VEP_SKIP_UNINFORMATIVE_X0 = True

BASE_FEATURE_COLUMNS = [
    "onset_score",
    "onset_z_score",
    "energy_score",
    "amplitude_score",
    "gamma_score",
    "hfa_score",
    "centrality_score",
]
LAPLACIAN_PE_COLUMNS = [f"lap_pe_{i+1}" for i in range(LAPLACIAN_PE_DIM)]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + LAPLACIAN_PE_COLUMNS


# ============================================================
# UTILITIES
# ============================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_vector(x):
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return x
    mn, mx = np.min(x), np.max(x)
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def normalize_matrix(W):
    W = np.asarray(W, dtype=float)
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"Connectivity must be square 2D matrix, got {W.shape}")
    W[W < 0] = 0.0
    np.fill_diagonal(W, 0.0)
    mx = np.max(W)
    if mx > 0:
        W = W / mx
    return W


def graph_normalize(W):
    A = normalize_matrix(W)
    A = A + np.eye(A.shape[0])
    deg = A.sum(axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(deg + 1e-8)
    D = np.diag(deg_inv_sqrt)
    return (D @ A @ D).astype(np.float32)


def save_fig(filename):
    """Save a figure without creating/retaining any GUI/Tk resources."""
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close("all")


# ============================================================
# LAPLACIAN POSITIONAL ENCODING
# ============================================================

def compute_laplacian_positional_encoding(W, k=LAPLACIAN_PE_DIM):
    W = normalize_matrix(W)
    n = W.shape[0]
    if n <= 2 or k <= 0:
        return np.zeros((n, k), dtype=np.float32)
    A = W + np.eye(n)
    deg = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-8))
    L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        order = np.argsort(eigvals)
        eigvecs = eigvecs[:, order]
        pe = eigvecs[:, 1:k + 1]
        if pe.shape[1] < k:
            pe = np.hstack([pe, np.zeros((n, k - pe.shape[1]))])
        for j in range(pe.shape[1]):
            if np.sum(pe[:, j]) < 0:
                pe[:, j] *= -1
        return pe.astype(np.float32)
    except Exception:
        return np.zeros((n, k), dtype=np.float32)


def add_laplacian_pe_to_features(W, X_df):
    X_df = X_df.copy()
    if not USE_LAPLACIAN_PE:
        for col in LAPLACIAN_PE_COLUMNS:
            X_df[col] = 0.0
        return X_df
    pe = compute_laplacian_positional_encoding(W, LAPLACIAN_PE_DIM)
    for i, col in enumerate(LAPLACIAN_PE_COLUMNS):
        X_df[col] = pe[:, i]
    return X_df


# ============================================================
# METRICS
# ============================================================

def safe_auc(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def safe_ap(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if np.sum(y) == 0:
        return np.nan
    return float(average_precision_score(y, p))


def topk_hit(y, p, k):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if len(y) == 0:
        return np.nan
    k = min(k, len(y))
    idx = np.argsort(p)[::-1][:k]
    return int(np.any(y[idx] == 1))


def recall_at_k(y, p, k):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    total_pos = np.sum(y == 1)
    if total_pos == 0:
        return np.nan
    k = min(k, len(y))
    idx = np.argsort(p)[::-1][:k]
    return float(np.sum(y[idx] == 1) / total_pos)


def mean_rank_first_positive(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if np.sum(y) == 0:
        return np.nan
    order = np.argsort(p)[::-1]
    for rank, idx in enumerate(order, start=1):
        if y[idx] == 1:
            return float(rank)
    return np.nan


def reciprocal_rank_first_positive(y, p):
    rank = mean_rank_first_positive(y, p)
    if np.isnan(rank) or rank <= 0:
        return np.nan
    return float(1.0 / rank)


def ndcg_at_k(y, p, k):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if len(y) == 0 or np.sum(y) == 0:
        return np.nan
    k = min(k, len(y))
    order = np.argsort(p)[::-1][:k]
    gains = y[order]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains * discounts)
    ideal = np.sort(y)[::-1][:k]
    idcg = np.sum(ideal * discounts)
    if idcg <= 0:
        return np.nan
    return float(dcg / idcg)


def bootstrap_ci_metric(y, p, metric_fn, n_boot=BOOTSTRAP_N, seed=RANDOM_SEED):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if len(y) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            val = metric_fn(y[idx], p[idx])
            if np.isfinite(val):
                vals.append(float(val))
        except Exception:
            continue
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    vals = np.asarray(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ============================================================
# SIGNAL FEATURE EXTRACTION
# ============================================================

def build_functional_connectivity(data):
    if data.shape[0] < 2:
        return np.eye(data.shape[0])
    corr = np.corrcoef(data)
    W = np.abs(corr)
    return normalize_matrix(W)


def bandpower(sig, sfreq, fmin, fmax):
    fmax = min(fmax, sfreq / 2.0 - 1.0)
    if fmax <= fmin:
        return 0.0
    nperseg = min(len(sig), int(2 * sfreq))
    if nperseg < 8:
        return 0.0
    freqs, psd = welch(sig, fs=sfreq, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if np.sum(mask) == 0:
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def compute_rms(data, sfreq):
    win = max(1, int(RMS_WINDOW_SEC * sfreq))
    return np.sqrt(uniform_filter1d(data ** 2, size=win, axis=1))


def detect_onsets(data, sfreq):
    n_ch, n_samp = data.shape
    baseline_samples = int(BASELINE_DURATION_SEC * sfreq)
    baseline_samples = min(baseline_samples, max(1, n_samp // 3))
    min_dur = max(1, int(MIN_ONSET_DURATION_SEC * sfreq))
    rms = compute_rms(data, sfreq)
    onset_times = np.full(n_ch, np.nan)
    onset_zmax = np.zeros(n_ch)
    for ch in range(n_ch):
        base = rms[ch, :baseline_samples]
        mu = np.mean(base)
        sd = np.std(base) + 1e-8
        z = (rms[ch] - mu) / sd
        onset_zmax[ch] = np.max(z)
        active = z > ONSET_Z_THRESHOLD
        for k in range(0, max(1, n_samp - min_dur)):
            if np.all(active[k:k + min_dur]):
                onset_times[ch] = k / sfreq
                break
    return onset_times, onset_zmax






# ============================================================
# RECORDING LOADERS AND BIDS SIDE-CARS
# ============================================================

def load_and_preprocess_recording(path, max_channels=None, tmin=None, tmax=None):
    if mne is None:
        raise ImportError("mne is required to load EDF/BDF/VHDR/FIF/SET files.")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".edf":
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    elif ext == ".bdf":
        raw = mne.io.read_raw_bdf(path, preload=True, verbose=False)
    elif ext == ".vhdr":
        raw = mne.io.read_raw_brainvision(path, preload=True, verbose=False)
    elif ext == ".fif":
        raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    elif ext == ".set":
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported recording type: {path}")
    sfreq = raw.info["sfreq"]
    nyq = sfreq / 2.0
    high = min(HIGH_FREQ, nyq - 1.0)
    picks = mne.pick_types(raw.info, eeg=True, seeg=True, ecog=True, misc=False, exclude=[])
    if len(picks) == 0:
        picks = np.arange(len(raw.ch_names))
    raw.pick(picks)
    raw.filter(l_freq=LOW_FREQ, h_freq=high, fir_design="firwin", verbose=False)
    if NOTCH_FREQ < nyq:
        raw.notch_filter(freqs=[NOTCH_FREQ], verbose=False)
    duration = raw.times[-1]
    if tmin is None or tmax is None:
        tmin = 0.0
        tmax = min(duration, ANALYSIS_DURATION_SEC)
    else:
        tmin = max(0.0, float(tmin))
        tmax = min(duration, float(tmax))
        if tmax <= tmin:
            tmin = 0.0
            tmax = min(duration, ANALYSIS_DURATION_SEC)
    raw.crop(tmin=tmin, tmax=tmax)
    data = np.nan_to_num(raw.get_data())
    ch_names = raw.ch_names
    sfreq = raw.info["sfreq"]
    if max_channels is not None and data.shape[0] > max_channels:
        var = np.var(data, axis=1)
        idx = np.argsort(var)[::-1][:max_channels]
        idx = np.sort(idx)
        data = data[idx]
        ch_names = [ch_names[i] for i in idx]
        selected_idx = idx
    else:
        selected_idx = np.arange(data.shape[0])
    return data, sfreq, ch_names, selected_idx


def get_subject_id_from_path(path):
    parts = path.replace("\\", "/").split("/")
    for p in parts:
        if p.startswith("sub-"):
            return p
    return os.path.basename(os.path.dirname(path))


def find_hup_recordings(root):
    exts = ["*.edf", "*.bdf", "*.vhdr", "*.fif", "*.set"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return sorted(list(set(files)))


def find_sidecar_for_recording(recording_path, suffix):
    folder = os.path.dirname(recording_path)
    base = os.path.basename(recording_path)
    stem = os.path.splitext(base)[0].replace("_ieeg", "")
    candidates = [os.path.join(folder, stem + "_" + suffix)]
    candidates.extend(glob.glob(os.path.join(folder, f"*{suffix}")))
    session_dir = os.path.dirname(folder)
    subject_dir = os.path.dirname(session_dir)
    candidates.extend(glob.glob(os.path.join(session_dir, "**", f"*{suffix}"), recursive=True))
    candidates.extend(glob.glob(os.path.join(subject_dir, "**", f"*{suffix}"), recursive=True))
    candidates = sorted(list(set([c for c in candidates if os.path.exists(c)])))
    if not candidates:
        return None
    for c in candidates:
        if stem in os.path.basename(c):
            return c
    return candidates[0]


def normalize_contact_name(x):
    x = str(x).strip().lower()
    x = x.replace("'", "")
    x = re.sub(r"[^a-zA-Z0-9]", "", x)
    return x


def split_bipolar_contacts(ch_name):
    raw = str(ch_name).strip()
    parts = re.split(r"[-–—]", raw)
    parts = [normalize_contact_name(x) for x in parts if str(x).strip()]
    base = normalize_contact_name(raw)
    out = []
    for x in parts + [base]:
        if x and x not in out:
            out.append(x)
    return out


def infer_soz_labels_from_channels(ch_names, recording_path):
    channels_path = find_sidecar_for_recording(recording_path, "channels.tsv")
    if channels_path is None:
        return None
    try:
        df = pd.read_csv(channels_path, sep="\t")
    except Exception:
        return None
    if "name" not in df.columns or "status_description" not in df.columns:
        return None
    y = np.zeros(len(ch_names), dtype=int)
    ch_norm_to_index = {normalize_contact_name(ch): i for i, ch in enumerate(ch_names)}
    found_any = False
    for _, row in df.iterrows():
        desc = str(row.get("status_description", "")).lower()
        if "soz" not in desc:
            continue
        norm_name = normalize_contact_name(row["name"])
        if norm_name in ch_norm_to_index:
            y[ch_norm_to_index[norm_name]] = 1
            found_any = True
    if found_any and np.sum(y) > 0:
        return y
    return None


def get_seizure_window_from_events(recording_path, pre_sec=60, post_sec=180):
    events_path = find_sidecar_for_recording(recording_path, "events.tsv")
    if events_path is None:
        return 0.0, ANALYSIS_DURATION_SEC
    try:
        df = pd.read_csv(events_path, sep="\t")
    except Exception:
        return 0.0, ANALYSIS_DURATION_SEC
    if "onset" not in df.columns:
        return 0.0, ANALYSIS_DURATION_SEC
    row_text = df.astype(str).apply(lambda row: " ".join(row.values).lower(), axis=1)
    onset_rows = df[row_text.str.contains("sz onset|seizure onset|ictal onset|onset", regex=True)]
    offset_rows = df[row_text.str.contains("sz offset|seizure offset|ictal offset|offset", regex=True)]
    if len(onset_rows) == 0:
        return 0.0, ANALYSIS_DURATION_SEC
    try:
        sz_onset = float(onset_rows.iloc[0]["onset"])
    except Exception:
        return 0.0, ANALYSIS_DURATION_SEC
    if len(offset_rows) > 0:
        try:
            sz_offset = float(offset_rows.iloc[0]["onset"])
        except Exception:
            sz_offset = sz_onset + post_sec
    else:
        sz_offset = sz_onset + post_sec
    tmin = max(0.0, sz_onset - pre_sec)
    tmax = min(sz_offset, sz_onset + post_sec)
    if tmax <= tmin:
        tmax = tmin + ANALYSIS_DURATION_SEC
    return tmin, tmax


# ============================================================
# STRUCTURE-AWARE COORDINATES AND REGIONS
# ============================================================

def read_electrode_table_for_recording(recording_path):
    epath = find_sidecar_for_recording(recording_path, "electrodes.tsv")
    if epath is None or not os.path.exists(epath):
        return None
    try:
        df = pd.read_csv(epath, sep="\t")
        needed = {"name", "x", "y", "z"}
        if not needed.issubset(set(df.columns)):
            return None
        df = df.copy()
        df["norm_name"] = df["name"].map(normalize_contact_name)
        for c in ["x", "y", "z"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["x", "y", "z"])
        if len(df) == 0:
            return None
        return df
    except Exception:
        return None


def align_electrode_coords_to_channels(ch_names, recording_path):
    edf = read_electrode_table_for_recording(recording_path)
    if edf is None:
        return None
    lookup = {row["norm_name"]: np.array([row["x"], row["y"], row["z"]], dtype=float) for _, row in edf.iterrows()}
    coords = []
    matched = 0
    for ch in ch_names:
        tokens = split_bipolar_contacts(ch)
        pts = [lookup[t] for t in tokens if t in lookup]
        if len(pts) == 0:
            coords.append([np.nan, np.nan, np.nan])
        else:
            coords.append(np.mean(np.vstack(pts), axis=0))
            matched += 1
    coords = np.asarray(coords, dtype=float)
    if matched < max(3, int(0.25 * len(ch_names))):
        return None
    for j in range(3):
        col = coords[:, j]
        med = np.nanmedian(col)
        if np.isnan(med):
            med = 0.0
        col[np.isnan(col)] = med
        coords[:, j] = col
    return coords


def build_spatial_adjacency_from_electrodes(coords, sigma=SPATIAL_SIGMA_MM):
    coords = np.asarray(coords, dtype=float)
    D = cdist(coords, coords, metric="euclidean")
    A = np.exp(-(D ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(A, 0.0)
    return normalize_matrix(A)


def combine_functional_spatial_adjacency(A_func, coords, functional_weight=FUNCTIONAL_ADJ_WEIGHT):
    if coords is None or not USE_SPATIAL_ADJACENCY:
        return normalize_matrix(A_func), None
    A_spatial = build_spatial_adjacency_from_electrodes(coords)
    A = functional_weight * normalize_matrix(A_func) + (1.0 - functional_weight) * A_spatial
    np.fill_diagonal(A, 0.0)
    return normalize_matrix(A), A_spatial


def make_synthetic_brain_coords(n, radius_x=70.0, radius_y=45.0, radius_z=35.0):
    if n <= 0:
        return np.zeros((0, 3))
    i = np.arange(n, dtype=float)
    theta = 2.0 * np.pi * i / max(1, n)
    z = np.linspace(-0.7, 0.7, n)
    r = np.sqrt(np.maximum(0.0, 1.0 - z ** 2))
    x = radius_x * r * np.cos(theta)
    y = radius_y * r * np.sin(theta)
    zz = radius_z * z
    return np.vstack([x, y, zz]).T


def pseudo_region_names(n_regions=N_PSEUDO_BRAIN_REGIONS):
    base = [
        "Frontal", "Temporal", "Parietal", "Occipital", "Insular", "Limbic",
        "Motor", "Somatosensory", "Cingulate", "Orbitofrontal", "Basal", "Posterior"
    ]
    if n_regions <= len(base):
        return base[:n_regions]
    return base + [f"Region-{i+1}" for i in range(len(base), n_regions)]


def create_pseudo_region_centroids(coords, n_regions=N_PSEUDO_BRAIN_REGIONS):
    coords = np.asarray(coords, dtype=float)
    if len(coords) == 0:
        return np.zeros((n_regions, 3)), pseudo_region_names(n_regions)
    order = np.argsort(coords[:, 0])
    groups = np.array_split(order, n_regions)
    centroids = []
    for g in groups:
        centroids.append(np.mean(coords[g], axis=0) if len(g) else np.nanmean(coords, axis=0))
    return np.asarray(centroids), pseudo_region_names(n_regions)


def map_channels_to_pseudo_regions(coords, n_regions=N_PSEUDO_BRAIN_REGIONS):
    region_coords, region_names = create_pseudo_region_centroids(coords, n_regions)
    D = cdist(np.asarray(coords, dtype=float), region_coords)
    channel_to_region = np.argmin(D, axis=1)
    return channel_to_region, region_coords, region_names


def aggregate_channel_scores_to_regions(channel_scores, channel_to_region, n_regions):
    scores = np.asarray(channel_scores, dtype=float)
    region_scores = np.zeros(n_regions, dtype=float)
    counts = np.zeros(n_regions, dtype=float)
    for i, r in enumerate(channel_to_region):
        region_scores[int(r)] += scores[i]
        counts[int(r)] += 1.0
    counts[counts == 0] = 1.0
    return normalize_vector(region_scores / counts)


def region_adjacency_from_centroids(region_coords, sigma=45.0):
    D = cdist(region_coords, region_coords)
    A = np.exp(-(D ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(A, 0.0)
    return normalize_matrix(A)


def add_structural_annotations_to_graph(graph, coords=None, final_scores=None, n_regions=N_PSEUDO_BRAIN_REGIONS):
    n = len(graph["channel_names"])
    if coords is None:
        coords = make_synthetic_brain_coords(n)
        graph["coordinate_source"] = "synthetic_layout_no_electrodes_tsv"
    else:
        graph["coordinate_source"] = "electrodes_tsv"
    coords = np.asarray(coords, dtype=float)
    graph["electrode_coords"] = coords
    channel_to_region, region_coords, region_names = map_channels_to_pseudo_regions(coords, n_regions)
    graph["channel_to_region"] = channel_to_region
    graph["region_coords"] = region_coords
    graph["region_names"] = region_names
    graph["region_adjacency"] = region_adjacency_from_centroids(region_coords)
    if final_scores is not None:
        graph["region_scores"] = aggregate_channel_scores_to_regions(final_scores, channel_to_region, len(region_names))
    return graph


# ============================================================
# PATIENT GRAPH FORMAT
# ============================================================

def make_patient_graph(patient_id, W, X_df, y=None, channel_names=None, source="unknown"):
    """Create a graph while keeping audit-only features separate from model inputs.

    X contains only FEATURE_COLUMNS and is the only matrix seen by the neural model.
    X_df keeps all derived/audit features (e.g., heuristic composite score and,
    for VEP only, true x0) so baselines and result tables remain available without
    leaking those variables into the learned representation.
    """
    W = normalize_matrix(W)
    full_df = X_df.copy().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    full_df = add_laplacian_pe_to_features(W, full_df)

    model_df = full_df.copy()
    for col in FEATURE_COLUMNS:
        if col not in model_df.columns:
            model_df[col] = 0.0
    model_df = model_df[FEATURE_COLUMNS].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    X = model_df.values.astype(np.float32)

    n = X.shape[0]
    if y is None:
        y = np.zeros(n, dtype=np.float32)
    else:
        y = np.asarray(y).astype(float).reshape(-1)
        if len(y) != n:
            raise ValueError(f"Label length mismatch in {patient_id}: X={n}, y={len(y)}")
        y = (y > 0).astype(np.float32)
    if channel_names is None:
        channel_names = [f"node_{i}" for i in range(n)]
    return {
        "patient_id": patient_id,
        "source": source,
        "W": W,
        "X": X,
        "X_df": full_df,
        "model_X_df": model_df,
        "y": y,
        "channel_names": channel_names,
    }


# ============================================================
# VEP COHORT LOADER
# ============================================================

def read_tsv_flexible(path):
    return pd.read_csv(path, sep="\t")


def load_weights_txt(path):
    W = np.loadtxt(path)
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"weights.txt is not square: {path}, shape={W.shape}")
    return normalize_matrix(W)


def find_subject_dirs_vep(root):
    tvb_root = os.path.join(root, "derivatives", "tvb")
    if not os.path.exists(tvb_root):
        raise FileNotFoundError(f"Cannot find derivatives/tvb inside VEP root: {tvb_root}")
    return sorted([os.path.join(tvb_root, d) for d in os.listdir(tvb_root)
                   if d.startswith("sub-") and os.path.isdir(os.path.join(tvb_root, d))])


def find_connectome_weights_for_subject(sub_dir):
    files = glob.glob(os.path.join(sub_dir, "struct", "**", "weights.txt"), recursive=True)
    if files:
        return files[0]
    zip_files = glob.glob(os.path.join(sub_dir, "struct", "*connectome*.zip"))
    for zpath in zip_files:
        try:
            extract_dir = os.path.splitext(zpath)[0]
            if not os.path.exists(extract_dir):
                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(zpath, "r") as zf:
                    zf.extractall(extract_dir)
            files = glob.glob(os.path.join(extract_dir, "**", "weights.txt"), recursive=True)
            if files:
                return files[0]
        except Exception as e:
            print(f"Could not extract connectome zip {zpath}: {e}")
    return None


def find_epileptor_parameter_files(sub_dir):
    files = glob.glob(os.path.join(sub_dir, "ses-*", "*hypothesis", "parameters", "*epileptor_parameters*.tsv"), recursive=True)
    return sorted(files)


def find_matching_source_timeseries(param_file):
    hyp_dir = os.path.dirname(os.path.dirname(param_file))
    base = os.path.basename(param_file)
    run_match = re.search(r"run-(\d+)", base)
    run_id = run_match.group(1) if run_match else None
    if run_id is not None:
        npz_files = glob.glob(os.path.join(hyp_dir, f"*simulated_source_timeseries_run-{run_id}.npz"))
        tsv_files = glob.glob(os.path.join(hyp_dir, f"*simulated_source_timeseries_run-{run_id}.tsv"))
    else:
        npz_files = glob.glob(os.path.join(hyp_dir, "*simulated_source_timeseries*.npz"))
        tsv_files = glob.glob(os.path.join(hyp_dir, "*simulated_source_timeseries*.tsv"))
    if npz_files:
        return npz_files[0]
    if tsv_files:
        return tsv_files[0]
    return None


def parse_numeric_vector_from_any(value):
    if isinstance(value, np.ndarray):
        return value.astype(float).reshape(-1)
    if isinstance(value, list):
        return np.asarray(value, dtype=float).reshape(-1)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.asarray([float(value)])
    s = str(value).replace("[", " ").replace("]", " ").replace(",", " ")
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", s)
    return np.asarray([float(x) for x in nums], dtype=float) if nums else np.asarray([], dtype=float)


def load_source_timeseries(path):
    if path is None:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        obj = np.load(path, allow_pickle=True)
        keys = list(obj.keys())
        preferred_keys = ["data", "source", "source_timeseries", "timeseries", "x", "X", "ts", "source_signal"]
        arr = None
        for k in preferred_keys:
            if k in keys:
                arr = obj[k]
                break
        if arr is None:
            for k in keys:
                candidate = obj[k]
                if isinstance(candidate, np.ndarray) and candidate.ndim >= 2:
                    arr = candidate
                    break
        if arr is None:
            return None
        arr = np.squeeze(np.asarray(arr, dtype=float))
        if arr.ndim == 3:
            if arr.shape[0] <= 10:
                arr = arr[0, :, :]
            elif arr.shape[-1] <= 10:
                arr = arr[:, :, 0].T
            else:
                arr = arr[:, :, 0].T
        if arr.ndim != 2:
            return None
        if arr.shape[0] > arr.shape[1]:
            arr = arr.T
        return np.nan_to_num(arr)
    if ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] == 0:
            return None
        arr = numeric_df.values.astype(float)
        if arr.shape[0] > arr.shape[1]:
            arr = arr.T
        return np.nan_to_num(arr)
    return None


def _coerce_binary_label_vector(values, n_nodes):
    """Return a validated 0/1 EZ vector from a candidate column, else None."""
    # Case 1: one cell stores the full vector.
    for value in values:
        vec = parse_numeric_vector_from_any(value)
        if len(vec) == n_nodes:
            rounded = np.rint(vec).astype(int)
            if np.all(np.isin(rounded, [0, 1])) and np.allclose(vec, rounded, atol=1e-6):
                return rounded

    # Case 2: one row per node with numeric 0/1 labels.
    try:
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        if len(arr) == n_nodes and np.all(np.isfinite(arr)):
            rounded = np.rint(arr).astype(int)
            if np.all(np.isin(rounded, [0, 1])) and np.allclose(arr, rounded, atol=1e-6):
                return rounded
    except Exception:
        pass

    # Case 3: one row per node with textual region classes (EZ/PZ/HZ/etc.).
    if len(values) == n_nodes:
        tokens = [str(v).strip().lower() for v in values]
        positive = {
            "ez", "epileptogenic", "epileptogenic zone", "epileptogenic_zone",
            "seizure onset zone", "seizure_onset_zone", "soz", "1", "true", "yes"
        }
        negative = {
            "pz", "propagation", "propagation zone", "propagation_zone",
            "hz", "healthy", "background", "non-ez", "nonez", "0", "false", "no"
        }
        known = [t in positive or t in negative for t in tokens]
        if sum(known) >= max(1, int(0.8 * n_nodes)):
            return np.asarray([1 if t in positive else 0 for t in tokens], dtype=int)
    return None


def _extract_explicit_ez_labels(param_file, param_df, n_nodes):
    """Search the parameter table and nearby hypothesis files for explicit EZ labels."""
    # Strongly preferred column names. Do not treat continuous 'epileptogenicity'
    # values as labels unless the content is genuinely binary/categorical.
    candidates = [
        "ez", "is_ez", "ez_label", "ez_labels", "ez_mask",
        "epileptogenic_zone", "epileptogenic_zone_label",
        "soz", "is_soz", "zone", "region_class", "node_class", "hypothesis_label"
    ]
    lower = {str(c).lower(): c for c in param_df.columns}
    for cand in candidates:
        if cand in lower:
            col = lower[cand]
            y = _coerce_binary_label_vector(param_df[col].values, n_nodes)
            if y is not None:
                return y, f"explicit:{os.path.basename(param_file)}:{col}"

    # Search nearby hypothesis/annotation tables when the parameter TSV itself
    # contains only x0 and dynamical parameters.
    search_dirs = {
        os.path.dirname(param_file),
        os.path.dirname(os.path.dirname(param_file)),
        os.path.dirname(os.path.dirname(os.path.dirname(param_file))),
    }
    patterns = ["*.tsv", "*.csv"]
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        files = []
        for pat in patterns:
            files.extend(glob.glob(os.path.join(d, pat)))
        # Prioritize filenames that look like hypothesis/EZ annotations.
        files = sorted(set(files), key=lambda f: (
            0 if re.search(r"(hypoth|ez|epilept|zone|label)", os.path.basename(f), re.I) else 1,
            os.path.basename(f)
        ))
        for path in files:
            if os.path.abspath(path) == os.path.abspath(param_file):
                continue
            try:
                sep = "\t" if path.lower().endswith(".tsv") else ","
                df = pd.read_csv(path, sep=sep)
            except Exception:
                continue
            lower2 = {str(c).lower(): c for c in df.columns}
            for cand in candidates:
                if cand in lower2:
                    col = lower2[cand]
                    y = _coerce_binary_label_vector(df[col].values, n_nodes)
                    if y is not None:
                        return y, f"explicit:{os.path.basename(path)}:{col}"
    return None, None


def _fallback_ez_labels_from_x0(x0, n_nodes):
    """Audited fallback used only when no explicit VEP EZ annotation is found."""
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    finite = np.isfinite(x0)
    if finite.sum() != n_nodes:
        return None, "invalid_x0"

    # Completely/near-constant x0 carries no information about an EZ set.
    if np.nanstd(x0) < 1e-8 or len(np.unique(np.round(x0, 6))) < 2:
        return None, "uninformative_x0"

    # Prefer an Epileptor-like excitability threshold when values look like
    # physical x0 parameters. If it yields an implausibly large EZ set, fall back
    # to a distribution-based top percentile rather than accepting all nodes.
    y = (x0 >= VEP_X0_EZ_THRESHOLD).astype(int)
    frac = float(y.mean())
    if VEP_MIN_EZ_NODES <= int(y.sum()) and frac <= VEP_MAX_EZ_FRACTION:
        return y, f"x0_threshold>={VEP_X0_EZ_THRESHOLD:g}"

    # Distribution-based fallback. Rank-order is preserved, but the positive set
    # is restricted to a small high-excitability tail.
    q = np.percentile(x0, VEP_FALLBACK_PERCENTILE)
    y = (x0 >= q).astype(int)

    # Handle tied percentiles deterministically by keeping the top K x0 nodes.
    max_pos = max(VEP_MIN_EZ_NODES, int(np.floor(VEP_MAX_EZ_FRACTION * n_nodes)))
    target_k = max(VEP_MIN_EZ_NODES, int(np.ceil((100.0 - VEP_FALLBACK_PERCENTILE) / 100.0 * n_nodes)))
    target_k = min(target_k, max_pos)
    if int(y.sum()) > max_pos or int(y.sum()) == n_nodes:
        order = np.argsort(x0)[::-1]
        y = np.zeros(n_nodes, dtype=int)
        y[order[:target_k]] = 1

    if int(y.sum()) < VEP_MIN_EZ_NODES:
        y[np.argmax(x0)] = 1

    frac = float(y.mean())
    if frac > VEP_MAX_EZ_FRACTION:
        return None, "fallback_too_many_positive"
    return y, f"x0_top_{100.0 - VEP_FALLBACK_PERCENTILE:.0f}pct_fallback"


def parse_epileptor_parameters(param_file, n_nodes):
    """Parse true x0 and obtain a defensible EZ label vector.

    Priority:
      1) explicit EZ/zone annotation in the parameter or nearby hypothesis files;
      2) audited x0-based fallback when explicit labels are unavailable.

    The function deliberately rejects uninformative all-equal x0 configurations
    instead of turning them into all-positive 162/162 EZ graphs.
    """
    df = read_tsv_flexible(param_file)
    original_cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in original_cols}
    x0_candidates = [
        "x0", "x_0", "excitability", "excitability_value", "exc", "eta",
        "epileptogenicity", "epileptogenicity_value"
    ]

    x0 = None
    x0_col = None
    for cand in x0_candidates:
        if cand in lower_map:
            x0_col = lower_map[cand]
            break

    if x0_col is not None:
        # First try a vector stored in a cell.
        for val in df[x0_col].values:
            vec = parse_numeric_vector_from_any(val)
            if len(vec) >= max(2, n_nodes // 2):
                x0 = vec
                break
        # Otherwise a conventional one-row-per-node numeric column.
        if x0 is None:
            vals = pd.to_numeric(df[x0_col], errors="coerce").to_numpy(dtype=float)
            if len(vals) >= max(2, n_nodes // 2) and np.isfinite(vals).sum() >= max(2, n_nodes // 2):
                x0 = vals[np.isfinite(vals)]

    if x0 is None:
        # Conservative fallback: search other columns only for a plausible vector.
        for col in df.columns:
            for val in df[col].values:
                vec = parse_numeric_vector_from_any(val)
                if len(vec) >= max(2, n_nodes // 2):
                    x0 = vec
                    x0_col = col
                    break
            if x0 is not None:
                break

    if x0 is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            vals = np.asarray(df[col].values, dtype=float).reshape(-1)
            if len(vals) >= max(2, n_nodes // 2) and np.nanstd(vals) > 1e-8:
                x0 = vals
                x0_col = col
                break

    if x0 is None:
        raise ValueError(
            f"Could not parse x0/excitability vector from {param_file}; columns={list(df.columns)}"
        )

    x0 = np.asarray(x0, dtype=float).reshape(-1)
    if len(x0) != n_nodes:
        if len(x0) > n_nodes:
            x0 = x0[:n_nodes]
        else:
            finite = x0[np.isfinite(x0)]
            fill = np.nanmedian(finite) if len(finite) else -3.0
            padded = np.full(n_nodes, fill, dtype=float)
            padded[:len(x0)] = x0
            x0 = padded

    if not np.all(np.isfinite(x0)):
        finite = x0[np.isfinite(x0)]
        fill = np.nanmedian(finite) if len(finite) else -3.0
        x0 = np.nan_to_num(x0, nan=fill, posinf=fill, neginf=fill)

    x0_norm = normalize_vector(x0)

    # Prefer explicit labels; use x0 only as an audited fallback.
    y, label_source = _extract_explicit_ez_labels(param_file, df, n_nodes)
    if y is None:
        y, label_source = _fallback_ez_labels_from_x0(x0, n_nodes)

    if y is None:
        if VEP_SKIP_UNINFORMATIVE_X0:
            raise ValueError(
                f"No defensible EZ labels for {param_file}: {label_source}. "
                "Graph skipped instead of creating an all-positive EZ target."
            )
        y = np.zeros(n_nodes, dtype=int)
        y[np.argmax(x0)] = 1
        label_source = "forced_argmax_fallback"

    y = np.asarray(y, dtype=int).reshape(-1)
    if len(y) != n_nodes:
        raise ValueError(f"EZ label length mismatch: expected {n_nodes}, got {len(y)}")
    n_pos = int(y.sum())
    frac = float(n_pos / max(1, n_nodes))
    if n_pos < VEP_MIN_EZ_NODES:
        raise ValueError(f"No EZ-positive nodes in {param_file}")
    if frac > VEP_MAX_EZ_FRACTION:
        raise ValueError(
            f"Suspicious VEP EZ target {n_pos}/{n_nodes} ({frac:.1%}) in {param_file}; "
            "graph skipped. Check the explicit hypothesis annotation."
        )

    # Store audit metadata without changing the historical 4-value return API.
    df.attrs["label_source"] = label_source
    df.attrs["x0_column"] = str(x0_col)
    df.attrs["x0_min"] = float(np.min(x0))
    df.attrs["x0_max"] = float(np.max(x0))
    df.attrs["x0_std"] = float(np.std(x0))
    df.attrs["x0_unique_count"] = int(len(np.unique(np.round(x0, 6))))
    df.attrs["ez_nodes"] = n_pos
    df.attrs["ez_fraction"] = frac

    print(
        f"Loaded VEP params: {os.path.basename(param_file)} | x0_col={x0_col} | "
        f"nodes={len(x0)} | EZ nodes={n_pos} | labels={label_source}"
    )
    return x0, x0_norm, y, df






# ============================================================
# HUP LOADERS
# ============================================================

def load_hup_dataset(root):
    print("\n====================================================")
    print("STAGE 2: Loading HUP real iEEG dataset")
    print("====================================================")
    if not os.path.exists(root):
        raise FileNotFoundError(f"HUP path not found: {root}")
    rec_files = find_hup_recordings(root)
    if not rec_files:
        raise RuntimeError("No HUP recording files found. Expected EDF/BDF/VHDR/FIF/SET.")
    print(f"Found HUP recording files: {len(rec_files)}")
    subject_to_files = {}
    for f in rec_files:
        sid = get_subject_id_from_path(f)
        subject_to_files.setdefault(sid, []).append(f)
    subject_ids = sorted(subject_to_files.keys())
    if MAX_HUP_PATIENTS is not None:
        subject_ids = subject_ids[:MAX_HUP_PATIENTS]
    patients = []
    for sid in subject_ids:
        files_all = sorted(subject_to_files[sid])
        if HUP_ICTAL_ONLY:
            files_all = [f for f in files_all if "task-ictal" in os.path.basename(f).lower()]
        files = files_all[:MAX_FILES_PER_HUP_PATIENT]
        for rec_path in files:
            try:
                tmin, tmax = get_seizure_window_from_events(rec_path)
                data, sfreq, ch_names, selected_idx = load_and_preprocess_recording(
                    rec_path, max_channels=MAX_CHANNELS_PER_RECORDING, tmin=tmin, tmax=tmax
                )
                y = infer_soz_labels_from_channels(ch_names, rec_path)
                if y is None or len(y) != len(ch_names) or np.sum(y) == 0:
                    print(f"No usable SOZ labels for: {os.path.basename(rec_path)}")
                    continue
                print(f"Loaded HUP labels: {os.path.basename(rec_path)} | channels={len(ch_names)} | SOZ={int(np.sum(y))}")
                W_func = build_functional_connectivity(data)
                raw_df = extract_signal_features(data, sfreq, ch_names)
                coords = align_electrode_coords_to_channels(ch_names, rec_path)
                W, W_spatial = combine_functional_spatial_adjacency(W_func, coords)
                X_df = convert_raw_features_to_node_features(raw_df, W)
                pid = f"{sid}_{os.path.splitext(os.path.basename(rec_path))[0]}"
                graph = make_patient_graph(pid, W, X_df, y, ch_names, source="HUP")
                graph["recording_path"] = rec_path
                graph["raw_features"] = raw_df
                graph["W_functional"] = W_func
                graph["W_spatial"] = W_spatial
                graph = add_structural_annotations_to_graph(graph, coords=coords)
                patients.append(graph)
            except Exception as e:
                print(f"Skipping HUP recording {rec_path}: {e}")
    if not patients:
        raise RuntimeError("No labelled HUP patient graphs were loaded.")
    print(f"Loaded labelled HUP patient graphs: {len(patients)}")
    print(f"Median SOZ channels per HUP graph: {np.median([int(np.sum(p['y'])) for p in patients]):.2f}")
    return patients



# ============================================================
# SPLITTING AND SCALING
# ============================================================

def get_group_id_for_split(g):
    pid = str(g.get("patient_id", ""))
    m = re.search(r"(sub-[A-Za-z0-9]+)", pid)
    if m:
        return m.group(1)
    return pid.split("_")[0]


def split_graphs_patient_level(graphs, test_size=0.20, val_size=0.20):
    groups = sorted(list({get_group_id_for_split(g) for g in graphs}))
    if len(groups) < 3:
        raise RuntimeError("Need at least 3 patient groups for train/val/test split.")
    train_groups, test_groups = train_test_split(groups, test_size=test_size, random_state=RANDOM_SEED, shuffle=True)
    if len(train_groups) < 2:
        raise RuntimeError("Not enough training groups after test split.")
    train_groups, val_groups = train_test_split(train_groups, test_size=val_size, random_state=RANDOM_SEED, shuffle=True)
    train_groups, val_groups, test_groups = set(train_groups), set(val_groups), set(test_groups)
    train = [g for g in graphs if get_group_id_for_split(g) in train_groups]
    val = [g for g in graphs if get_group_id_for_split(g) in val_groups]
    test = [g for g in graphs if get_group_id_for_split(g) in test_groups]
    return train, val, test


def fit_scaler(graphs):
    X_all = np.vstack([g["X"] for g in graphs])
    scaler = StandardScaler()
    scaler.fit(X_all)
    return scaler


def apply_scaler(graphs, scaler):
    new_graphs = []
    for g in graphs:
        g2 = dict(g)
        g2["X"] = scaler.transform(g["X"]).astype(np.float32)
        new_graphs.append(g2)
    return new_graphs


# ============================================================
# MODEL
# ============================================================

class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, X, A):
        h = self.linear(A @ X)
        return self.norm(h)






def compute_pos_weight(graphs):
    y = np.concatenate([g["y"] for g in graphs])
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    if pos == 0:
        return torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
    w = neg / (pos + 1e-8)
    w = min(float(w), MAX_POS_WEIGHT)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def pairwise_ranking_loss_from_logits(logits, targets, margin=0.15):
    targets = targets.float()
    pos = logits[targets > 0.5]
    neg = logits[targets <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_zeros(())
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    return F.relu(margin - diff).mean()


class FocalDiceRankingLoss(nn.Module):
    def __init__(self, pos_weight=None, gamma=FOCAL_GAMMA, dice_weight=DICE_LOSS_WEIGHT, ranking_weight=RANKING_LOSS_WEIGHT):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.ranking_weight = ranking_weight

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1.0 - prob) * (1.0 - targets)
        focal_loss = (((1.0 - pt) ** self.gamma) * bce).mean()
        smooth = 1.0
        intersection = torch.sum(prob * targets)
        dice = (2.0 * intersection + smooth) / (torch.sum(prob) + torch.sum(targets) + smooth)
        dice_loss = 1.0 - dice
        rank_loss = pairwise_ranking_loss_from_logits(logits, targets)
        return focal_loss + self.dice_weight * dice_loss + self.ranking_weight * rank_loss












def save_node_predictions(model, graphs, filename):
    rows = []
    for g in graphs:
        p = predict_graph(model, g)
        y = g["y"].astype(int)
        for i in range(len(y)):
            row = {
                "patient_id": g["patient_id"],
                "source": g["source"],
                "node_index": i,
                "channel": g["channel_names"][i] if i < len(g["channel_names"]) else f"node_{i}",
                "true_label": int(y[i]),
                "predicted_probability": float(p[i]),
            }
            for col in BASE_FEATURE_COLUMNS:
                if col in g["X_df"].columns:
                    row[col] = float(g["X_df"][col].iloc[i])
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, filename), index=False)


# ============================================================
# BASELINES AND HYBRID SCORES
# ============================================================

def get_feature_score_arrays(graphs, feature_name):
    scores = []
    for g in graphs:
        vals = np.asarray(g["X_df"][feature_name].values, dtype=float)
        scores.append(normalize_vector(vals))
    return scores


def get_model_score_arrays(model, graphs_scaled):
    return [predict_graph(model, g) for g in graphs_scaled]




def evaluate_feature_only_baseline(graphs, feature_name, method_name):
    return evaluate_score_arrays(graphs, get_feature_score_arrays(graphs, feature_name), method_name=method_name, threshold=0.5)


def run_baselines(graphs, dataset_name):
    rows = []
    baselines = [
        ("onset_score", "onset_only"),
        ("energy_score", "energy_only"),
        ("hfa_score", "hfa_only"),
        ("centrality_score", "centrality_only"),
        ("heuristic_composite_score", "heuristic_composite_only"),
    ]
    for col, label in baselines:
        if col in graphs[0]["X_df"].columns:
            try:
                row = evaluate_feature_only_baseline(graphs, col, label)
                row["dataset"] = dataset_name
                rows.append(row)
            except Exception:
                continue
    return pd.DataFrame(rows)


def combine_score_components(component_scores, weights):
    names = list(weights.keys())
    n_graphs = len(component_scores[names[0]])
    combined = []
    for gi in range(n_graphs):
        s = None
        for name in names:
            vals = normalize_vector(component_scores[name][gi])
            if s is None:
                s = weights[name] * vals
            else:
                s = s + weights[name] * vals
        combined.append(normalize_vector(s))
    return combined






def simplex_weight_grid(component_names, step=HYBRID_GRID_STEP):
    units = int(round(1.0 / step))
    names = list(component_names)
    weights = []
    def rec(prefix, remaining, idx):
        if idx == len(names) - 1:
            weights.append(prefix + [remaining])
            return
        for v in range(remaining + 1):
            rec(prefix + [v], remaining - v, idx + 1)
    rec([], units, 0)
    for w in weights:
        yield {name: val / units for name, val in zip(names, w)}


def tune_hybrid_weights(val_graphs_raw, val_component_scores, component_names=None):
    if component_names is None:
        component_names = list(val_component_scores.keys())
    best = None
    for weights in simplex_weight_grid(component_names, step=HYBRID_GRID_STEP):
        if max(weights.values()) <= 0:
            continue
        scores = combine_score_components(val_component_scores, weights)
        thr, _ = find_best_threshold_for_score_arrays(val_graphs_raw, scores)
        metrics = evaluate_score_arrays(val_graphs_raw, scores, "validation_hybrid", threshold=thr)
        score = ranking_selection_score(metrics)
        if best is None or score > best["selection_score"]:
            best = {"weights": weights, "threshold": thr, "metrics": metrics, "selection_score": score}
    return best


# ============================================================
# PLOTS AND OUTPUT TABLES
# ============================================================

def plot_history(history, filename_prefix):
    if history.empty:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["train_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{filename_prefix}: Training loss")
    save_fig(f"{filename_prefix}_training_loss.png")
    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["val_auc"], label="Validation AUC")
    plt.plot(history["epoch"], history["val_ap"], label="Validation AUPRC")
    plt.plot(history["epoch"], history["val_top5_hit"], label="Validation Top-5 hit")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title(f"{filename_prefix}: Validation metrics")
    plt.legend()
    save_fig(f"{filename_prefix}_validation_metrics.png")


def plot_metric_comparison(baseline_df, model_metrics, filename, title):
    if baseline_df is None:
        baseline_df = pd.DataFrame()
    rows = baseline_df.copy()
    model_row = {
        "method": "pVEP-EZNet",
        "auc": model_metrics.get("auc", np.nan),
        "average_precision": model_metrics.get("average_precision", np.nan),
        "f1": model_metrics.get("f1", np.nan),
        "mean_top3_hit": model_metrics.get("mean_top3_hit", np.nan),
        "mean_top5_hit": model_metrics.get("mean_top5_hit", np.nan),
    }
    if "method" not in rows.columns and "baseline" in rows.columns:
        rows = rows.rename(columns={"baseline": "method"})
    rows = pd.concat([rows, pd.DataFrame([model_row])], ignore_index=True)
    if "method" not in rows.columns or "auc" not in rows.columns:
        return
    plt.figure(figsize=(10, 5))
    plt.bar(rows["method"], rows["auc"])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("AUC")
    plt.title(title)
    save_fig(filename)


def plot_ablation_metric_heatmap(df, filename="HUP_metric_heatmap.png"):
    metrics = ["auc", "average_precision", "f1", "mean_top3_hit", "mean_top5_hit", "mean_top10_hit", "mean_rank_first_positive", "mean_mrr"]
    metrics = [m for m in metrics if m in df.columns]
    if not metrics or "method" not in df.columns:
        return
    plot_df = df[["method"] + metrics].copy()
    mat = []
    for m in metrics:
        vals = plot_df[m].astype(float).values
        if m == "mean_rank_first_positive":
            vals = -vals
        mn, mx = np.nanmin(vals), np.nanmax(vals)
        norm = np.zeros_like(vals) if mx - mn < 1e-12 else (vals - mn) / (mx - mn)
        mat.append(norm)
    mat = np.vstack(mat).T
    plt.figure(figsize=(12, max(4, 0.45 * len(plot_df))))
    plt.imshow(mat, aspect="auto")
    plt.yticks(np.arange(len(plot_df)), plot_df["method"].values)
    plt.xticks(np.arange(len(metrics)), [m.replace("mean_", "").replace("average_precision", "AUPRC").replace("_", " ") for m in metrics], rotation=35, ha="right")
    plt.colorbar(label="Normalized performance; rank inverted")
    plt.title("HUP ablation metric heatmap")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            raw = plot_df[metrics[j]].iloc[i]
            plt.text(j, i, f"{raw:.2f}", ha="center", va="center", fontsize=7)
    save_fig(filename)


def plot_topk_hit_curves(df, filename="HUP_topk_hit_curves.png"):
    ks = [k for k in TOP_K_LIST if f"mean_top{k}_hit" in df.columns]
    if not ks or "method" not in df.columns:
        return
    keep_methods = [
        "HUP-only graph model",
        "VEP-pretrained + HUP-finetuned GCN",
        "HUP + VEP-transfer score ensemble",
        "onset only",
        "energy only",
        "heuristic composite only",
        "heuristic_composite_score",
    ]
    plot_df = df[df["method"].isin(keep_methods)].copy()
    if plot_df.empty:
        plot_df = df.copy()
    plt.figure(figsize=(8, 5))
    for _, row in plot_df.iterrows():
        y = [row[f"mean_top{k}_hit"] for k in ks]
        plt.plot(ks, y, marker="o", label=row["method"])
    plt.xlabel("k")
    plt.ylabel("Mean Top-k hit rate")
    plt.ylim(0, 1.05)
    plt.title("HUP SOZ ranking: Top-k hit curves")
    plt.legend(fontsize=8)
    save_fig(filename)


def plot_precision_recall_curves(method_score_dict, graphs_raw, filename="HUP_precision_recall_curves.png"):
    plt.figure(figsize=(7, 5))
    for method, scores in method_score_dict.items():
        all_y, all_p = [], []
        for g, p in zip(graphs_raw, scores):
            all_y.extend(g["y"].astype(int).tolist())
            all_p.extend(normalize_vector(p).tolist())
        all_y = np.asarray(all_y).astype(int)
        all_p = np.asarray(all_p).astype(float)
        if np.sum(all_y) == 0:
            continue
        prec, rec, _ = precision_recall_curve(all_y, all_p)
        ap = safe_ap(all_y, all_p)
        plt.plot(rec, prec, label=f"{method} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("HUP precision-recall curves")
    plt.legend(fontsize=8)
    save_fig(filename)


def plot_pooled_roc_pr_for_scores(graphs_raw, score_arrays, filename="HUP_pooled_ROC_PR_curves.png"):
    all_y, all_p = [], []
    for g, p in zip(graphs_raw, score_arrays):
        all_y.extend(g["y"].astype(int).tolist())
        all_p.extend(normalize_vector(p).tolist())
    all_y = np.asarray(all_y).astype(int)
    all_p = np.asarray(all_p).astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axes[0]
    if len(np.unique(all_y)) > 1:
        fpr, tpr, _ = roc_curve(all_y, all_p)
        auc = roc_auc_score(all_y, all_p)
        ax.plot(fpr, tpr, linewidth=2.2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Held-out HUP ROC curve")
    ax.legend(frameon=False)
    ax = axes[1]
    if np.sum(all_y) > 0:
        prec, rec, _ = precision_recall_curve(all_y, all_p)
        ap = average_precision_score(all_y, all_p)
        ax.plot(rec, prec, linewidth=2.2, label=f"AUPRC = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Held-out HUP precision-recall curve")
    ax.legend(frameon=False)
    save_fig(filename)


def plot_confusion_calibration_for_scores(graphs_raw, score_arrays, threshold=0.5, filename="HUP_confusion_calibration.png"):
    all_y, all_p = [], []
    for g, p in zip(graphs_raw, score_arrays):
        all_y.extend(g["y"].astype(int).tolist())
        all_p.extend(normalize_vector(p).tolist())
    all_y = np.asarray(all_y).astype(int)
    all_p = np.asarray(all_p).astype(float)
    pred = (all_p >= threshold).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    cm = confusion_matrix(all_y, pred, labels=[0, 1])
    ax = axes[0]
    im = ax.imshow(cm, aspect="equal")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Non-SOZ", "SOZ"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Non-SOZ", "SOZ"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix, threshold={threshold:.2f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax = axes[1]
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Ideal")
    bins = np.linspace(0, 1, CALIBRATION_BINS + 1)
    xs, ys, counts = [], [], []
    for i in range(CALIBRATION_BINS):
        mask = (all_p >= bins[i]) & ((all_p < bins[i + 1]) if i < CALIBRATION_BINS - 1 else (all_p <= bins[i + 1]))
        if np.sum(mask) > 0:
            xs.append(float(np.mean(all_p[mask])))
            ys.append(float(np.mean(all_y[mask])))
            counts.append(int(np.sum(mask)))
    if xs:
        ax.plot(xs, ys, marker="o", linewidth=2, label="Model")
        for x, yv, c in zip(xs, ys, counts):
            ax.text(x, yv, str(c), fontsize=7)
    ax.set_xlabel("Predicted score")
    ax.set_ylabel("Observed SOZ frequency")
    ax.set_title("Calibration curve")
    ax.legend(frameon=False)
    save_fig(filename)




def plot_score_density_soz_vs_nonsoz(graphs_raw, score_arrays, filename="HUP_score_density_SOZ_vs_nonSOZ.png"):
    y, p = [], []
    for g, s in zip(graphs_raw, score_arrays):
        y.extend(g["y"].astype(int).tolist())
        p.extend(normalize_vector(s).tolist())
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    plt.figure(figsize=(7, 4.8))
    if np.sum(y == 0) > 0:
        plt.hist(p[y == 0], bins=30, alpha=0.65, density=True, label="Non-SOZ")
    if np.sum(y == 1) > 0:
        plt.hist(p[y == 1], bins=20, alpha=0.65, density=True, label="SOZ")
    plt.xlabel("Predicted score")
    plt.ylabel("Density")
    plt.title("Score distribution for SOZ and non-SOZ contacts")
    plt.legend(frameon=False)
    save_fig(filename)


def generate_publishable_result_pack(hup_test_raw, hybrid_scores_test, threshold):
    plot_pooled_roc_pr_for_scores(hup_test_raw, hybrid_scores_test)
    plot_confusion_calibration_for_scores(hup_test_raw, hybrid_scores_test, threshold=threshold)
    plot_bootstrap_metric_ci(hup_test_raw, hybrid_scores_test)
    plot_score_density_soz_vs_nonsoz(hup_test_raw, hybrid_scores_test)


def make_node_prediction_dataframe(graphs_raw, score_arrays, method_name="hybrid_pvep_eznet"):
    rows = []
    for g, scores in zip(graphs_raw, score_arrays):
        scores = normalize_vector(scores)
        coords = g.get("electrode_coords", make_synthetic_brain_coords(len(scores)))
        channel_to_region = g.get("channel_to_region", np.zeros(len(scores), dtype=int))
        region_names = g.get("region_names", pseudo_region_names())
        y = g.get("y", np.zeros(len(scores))).astype(int)
        for i, ch in enumerate(g["channel_names"]):
            r = int(channel_to_region[i]) if i < len(channel_to_region) else 0
            rows.append({
                "method": method_name,
                "patient_id": g["patient_id"],
                "channel": ch,
                "true_soz": int(y[i]) if i < len(y) else 0,
                "score": float(scores[i]),
                "x": float(coords[i, 0]),
                "y_coord": float(coords[i, 1]),
                "z": float(coords[i, 2]),
                "pseudo_region_index": r,
                "pseudo_region_name": region_names[r] if r < len(region_names) else f"Region-{r}",
                "coordinate_source": g.get("coordinate_source", "unknown"),
            })
    return pd.DataFrame(rows)


def make_region_prediction_dataframe(graphs_raw, score_arrays, method_name="hybrid_pvep_eznet"):
    rows = []
    for g, scores in zip(graphs_raw, score_arrays):
        scores = normalize_vector(scores)
        if "electrode_coords" not in g:
            add_structural_annotations_to_graph(g, None)
        channel_to_region = g["channel_to_region"]
        region_names = g["region_names"]
        region_coords = g["region_coords"]
        region_scores = aggregate_channel_scores_to_regions(scores, channel_to_region, len(region_names))
        y = g.get("y", np.zeros(len(scores))).astype(int)
        region_soz = aggregate_channel_scores_to_regions(y, channel_to_region, len(region_names))
        region_soz_binary = (region_soz > 0).astype(int)
        for r, name in enumerate(region_names):
            rows.append({
                "method": method_name,
                "patient_id": g["patient_id"],
                "pseudo_region_index": r,
                "pseudo_region_name": name,
                "region_score": float(region_scores[r]),
                "region_has_soz_channel": int(region_soz_binary[r]),
                "region_x": float(region_coords[r, 0]),
                "region_y": float(region_coords[r, 1]),
                "region_z": float(region_coords[r, 2]),
                "n_channels_mapped": int(np.sum(channel_to_region == r)),
                "coordinate_source": g.get("coordinate_source", "unknown"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["region_rank_within_patient"] = df.groupby("patient_id")["region_score"].rank(ascending=False, method="first")
    return df



def save_hup_structure_outputs(hup_test_raw, hybrid_scores_test):
    """Save HUP channel- and region-level structure-aware prediction tables.

        """
    node_df = make_node_prediction_dataframe(
        hup_test_raw, hybrid_scores_test, "Hybrid pVEP-EZNet"
    )
    region_df = make_region_prediction_dataframe(
        hup_test_raw, hybrid_scores_test, "Hybrid pVEP-EZNet"
    )
    node_df.to_csv(
        os.path.join(OUTPUT_DIR, "HUP_test_channel_structure_predictions.csv"),
        index=False,
    )
    region_df.to_csv(
        os.path.join(OUTPUT_DIR, "HUP_test_region_predictions.csv"),
        index=False,
    )
    return node_df, region_df


# ============================================================
# ABLATION STUDY
# ============================================================



# ============================================================
# JOURNAL-V3 FINAL DEFINITIONS
# These are the final active model/training definitions used by main().
# ============================================================

from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

# ---- Journal-strength settings ----
# HFA is kept consistent with the 1-120 Hz preprocessing band.
HFA_LOW_FREQ = 80.0
HFA_HIGH_FREQ = 120.0

# Avoid deterministic duplication of the hand-crafted epileptogenicity score.
# The heuristic composite is retained only in X_df for an explicit baseline/audit;
# it is not fed to the neural model. HUP does not receive a fabricated x0 proxy.
BASE_FEATURE_COLUMNS = [
    "onset_score",
    "onset_z_score",
    "energy_score",
    "amplitude_score",
    "gamma_score",
    "hfa_score",
    "centrality_score",
]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + LAPLACIAN_PE_COLUMNS

# VEP graphs without simulated source time-series make cross-domain feature
# semantics weak and can create near-direct x0 label leakage. For the final
# journal experiment keep this True. If too few VEP runs survive, set False
# and report that limitation explicitly.
REQUIRE_VEP_SOURCE_TS = True

# Graph regularisation / attention
GRAPH_TOP_K = 16
GRAPH_MIN_EDGE = 0.02
ATTENTION_HEADS = 4

# Ranking-first objective
FOCAL_WEIGHT_V2 = 0.30
DICE_WEIGHT_V2 = 0.08
HARD_PAIR_WEIGHT_V2 = 0.28
LISTWISE_WEIGHT_V2 = 0.18
SOFT_RANK_WEIGHT_V2 = 0.24
HARD_NEGATIVE_K = 24
PAIRWISE_MARGIN_V2 = 0.20
SOFT_RANK_TAU = 0.25

# Simulation-to-real domain alignment during HUP fine-tuning.
USE_DOMAIN_MMD = True
DOMAIN_MMD_WEIGHT = 0.025
DOMAIN_MMD_MAX_NODES = 64
GLOBAL_VEP_DOMAIN_GRAPHS = None

# Optional long-running journal experiments.
RUN_FULL_ARCH_ABLATIONS = False
RUN_REPEATED_GROUP_CV = False
CV_SEEDS = [42, 52, 62, 72, 82]
CV_FOLDS = 5

# Publication-oriented controlled VEP/onset sensitivity.
# Configurations 1--4 use the identical architecture, patients, graph, loss,
# optimizer and validation protocol. Only onset features and VEP initialization
# are toggled. MMD is disabled in configs 1--4 so the comparison isolates VEP
# *pretraining* itself. The fifth row is the complete proposed model.
RUN_CONTROLLED_VEP_ONSET_SENSITIVITY = True
SENSITIVITY_PRETRAIN_EPOCHS = PRETRAIN_EPOCHS
SENSITIVITY_FINETUNE_EPOCHS = FINETUNE_EPOCHS
CONTROL_NO_ONSET_BASE_COLUMNS = [
    "energy_score", "amplitude_score", "gamma_score", "hfa_score", "centrality_score"
]
CONTROL_WITH_ONSET_BASE_COLUMNS = [
    "onset_score", "onset_z_score",
    "energy_score", "amplitude_score", "gamma_score", "hfa_score", "centrality_score"
]

# The final paper should report the learned model first. The optional hybrid
# is retained as a supplementary comparison rather than being the sole claim.
PRIMARY_REPORT_MODEL = "transfer_learned"


def extract_signal_features(data, sfreq, ch_names):
    """Feature extraction with HFA band consistent with the 120-Hz low-pass."""
    n_ch, n_samp = data.shape
    baseline_samples = int(BASELINE_DURATION_SEC * sfreq)
    baseline_samples = min(baseline_samples, max(1, n_samp // 3))
    onset_times, onset_zmax = detect_onsets(data, sfreq)
    rows = []
    for ch in range(n_ch):
        sig = np.nan_to_num(data[ch])
        base = sig[:baseline_samples]
        event = sig[baseline_samples:]
        if len(event) < int(2 * sfreq):
            event = sig
        baseline_energy = np.mean(base ** 2) + 1e-12
        event_energy = np.mean(event ** 2) + 1e-12
        baseline_amp = np.mean(np.abs(base)) + 1e-12
        event_amp = np.mean(np.abs(event)) + 1e-12
        theta = bandpower(sig, sfreq, 4, 8)
        alpha = bandpower(sig, sfreq, 8, 13)
        beta = bandpower(sig, sfreq, 13, 30)
        gamma = bandpower(sig, sfreq, 30, 80)
        hfa_hi = min(HFA_HIGH_FREQ, sfreq / 2.0 - 1.0)
        hfa = bandpower(sig, sfreq, HFA_LOW_FREQ, hfa_hi) if hfa_hi > HFA_LOW_FREQ else 0.0
        total = theta + alpha + beta + gamma + hfa + 1e-12
        rows.append({
            "channel": ch_names[ch],
            "onset_time_sec": onset_times[ch],
            "onset_zmax": onset_zmax[ch],
            "energy_ratio": event_energy / baseline_energy,
            "amplitude_ratio": event_amp / baseline_amp,
            "theta_power": theta,
            "alpha_power": alpha,
            "beta_power": beta,
            "gamma_power": gamma,
            "hfa_power": hfa,
            "gamma_ratio": gamma / total,
            "hfa_ratio": hfa / total,
        })
    return pd.DataFrame(rows)


def convert_raw_features_to_node_features(raw_df, W):
    """Build real-iEEG biomarkers; no synthetic HUP x0 proxy is created."""
    n = len(raw_df)
    onset = raw_df["onset_time_sec"].values
    valid = ~np.isnan(onset)
    onset_score = np.zeros(n)
    if np.sum(valid) > 1:
        mn, mx = np.min(onset[valid]), np.max(onset[valid])
        onset_score[valid] = 1.0 - ((onset[valid] - mn) / (mx - mn + 1e-8))
    elif np.sum(valid) == 1:
        onset_score[np.where(valid)[0][0]] = 1.0
    onset_z_score = normalize_vector(raw_df["onset_zmax"].values)
    energy_score = normalize_vector(np.log1p(raw_df["energy_ratio"].values))
    amplitude_score = normalize_vector(np.log1p(raw_df["amplitude_ratio"].values))
    gamma_score = normalize_vector(raw_df["gamma_ratio"].values)
    hfa_score = normalize_vector(raw_df["hfa_ratio"].values)
    centrality_score = normalize_vector(W.sum(axis=1))
    heuristic_composite_score = normalize_vector(
        0.30 * onset_score +
        0.20 * onset_z_score +
        0.15 * energy_score +
        0.10 * amplitude_score +
        0.15 * hfa_score +
        0.05 * gamma_score +
        0.05 * centrality_score
    )
    # No HUP x0 is fabricated here. True Epileptor x0 is retained only in VEP.
    return pd.DataFrame({
        "onset_score": onset_score,
        "onset_z_score": onset_z_score,
        "energy_score": energy_score,
        "amplitude_score": amplitude_score,
        "gamma_score": gamma_score,
        "hfa_score": hfa_score,
        "centrality_score": centrality_score,
        "heuristic_composite_score": heuristic_composite_score,
    })


def build_vep_features_from_timeseries_or_params(W, x0, x0_norm, source_ts):
    """Build cross-domain features without feeding the true VEP x0 value directly."""
    n_nodes = W.shape[0]
    centrality_score = normalize_vector(W.sum(axis=1))
    if source_ts is None:
        # For fallback runs, do not replicate x0 into every feature; this avoids
        # an almost direct label proxy. These graphs are normally skipped when
        # REQUIRE_VEP_SOURCE_TS=True.
        onset_score = np.zeros(n_nodes)
        onset_z_score = np.zeros(n_nodes)
        energy_score = np.zeros(n_nodes)
        amplitude_score = np.zeros(n_nodes)
        gamma_score = np.zeros(n_nodes)
        hfa_score = np.zeros(n_nodes)
    else:
        if source_ts.shape[0] != n_nodes:
            source_ts = source_ts[:n_nodes, :] if source_ts.shape[0] > n_nodes else None
        if source_ts is None:
            onset_score = np.zeros(n_nodes)
            onset_z_score = np.zeros(n_nodes)
            energy_score = np.zeros(n_nodes)
            amplitude_score = np.zeros(n_nodes)
            gamma_score = np.zeros(n_nodes)
            hfa_score = np.zeros(n_nodes)
        else:
            data = np.nan_to_num(source_ts)
            amp = np.mean(np.abs(data), axis=1)
            energy = np.mean(data ** 2, axis=1)
            peak = np.max(np.abs(data), axis=1)
            variability = np.std(data, axis=1)
            onset_times = np.full(n_nodes, np.nan)
            for i in range(n_nodes):
                sig = np.abs(data[i])
                base_n = max(4, int(0.20 * len(sig)))
                base = sig[:base_n]
                thr = np.mean(base) + 3.0 * (np.std(base) + 1e-8)
                active = sig > thr
                min_len = max(3, int(0.01 * data.shape[1]))
                for k in range(base_n, max(base_n + 1, data.shape[1] - min_len)):
                    if np.all(active[k:k + min_len]):
                        onset_times[i] = k
                        break
            valid = ~np.isnan(onset_times)
            onset_score = np.zeros(n_nodes)
            if np.sum(valid) > 1:
                mn, mx = np.nanmin(onset_times), np.nanmax(onset_times)
                onset_score[valid] = 1.0 - ((onset_times[valid] - mn) / (mx - mn + 1e-8))
            elif np.sum(valid) == 1:
                onset_score[np.where(valid)[0][0]] = 1.0
            onset_z_score = normalize_vector(peak)
            energy_score = normalize_vector(np.log1p(energy))
            amplitude_score = normalize_vector(np.log1p(amp))
            gamma_score = normalize_vector(variability)
            hfa_score = normalize_vector(peak / (amp + 1e-8))
    heuristic_composite_score = normalize_vector(
        0.30 * onset_score + 0.20 * onset_z_score + 0.15 * energy_score +
        0.10 * amplitude_score + 0.15 * hfa_score + 0.05 * gamma_score +
        0.05 * centrality_score
    )
    # Store the *real* VEP x0 only for audit / auxiliary analysis, not FEATURE_COLUMNS.
    return pd.DataFrame({
        "x0": np.asarray(x0, dtype=float),
        "onset_score": onset_score,
        "onset_z_score": onset_z_score,
        "energy_score": energy_score,
        "amplitude_score": amplitude_score,
        "gamma_score": gamma_score,
        "hfa_score": hfa_score,
        "centrality_score": centrality_score,
        "heuristic_composite_score": heuristic_composite_score,
    })


def load_vep_cohort(root):
    print("\n====================================================")
    print("STAGE 1: Loading VEP cohort (journal-v3 corrected)")
    print("====================================================")
    if not os.path.exists(root):
        raise FileNotFoundError(f"VEP path not found: {root}")
    sub_dirs = find_subject_dirs_vep(root)
    print(f"Found VEP subject folders: {len(sub_dirs)}")
    patients = []
    skipped_no_ts = 0
    for sub_dir in sub_dirs:
        subject_id = os.path.basename(sub_dir)
        weights_path = find_connectome_weights_for_subject(sub_dir)
        if weights_path is None:
            continue
        try:
            W = load_weights_txt(weights_path)
        except Exception:
            continue
        n_nodes = W.shape[0]
        for param_file in find_epileptor_parameter_files(sub_dir):
            try:
                x0_raw, x0_norm, y, param_df = parse_epileptor_parameters(param_file, n_nodes=n_nodes)
                ts_path = find_matching_source_timeseries(param_file)
                source_ts = load_source_timeseries(ts_path)
                if REQUIRE_VEP_SOURCE_TS and source_ts is None:
                    skipped_no_ts += 1
                    continue
                X_df = build_vep_features_from_timeseries_or_params(W, x0_raw, x0_norm, source_ts)
                hyp_name = os.path.basename(os.path.dirname(os.path.dirname(param_file)))
                ses_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(param_file))))
                run_match = re.search(r"run-(\d+)", os.path.basename(param_file))
                run_name = f"run-{run_match.group(1)}" if run_match else "run-unknown"
                patient_id = f"{subject_id}_{ses_name}_{hyp_name}_{run_name}"
                channel_names = [f"region_{i:03d}" for i in range(n_nodes)]
                g = make_patient_graph(patient_id, W, X_df, y, channel_names, source="VEP")
                g["weights_path"] = weights_path
                g["parameter_path"] = param_file
                g["source_timeseries_path"] = ts_path
                g["vep_label_source"] = param_df.attrs.get("label_source", "unknown")
                g["x0_min"] = param_df.attrs.get("x0_min", np.nan)
                g["x0_max"] = param_df.attrs.get("x0_max", np.nan)
                g["x0_std"] = param_df.attrs.get("x0_std", np.nan)
                g["x0_unique_count"] = param_df.attrs.get("x0_unique_count", np.nan)
                g["ez_fraction"] = param_df.attrs.get("ez_fraction", float(np.mean(y)))
                patients.append(g)
            except Exception as e:
                print(f"Skipping VEP run {os.path.basename(param_file)}: {e}")
    if not patients:
        raise RuntimeError(
            "No valid VEP graphs were loaded. If your release does not include simulated source time-series, "
            "set REQUIRE_VEP_SOURCE_TS=False, but do not claim direct signal-domain transfer."
        )
    print(f"Loaded VEP graph instances: {len(patients)} | skipped without source TS: {skipped_no_ts}")
    audit_df = pd.DataFrame([{
        "patient_id": g["patient_id"],
        "nodes": len(g["y"]),
        "positive_EZ_nodes": int(np.sum(g["y"])),
        "ez_fraction": float(np.mean(g["y"])),
        "label_source": g.get("vep_label_source", "unknown"),
        "x0_min": g.get("x0_min", np.nan),
        "x0_max": g.get("x0_max", np.nan),
        "x0_std": g.get("x0_std", np.nan),
        "x0_unique_count": g.get("x0_unique_count", np.nan),
        "source_timeseries_path": g.get("source_timeseries_path", "")
    } for g in patients])
    audit_df.to_csv(os.path.join(OUTPUT_DIR, "VEP_loader_audit.csv"), index=False)
    if (audit_df["positive_EZ_nodes"] >= audit_df["nodes"]).any():
        raise RuntimeError("VEP label audit failed: at least one graph has all nodes labelled EZ.")
    return patients


def sparsify_adjacency(W, top_k=GRAPH_TOP_K, min_edge=GRAPH_MIN_EDGE):
    """Keep strongest neighbours per node; symmetrise afterwards."""
    W = normalize_matrix(W).copy()
    n = W.shape[0]
    if n <= 1:
        return W.astype(np.float32)
    out = np.zeros_like(W)
    k = min(top_k, max(1, n - 1))
    for i in range(n):
        idx = np.argsort(W[i])[::-1]
        idx = [j for j in idx if j != i and W[i, j] >= min_edge][:k]
        out[i, idx] = W[i, idx]
    out = np.maximum(out, out.T)
    np.fill_diagonal(out, 1.0)
    return out.astype(np.float32)


def graph_to_tensors(g):
    X = torch.tensor(g["X"], dtype=torch.float32, device=DEVICE)
    A = torch.tensor(sparsify_adjacency(g["W"]), dtype=torch.float32, device=DEVICE)
    y = torch.tensor(g["y"], dtype=torch.float32, device=DEVICE)
    return X, A, y


class WeightedGraphAttention(nn.Module):
    def __init__(self, hidden_dim, heads=ATTENTION_HEADS, dropout=DROPOUT):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by number of attention heads")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H, A):
        n = H.shape[0]
        q = self.q(H).view(n, self.heads, self.head_dim).transpose(0, 1)
        k = self.k(H).view(n, self.heads, self.head_dim).transpose(0, 1)
        v = self.v(H).view(n, self.heads, self.head_dim).transpose(0, 1)
        score = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(float(self.head_dim))
        edge = torch.clamp(A, min=0.0)
        mask = edge > 0
        edge_bias = torch.log(edge + 1e-6)
        score = score + edge_bias.unsqueeze(0)
        score = score.masked_fill(~mask.unsqueeze(0), -1e4)
        attn = torch.softmax(score, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(0, 1).contiguous().view(n, self.hidden_dim)
        out = self.o(out)
        return self.norm(H + self.dropout(out)), attn.mean(dim=0)


class PVEPEZNet(nn.Module):
    """Residual feature-gated GCN + weighted graph attention + local pathway."""
    def __init__(self, in_dim, hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
                 use_attention=True, use_local=True, use_gate=True, use_residual=True):
        super().__init__()
        self.use_attention = use_attention
        self.use_local = use_local
        self.use_gate = use_gate
        self.use_residual = use_residual
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.feature_gate = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid())
        self.gcn1 = DenseGCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = DenseGCNLayer(hidden_dim, hidden_dim)
        self.attn = WeightedGraphAttention(hidden_dim, heads=ATTENTION_HEADS, dropout=dropout)
        self.local_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.fusion_gate = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.Sigmoid())
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = dropout

    def forward(self, X, A, return_embedding=False):
        h0 = self.input_proj(X)
        if self.use_gate:
            h0 = h0 * self.feature_gate(X)
        h1 = F.gelu(self.gcn1(h0, graph_normalized_tensor(A)))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2_raw = self.gcn2(h1, graph_normalized_tensor(A))
        h2 = F.gelu(h2_raw + h0) if self.use_residual else F.gelu(h2_raw)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        if self.use_attention:
            hatt, attn = self.attn(h2, A)
        else:
            hatt, attn = h2, None
        local = self.local_mlp(X) if self.use_local else torch.zeros_like(h2)
        fg = self.fusion_gate(torch.cat([h2, hatt, local], dim=-1))
        graph_h = fg * hatt + (1.0 - fg) * h2
        h = torch.cat([graph_h, local], dim=-1)
        logits = self.out(h).squeeze(-1)
        if return_embedding:
            return logits, graph_h, attn
        return logits


def graph_normalized_tensor(A):
    """Symmetric normalisation on a torch adjacency tensor."""
    A = torch.clamp(A, min=0.0)
    eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    A = torch.maximum(A, eye)
    deg = A.sum(dim=1)
    d = torch.rsqrt(deg + 1e-8)
    return d[:, None] * A * d[None, :]


def hard_pairwise_ranking_loss(logits, targets, margin=PAIRWISE_MARGIN_V2, hard_k=HARD_NEGATIVE_K):
    pos = logits[targets > 0.5]
    neg = logits[targets <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_zeros(())
    k = min(hard_k, neg.numel())
    hard_neg = torch.topk(neg, k=k, largest=True).values
    diff = pos.unsqueeze(1) - hard_neg.unsqueeze(0)
    return F.softplus(margin - diff).mean()


def listwise_soz_loss(logits, targets):
    pos_mask = targets > 0.5
    if pos_mask.sum() == 0:
        return logits.new_zeros(())
    log_probs = F.log_softmax(logits, dim=0)
    target_dist = pos_mask.float() / pos_mask.float().sum()
    return -(target_dist * log_probs).sum()


def soft_first_rank_loss(logits, targets, tau=SOFT_RANK_TAU):
    pos = logits[targets > 0.5]
    neg = logits[targets <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_zeros(())
    # Smooth rank of every positive relative to negatives.
    soft_ranks = 1.0 + torch.sigmoid((neg.unsqueeze(0) - pos.unsqueeze(1)) / tau).sum(dim=1)
    # Soft minimum approximates the rank of the first/best SOZ contact.
    best_rank = -tau * torch.logsumexp(-soft_ranks / tau, dim=0)
    best_rank = torch.clamp(best_rank, min=1.0)
    return torch.log(best_rank)


class RankingFirstLossV2(nn.Module):
    def __init__(self, pos_weight=None, disable_ranking=False):
        super().__init__()
        self.pos_weight = pos_weight
        self.disable_ranking = disable_ranking

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1.0 - prob) * (1.0 - targets)
        focal = (((1.0 - pt) ** FOCAL_GAMMA) * bce).mean()
        smooth = 1.0
        inter = torch.sum(prob * targets)
        dice = 1.0 - (2.0 * inter + smooth) / (torch.sum(prob) + torch.sum(targets) + smooth)
        if self.disable_ranking:
            return focal + 0.10 * dice
        pair = hard_pairwise_ranking_loss(logits, targets)
        list_loss = listwise_soz_loss(logits, targets)
        soft_rank = soft_first_rank_loss(logits, targets)
        return (
            FOCAL_WEIGHT_V2 * focal + DICE_WEIGHT_V2 * dice +
            HARD_PAIR_WEIGHT_V2 * pair + LISTWISE_WEIGHT_V2 * list_loss +
            SOFT_RANK_WEIGHT_V2 * soft_rank
        )


def _subsample_nodes(x, max_nodes=DOMAIN_MMD_MAX_NODES):
    if x.shape[0] <= max_nodes:
        return x
    idx = torch.randperm(x.shape[0], device=x.device)[:max_nodes]
    return x[idx]


def rbf_mmd(x, y):
    x = _subsample_nodes(x)
    y = _subsample_nodes(y)
    z = torch.cat([x, y], dim=0)
    with torch.no_grad():
        d2 = torch.cdist(z, z, p=2).pow(2)
        nz = d2[d2 > 0]
        sigma2 = torch.median(nz) if nz.numel() else torch.tensor(1.0, device=z.device)
        sigma2 = torch.clamp(sigma2, min=1e-3)
    def kernel(a, b):
        d = torch.cdist(a, b, p=2).pow(2)
        return torch.exp(-d / (2.0 * sigma2))
    return kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean()


def _model_flags(ablation="full"):
    return {
        "use_attention": ablation != "no_attention",
        "use_local": ablation != "no_local",
        "use_gate": ablation != "no_feature_gate",
        "use_residual": ablation != "no_residual",
    }


def predict_graph(model, g):
    model.eval()
    X, A, _ = graph_to_tensors(g)
    with torch.no_grad():
        logits = model(X, A)
        return torch.sigmoid(logits).detach().cpu().numpy()


def aggregate_predictions_by_patient(graphs, score_arrays):
    """Average repeated-seizure predictions by normalized contact name per patient."""
    store = defaultdict(lambda: defaultdict(lambda: {"scores": [], "labels": []}))
    for g, scores in zip(graphs, score_arrays):
        sid = get_group_id_for_split(g)
        for ch, y, s in zip(g["channel_names"], g["y"], scores):
            key = normalize_contact_name(ch)
            store[sid][key]["scores"].append(float(s))
            store[sid][key]["labels"].append(int(y))
    out = []
    for sid, contacts in store.items():
        names, y, p = [], [], []
        for ch_key, vals in contacts.items():
            names.append(ch_key)
            y.append(int(max(vals["labels"])))
            p.append(float(np.mean(vals["scores"])))
        out.append({"patient_id": sid, "channel_names": names,
                    "y": np.asarray(y, dtype=int), "p": np.asarray(p, dtype=float)})
    return out


def _metrics_from_patient_prediction_objects(patient_objs, threshold=0.5):
    all_y = np.concatenate([o["y"] for o in patient_objs]) if patient_objs else np.array([], dtype=int)
    all_p = np.concatenate([o["p"] for o in patient_objs]) if patient_objs else np.array([], dtype=float)
    rows = []
    for o in patient_objs:
        y, p = o["y"], o["p"]
        row = {
            "patient_id": o["patient_id"],
            "auc": safe_auc(y, p),
            "average_precision": safe_ap(y, p),
            "mean_rank_first_positive": mean_rank_first_positive(y, p),
            "mrr": reciprocal_rank_first_positive(y, p),
        }
        for k in TOP_K_LIST:
            row[f"top{k}_hit"] = topk_hit(y, p, k)
            row[f"recall_at_{k}"] = recall_at_k(y, p, k)
            row[f"ndcg_at_{k}"] = ndcg_at_k(y, p, k)
        rows.append(row)
    pdf = pd.DataFrame(rows)
    pred = (all_p >= threshold).astype(int) if len(all_p) else np.array([], dtype=int)
    if len(all_y):
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_y, pred, average="binary", zero_division=0
        )
        acc = accuracy_score(all_y, pred)
        brier = brier_score_loss(all_y, all_p) if len(np.unique(all_y)) > 1 else np.nan
    else:
        precision = recall = f1 = acc = brier = np.nan
    m = {
        "auc": safe_auc(all_y, all_p) if len(all_y) else np.nan,
        "average_precision": safe_ap(all_y, all_p) if len(all_y) else np.nan,
        "accuracy": float(acc), "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "threshold": float(threshold), "brier_score": float(brier) if np.isfinite(brier) else np.nan,
        "n_patients": int(len(patient_objs)),
    }
    if not pdf.empty:
        for k in TOP_K_LIST:
            m[f"mean_top{k}_hit"] = float(pdf[f"top{k}_hit"].mean())
            m[f"mean_recall_at_{k}"] = float(pdf[f"recall_at_{k}"].mean())
            m[f"mean_ndcg_at_{k}"] = float(pdf[f"ndcg_at_{k}"].mean())
        m["mean_rank_first_positive"] = float(pdf["mean_rank_first_positive"].mean())
        m["median_rank_first_positive"] = float(pdf["mean_rank_first_positive"].median())
        m["mean_mrr"] = float(pdf["mrr"].mean())
    return m, pdf


def evaluate_graphs(model, graphs, quiet=False, threshold=0.5):
    score_arrays = [predict_graph(model, g) for g in graphs]
    # HUP is evaluated at patient level; VEP stays graph-instance level.
    if graphs and all(g.get("source") == "HUP" for g in graphs):
        pobj = aggregate_predictions_by_patient(graphs, score_arrays)
        metrics, _ = _metrics_from_patient_prediction_objects(pobj, threshold=threshold)
    else:
        pobj = [{"patient_id": g["patient_id"], "y": g["y"].astype(int), "p": np.asarray(p)}
                for g, p in zip(graphs, score_arrays)]
        metrics, _ = _metrics_from_patient_prediction_objects(pobj, threshold=threshold)
    if not quiet:
        print("\nEvaluation:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, (float, int, np.floating)) else f"{k}: {v}")
    return metrics


def evaluate_score_arrays(graphs_raw, score_arrays, method_name="score", threshold=0.5):
    if graphs_raw and all(g.get("source") == "HUP" for g in graphs_raw):
        pobj = aggregate_predictions_by_patient(graphs_raw, score_arrays)
    else:
        pobj = [{"patient_id": g["patient_id"], "y": g["y"].astype(int), "p": normalize_vector(p)}
                for g, p in zip(graphs_raw, score_arrays)]
    out, _ = _metrics_from_patient_prediction_objects(pobj, threshold=threshold)
    out["method"] = method_name
    return out


def choose_selection_score(val_metrics):
    """One consistent ranking-first objective for checkpointing."""
    mrr = val_metrics.get("mean_mrr", np.nan)
    hit1 = val_metrics.get("mean_top1_hit", np.nan)
    ndcg5 = val_metrics.get("mean_ndcg_at_5", np.nan)
    ap = val_metrics.get("average_precision", np.nan)
    terms = []
    for v, w in [(mrr, 0.45), (hit1, 0.25), (ndcg5, 0.20), (ap, 0.10)]:
        if np.isfinite(v):
            terms.append(w * v)
    return float(np.sum(terms)) if terms else -1e9


def ranking_selection_score(metrics):
    return choose_selection_score(metrics)


def train_graph_model(train_graphs, val_graphs, epochs, in_dim, init_state=None,
                      stage_name="training", ablation="full"):
    global GLOBAL_VEP_DOMAIN_GRAPHS
    flags = _model_flags(ablation)
    model = PVEPEZNet(in_dim=in_dim, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, **flags).to(DEVICE)
    if init_state is not None:
        model.load_state_dict(init_state, strict=False)
    pos_weight = compute_pos_weight(train_graphs)
    loss_fn = RankingFirstLossV2(pos_weight=pos_weight, disable_ranking=(ablation == "no_rank_loss"))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10)
    best_score, best_state, wait = -1e9, None, 0
    history = []
    is_hup = len(train_graphs) > 0 and all(g.get("source") == "HUP" for g in train_graphs)
    source_pool = GLOBAL_VEP_DOMAIN_GRAPHS if (is_hup and USE_DOMAIN_MMD) else None
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for idx in np.random.permutation(len(train_graphs)):
            g = train_graphs[idx]
            X, A, y = graph_to_tensors(g)
            opt.zero_grad()
            logits, emb, _ = model(X, A, return_embedding=True)
            loss = loss_fn(logits, y)
            if source_pool:
                sg = source_pool[np.random.randint(len(source_pool))]
                Xs, As, _ = graph_to_tensors(sg)
                _, emb_s, _ = model(Xs, As, return_embedding=True)
                loss = loss + DOMAIN_MMD_WEIGHT * rbf_mmd(emb, emb_s)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            losses.append(float(loss.item()))
        val_metrics = evaluate_graphs(model, val_graphs, quiet=True)
        selection_score = choose_selection_score(val_metrics)
        scheduler.step(selection_score)
        row = {
            "stage": stage_name, "ablation": ablation, "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_auc": val_metrics.get("auc", np.nan),
            "val_ap": val_metrics.get("average_precision", np.nan),
            "val_mrr": val_metrics.get("mean_mrr", np.nan),
            "val_top1_hit": val_metrics.get("mean_top1_hit", np.nan),
            "val_top5_hit": val_metrics.get("mean_top5_hit", np.nan),
            "val_ndcg5": val_metrics.get("mean_ndcg_at_5", np.nan),
            "val_first_soz_rank": val_metrics.get("mean_rank_first_positive", np.nan),
            "selection_score": selection_score,
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(row)
        if selection_score > best_score + 1e-6:
            best_score = selection_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 20 == 0:
            print(
                f"{stage_name} [{ablation}] | ep {epoch:03d} | loss={row['train_loss']:.4f} "
                f"| MRR={row['val_mrr']:.3f} | Hit1={row['val_top1_hit']:.3f} "
                f"| nDCG5={row['val_ndcg5']:.3f} | score={selection_score:.3f}"
            )
        if wait >= min(PATIENCE, 45):
            print(f"{stage_name}: early stopping at epoch {epoch}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def patient_level_bootstrap_ci(graphs_raw, score_arrays, metric="auc", n_boot=BOOTSTRAP_N, seed=RANDOM_SEED):
    pobj = aggregate_predictions_by_patient(graphs_raw, score_arrays)
    if not pobj:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = [pobj[i] for i in rng.integers(0, len(pobj), size=len(pobj))]
        if metric in {"auc", "auprc"}:
            y = np.concatenate([o["y"] for o in sample])
            p = np.concatenate([o["p"] for o in sample])
            v = safe_auc(y, p) if metric == "auc" else safe_ap(y, p)
        else:
            mm, _ = _metrics_from_patient_prediction_objects(sample)
            lookup = {"mrr": "mean_mrr", "hit1": "mean_top1_hit", "hit5": "mean_top5_hit",
                      "hit10": "mean_top10_hit", "ndcg5": "mean_ndcg_at_5", "first_rank": "mean_rank_first_positive"}
            v = mm.get(lookup[metric], np.nan)
        if np.isfinite(v):
            vals.append(float(v))
    vals = np.asarray(vals)
    return (float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) else (np.nan, np.nan, np.nan)


def plot_bootstrap_metric_ci(graphs_raw, score_arrays, filename="HUP_bootstrap_metric_CI.png"):
    rows = []
    for label, key in [("AUC", "auc"), ("AUPRC", "auprc"), ("MRR", "mrr"), ("Hit@5", "hit5")]:
        mean, lo, hi = patient_level_bootstrap_ci(graphs_raw, score_arrays, metric=key)
        rows.append((label, mean, lo, hi))
    ci_df = pd.DataFrame(rows, columns=["metric", "mean", "ci_low", "ci_high"])
    ci_df.to_csv(os.path.join(OUTPUT_DIR, "HUP_patient_bootstrap_metric_CI.csv"), index=False)
    plt.figure(figsize=(7.5, 4.5))
    x = np.arange(len(ci_df))
    y = ci_df["mean"].values
    yerr = np.vstack([y - ci_df["ci_low"].values, ci_df["ci_high"].values - y])
    plt.bar(x, y)
    plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=5, linewidth=1.4)
    plt.xticks(x, ci_df["metric"].values)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Held-out HUP patient-level 95% bootstrap CI")
    save_fig(filename)



def find_best_threshold(model, val_graphs):
    scores = [predict_graph(model, g) for g in val_graphs]
    return find_best_threshold_for_score_arrays(val_graphs, scores)


def find_best_threshold_for_score_arrays(graphs_raw, score_arrays):
    if graphs_raw and all(g.get("source") == "HUP" for g in graphs_raw):
        pobj = aggregate_predictions_by_patient(graphs_raw, score_arrays)
        all_y = np.concatenate([o["y"] for o in pobj])
        all_p = np.concatenate([o["p"] for o in pobj])
    else:
        all_y = np.concatenate([g["y"].astype(int) for g in graphs_raw])
        all_p = np.concatenate([normalize_vector(p) for p in score_arrays])
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        pred = (all_p >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(all_y, pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr, best_f1


def _tabular_xy(graphs):
    X = np.vstack([g["X"] for g in graphs])
    y = np.concatenate([g["y"].astype(int) for g in graphs])
    return X, y


def _predict_tabular_by_graph(clf, graphs):
    out = []
    for g in graphs:
        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(g["X"])[:, 1]
        else:
            d = clf.decision_function(g["X"])
            p = 1.0 / (1.0 + np.exp(-d))
        out.append(np.asarray(p, dtype=float))
    return out


def run_supervised_baselines(train_graphs, val_graphs, test_graphs_raw, test_graphs_scaled):
    """Leakage-safe non-graph baselines trained only on HUP training contacts."""
    Xtr, ytr = _tabular_xy(train_graphs)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", random_state=RANDOM_SEED, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, random_state=RANDOM_SEED),
        "MLP": MLPClassifier(hidden_layer_sizes=(96, 48), max_iter=400, early_stopping=True, random_state=RANDOM_SEED),
    }
    rows = []
    for name, clf in models.items():
        try:
            clf.fit(Xtr, ytr)
            test_scores = _predict_tabular_by_graph(clf, test_graphs_scaled)
            m = evaluate_score_arrays(test_graphs_raw, test_scores, method_name=name, threshold=0.5)
            rows.append(m)
        except Exception as e:
            print(f"Baseline {name} failed: {e}")
    # Chance ranking baseline for clinical context.
    rng = np.random.default_rng(RANDOM_SEED)
    chance_scores = [rng.random(len(g["y"])) for g in test_graphs_raw]
    rows.append(evaluate_score_arrays(test_graphs_raw, chance_scores, method_name="Random ranking", threshold=0.5))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "HUP_supervised_baselines.csv"), index=False)
    return df


def paired_patient_bootstrap(graphs_raw, scores_a, scores_b, metric="mrr", n_boot=2000, seed=RANDOM_SEED):
    pa = {o["patient_id"]: o for o in aggregate_predictions_by_patient(graphs_raw, scores_a)}
    pb = {o["patient_id"]: o for o in aggregate_predictions_by_patient(graphs_raw, scores_b)}
    ids = sorted(set(pa).intersection(pb))
    if not ids:
        return {"metric": metric, "delta": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    def val(objs):
        m, _ = _metrics_from_patient_prediction_objects(objs)
        key = {"mrr": "mean_mrr", "hit1": "mean_top1_hit", "hit5": "mean_top5_hit",
               "hit10": "mean_top10_hit", "ndcg5": "mean_ndcg_at_5", "first_rank": "mean_rank_first_positive"}[metric]
        return m[key]
    full_a, full_b = [pa[i] for i in ids], [pb[i] for i in ids]
    delta = val(full_a) - val(full_b)
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(ids), size=len(ids))
        aa = [pa[ids[i]] for i in idx]
        bb = [pb[ids[i]] for i in idx]
        ds.append(val(aa) - val(bb))
    ds = np.asarray(ds)
    return {
        "metric": metric, "delta": float(delta),
        "ci_low": float(np.percentile(ds, 2.5)), "ci_high": float(np.percentile(ds, 97.5)),
        "bootstrap_p_two_sided": float(min(1.0, 2.0 * min(np.mean(ds <= 0), np.mean(ds >= 0)))),
    }


def zero_laplacian_pe(graphs):
    out = []
    n_pe = len(LAPLACIAN_PE_COLUMNS)
    for g in graphs:
        g2 = dict(g)
        X = g["X"].copy()
        if n_pe > 0:
            X[:, -n_pe:] = 0.0
        g2["X"] = X
        out.append(g2)
    return out


def run_ablation_study(hup_train_s, hup_val_s, hup_test_s, hup_test_raw,
                       vep_scaled_hup_test, pre_model, finetuned_model):
    """Basic ablation plus optional architecture ablations."""
    print("\n====================================================")
    print("HUP TRUE COMPONENT ABLATION STUDY")
    print("====================================================")
    rows = []
    thr_full, _ = find_best_threshold(finetuned_model, hup_val_s)
    m = evaluate_graphs(finetuned_model, hup_test_s, quiet=True, threshold=thr_full)
    m["method"] = "VEP-pretrained + HUP-finetuned pVEP-EZNet-v3"
    rows.append(m)

    print("Training HUP-only model (no VEP initialization and no VEP-domain MMD)...")
    # A true HUP-only control must not receive VEP information through the MMD pool.
    # Temporarily clear the source-domain graphs while this model is trained.
    global GLOBAL_VEP_DOMAIN_GRAPHS
    _saved_domain_pool = GLOBAL_VEP_DOMAIN_GRAPHS
    GLOBAL_VEP_DOMAIN_GRAPHS = None
    try:
        hup_only_model, hup_only_hist = train_graph_model(
            hup_train_s, hup_val_s, epochs=FINETUNE_EPOCHS, in_dim=len(FEATURE_COLUMNS),
            init_state=None, stage_name="Ablation_HUP_only", ablation="full"
        )
    finally:
        GLOBAL_VEP_DOMAIN_GRAPHS = _saved_domain_pool
    plot_history(hup_only_hist, "Ablation_HUP_only")
    thr_hup, _ = find_best_threshold(hup_only_model, hup_val_s)
    mh = evaluate_graphs(hup_only_model, hup_test_s, quiet=True, threshold=thr_hup)
    mh["method"] = "HUP-only pVEP-EZNet-v3 (no VEP pretraining)"
    rows.append(mh)

    # Zero-shot VEP is retained as a domain-gap audit, not a competitive baseline.
    try:
        mz = evaluate_graphs(pre_model, vep_scaled_hup_test, quiet=True)
        mz["method"] = "VEP-only zero-shot domain-gap audit"
        rows.append(mz)
    except Exception:
        pass

    if RUN_FULL_ARCH_ABLATIONS:
        for ab in ["no_attention", "no_local", "no_feature_gate", "no_residual", "no_rank_loss"]:
            print(f"Training ablation: {ab}")
            model_ab, _ = train_graph_model(
                hup_train_s, hup_val_s, epochs=FINETUNE_EPOCHS, in_dim=len(FEATURE_COLUMNS),
                init_state=None if ab == "no_rank_loss" else None,
                stage_name=f"Ablation_{ab}", ablation=ab
            )
            thr_ab, _ = find_best_threshold(model_ab, hup_val_s)
            ma = evaluate_graphs(model_ab, hup_test_s, quiet=True, threshold=thr_ab)
            ma["method"] = ab
            rows.append(ma)

        # Positional encoding ablation uses the same architecture but zeros PE.
        tr0, va0, te0 = zero_laplacian_pe(hup_train_s), zero_laplacian_pe(hup_val_s), zero_laplacian_pe(hup_test_s)
        model_pe, _ = train_graph_model(tr0, va0, epochs=FINETUNE_EPOCHS, in_dim=len(FEATURE_COLUMNS),
                                        init_state=None, stage_name="Ablation_no_laplacian_pe")
        thr_pe, _ = find_best_threshold(model_pe, va0)
        mpe = evaluate_graphs(model_pe, te0, quiet=True, threshold=thr_pe)
        mpe["method"] = "no_laplacian_pe"
        rows.append(mpe)

    # Handcrafted priors are reported as baselines, not called architecture ablations.
    for feat, name in [
        ("heuristic_composite_score", "heuristic composite only"),
        ("onset_score", "onset prior only"),
        ("energy_score", "energy prior only"),
        ("hfa_score", "HFA prior only"),
    ]:
        rows.append(evaluate_feature_only_baseline(hup_test_raw, feat, name))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "HUP_ablation_study.csv"), index=False)
    print(df.to_string(index=False))
    return df, hup_only_model, thr_hup


def run_repeated_group_cv(hup_graphs, pretrained_state):
    """Optional 5-seed x 5-fold patient-group CV for the final manuscript."""
    if not RUN_REPEATED_GROUP_CV:
        return pd.DataFrame()
    groups = sorted({get_group_id_for_split(g) for g in hup_graphs})
    rows = []
    for seed in CV_SEEDS:
        kf = KFold(n_splits=min(CV_FOLDS, len(groups)), shuffle=True, random_state=seed)
        groups_arr = np.asarray(groups)
        for fold, (trainval_idx, test_idx) in enumerate(kf.split(groups_arr), start=1):
            test_groups = set(groups_arr[test_idx])
            trainval_groups = list(groups_arr[trainval_idx])
            rng = np.random.default_rng(seed + fold)
            rng.shuffle(trainval_groups)
            n_val = max(1, int(round(0.2 * len(trainval_groups))))
            val_groups = set(trainval_groups[:n_val])
            train_groups = set(trainval_groups[n_val:])
            tr = [g for g in hup_graphs if get_group_id_for_split(g) in train_groups]
            va = [g for g in hup_graphs if get_group_id_for_split(g) in val_groups]
            te = [g for g in hup_graphs if get_group_id_for_split(g) in test_groups]
            scaler = fit_scaler(tr)
            trs, vas, tes = apply_scaler(tr, scaler), apply_scaler(va, scaler), apply_scaler(te, scaler)
            set_seed(seed * 100 + fold)
            model, _ = train_graph_model(trs, vas, epochs=FINETUNE_EPOCHS, in_dim=len(FEATURE_COLUMNS),
                                         init_state=pretrained_state, stage_name=f"CV_s{seed}_f{fold}")
            thr, _ = find_best_threshold(model, vas)
            m = evaluate_graphs(model, tes, quiet=True, threshold=thr)
            m.update({"seed": seed, "fold": fold})
            rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "HUP_repeated_group_CV_fold_metrics.csv"), index=False)
    numeric = df.select_dtypes(include=[np.number]).columns
    summary = pd.DataFrame({"metric": numeric,
                            "mean": [df[c].mean() for c in numeric],
                            "std": [df[c].std(ddof=1) for c in numeric]})
    summary.to_csv(os.path.join(OUTPUT_DIR, "HUP_repeated_group_CV_mean_std.csv"), index=False)
    return df



# ============================================================
# CONTROLLED VEP / ONSET CONTRIBUTION SENSITIVITY
# ============================================================

def _feature_columns_for_control(include_onset: bool):
    base = CONTROL_WITH_ONSET_BASE_COLUMNS if include_onset else CONTROL_NO_ONSET_BASE_COLUMNS
    return list(base) + list(LAPLACIAN_PE_COLUMNS)


def _make_feature_view(graphs, feature_columns):
    """Return graph copies whose neural X contains exactly feature_columns.

    X_df is left untouched for audit/plots. This makes onset removal real rather
    than setting its final ensemble weight to zero. The graph topology, labels,
    patients and all non-onset features remain identical.
    """
    out = []
    for g in graphs:
        g2 = dict(g)
        df = g["X_df"].copy()
        for c in feature_columns:
            if c not in df.columns:
                df[c] = 0.0
        g2["X"] = df[feature_columns].replace([np.inf, -np.inf], 0.0).fillna(0.0).values.astype(np.float32)
        g2["sensitivity_feature_columns"] = list(feature_columns)
        out.append(g2)
    return out


def _prepare_control_split(train_raw, val_raw, test_raw, feature_columns):
    tr = _make_feature_view(train_raw, feature_columns)
    va = _make_feature_view(val_raw, feature_columns)
    te = _make_feature_view(test_raw, feature_columns)
    scaler = fit_scaler(tr)  # train-only fit: no validation/test leakage
    return apply_scaler(tr, scaler), apply_scaler(va, scaler), apply_scaler(te, scaler), scaler


def _train_controlled_hup_model(train_s, val_s, in_dim, init_state, stage_name):
    """Train one controlled HUP model with MMD disabled.

    This is deliberate: configurations 1--4 isolate VEP *initialization* from
    VEP-domain alignment. The complete proposed model remains row 5.
    """
    global GLOBAL_VEP_DOMAIN_GRAPHS
    saved_pool = GLOBAL_VEP_DOMAIN_GRAPHS
    GLOBAL_VEP_DOMAIN_GRAPHS = None
    try:
        model, hist = train_graph_model(
            train_s, val_s,
            epochs=SENSITIVITY_FINETUNE_EPOCHS,
            in_dim=in_dim,
            init_state=init_state,
            stage_name=stage_name,
            ablation="full",
        )
    finally:
        GLOBAL_VEP_DOMAIN_GRAPHS = saved_pool
    hist.to_csv(os.path.join(OUTPUT_DIR, f"{stage_name}_history.csv"), index=False)
    return model


def _evaluate_control_model(model, val_s, test_s, test_raw, method_name):
    val_scores = get_model_score_arrays(model, val_s)
    threshold, _ = find_best_threshold_for_score_arrays(val_s, val_scores)
    test_scores = get_model_score_arrays(model, test_s)
    metrics = evaluate_score_arrays(test_raw, test_scores, method_name=method_name, threshold=threshold)
    metrics["threshold"] = float(threshold)
    metrics["n_parameters"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return metrics, val_scores, test_scores


def _sensitivity_bootstrap_delta(graphs_raw, scores_a, scores_b, metric, n_boot=2000, seed=RANDOM_SEED):
    """Paired patient bootstrap for both discrimination and ranking metrics."""
    pa = {o["patient_id"]: o for o in aggregate_predictions_by_patient(graphs_raw, scores_a)}
    pb = {o["patient_id"]: o for o in aggregate_predictions_by_patient(graphs_raw, scores_b)}
    ids = sorted(set(pa).intersection(pb))
    if not ids:
        return {"metric": metric, "delta_A_minus_B": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    def value(objs):
        if metric in {"auc", "auprc"}:
            y = np.concatenate([o["y"] for o in objs])
            p = np.concatenate([o["p"] for o in objs])
            return safe_auc(y, p) if metric == "auc" else safe_ap(y, p)
        m, _ = _metrics_from_patient_prediction_objects(objs)
        key = {
            "mrr": "mean_mrr",
            "first_rank": "mean_rank_first_positive",
            "hit5": "mean_top5_hit",
            "ndcg5": "mean_ndcg_at_5",
        }[metric]
        return m.get(key, np.nan)

    full_a = [pa[i] for i in ids]
    full_b = [pb[i] for i in ids]
    delta = value(full_a) - value(full_b)
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n_boot):
        ii = rng.integers(0, len(ids), size=len(ids))
        aa = [pa[ids[j]] for j in ii]
        bb = [pb[ids[j]] for j in ii]
        d = value(aa) - value(bb)
        if np.isfinite(d):
            ds.append(float(d))
    ds = np.asarray(ds, dtype=float)
    p2 = float(min(1.0, 2.0 * min(np.mean(ds <= 0), np.mean(ds >= 0)))) if len(ds) else np.nan
    return {
        "metric": metric,
        "delta_A_minus_B": float(delta) if np.isfinite(delta) else np.nan,
        "ci_low": float(np.percentile(ds, 2.5)) if len(ds) else np.nan,
        "ci_high": float(np.percentile(ds, 97.5)) if len(ds) else np.nan,
        "bootstrap_p_two_sided": p2,
    }


def _plot_controlled_sensitivity(df, filename="HUP_VEP_onset_controlled_sensitivity.png"):
    cols = ["auc", "average_precision", "mean_mrr", "mean_top5_hit"]
    labels = ["AUC", "AUPRC", "MRR", "Hit@5"]
    mat = df[cols].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(8.8, max(3.8, 0.62 * len(df))))
    im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["configuration"].tolist(), fontsize=8)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title("Controlled VEP/onset contribution sensitivity (same HUP split)")
    save_fig(filename)


def run_controlled_vep_onset_sensitivity(
        vep_train_raw, vep_val_raw,
        hup_train_raw, hup_val_raw, hup_test_raw,
        full_vep_pretrained_state,
        full_model_scores, full_model_threshold,
        full_model_weights=None):
    """Generate the exact five publication-oriented controlled configurations.

    Rows 1--4 use the *same architecture, graph construction, loss, optimizer,
    HUP patient split and validation protocol*. The only factorial changes are:
      - onset_score + onset_z_score absent/present;
      - random HUP initialization vs VEP-pretrained initialization.

    MMD is disabled in rows 1--4 so 'VEP' means VEP PRETRAINING, not an extra
    domain-alignment signal. Row 5 is the complete proposed model used elsewhere
    in the manuscript. No raw onset score is added again at ensemble level.
    """
    if not RUN_CONTROLLED_VEP_ONSET_SENSITIVITY:
        return pd.DataFrame(), {}

    print("\n====================================================")
    print("CONTROLLED VEP / ONSET CONTRIBUTION SENSITIVITY")
    print("====================================================")

    # Record the exact HUP patient assignment proving all five rows share a split.
    split_rows = []
    for split_name, graphs in [("train", hup_train_raw), ("validation", hup_val_raw), ("test", hup_test_raw)]:
        for g in graphs:
            split_rows.append({
                "split": split_name,
                "patient_group": get_group_id_for_split(g),
                "graph_id": g.get("patient_id", ""),
                "n_contacts": len(g.get("y", [])),
                "n_soz_contacts": int(np.sum(g.get("y", []))),
            })
    pd.DataFrame(split_rows).to_csv(
        os.path.join(OUTPUT_DIR, "HUP_VEP_onset_sensitivity_same_split_manifest.csv"), index=False
    )

    no_onset_cols = _feature_columns_for_control(False)
    with_onset_cols = _feature_columns_for_control(True)

    # Train-only scalers for each feature space. The same scaler is used for
    # all methods that share that feature space, making the comparison controlled.
    htr_no, hva_no, hte_no, hsc_no = _prepare_control_split(
        hup_train_raw, hup_val_raw, hup_test_raw, no_onset_cols
    )
    htr_on, hva_on, hte_on, hsc_on = _prepare_control_split(
        hup_train_raw, hup_val_raw, hup_test_raw, with_onset_cols
    )

    # VEP no-onset pretraining model (required for row 3).
    vtr_no = _make_feature_view(vep_train_raw, no_onset_cols)
    vva_no = _make_feature_view(vep_val_raw, no_onset_cols)
    vsc_no = fit_scaler(vtr_no)
    vtr_no_s = apply_scaler(vtr_no, vsc_no)
    vva_no_s = apply_scaler(vva_no, vsc_no)
    print("Training no-onset VEP pretraining control...")
    vep_no_model, vep_no_hist = train_graph_model(
        vtr_no_s, vva_no_s,
        epochs=SENSITIVITY_PRETRAIN_EPOCHS,
        in_dim=len(no_onset_cols),
        init_state=None,
        stage_name="Sensitivity_VEP_pretrain_no_onset",
        ablation="full",
    )
    vep_no_hist.to_csv(os.path.join(OUTPUT_DIR, "Sensitivity_VEP_pretrain_no_onset_history.csv"), index=False)
    vep_no_state = {k: v.detach().cpu().clone() for k, v in vep_no_model.state_dict().items()}

    results = []
    scores = {}

    # 1) HUP only, no onset, no VEP.
    print("[1/5] HUP only (no onset, no VEP)")
    m1_model = _train_controlled_hup_model(
        htr_no, hva_no, len(no_onset_cols), None, "Sensitivity_HUP_no_onset_no_VEP"
    )
    m1, _, s1 = _evaluate_control_model(m1_model, hva_no, hte_no, hup_test_raw, "HUP only (no onset)")
    m1.update({"configuration":"HUP only (no onset)", "onset":"No", "vep_pretraining":"No", "domain_mmd":"No", "fusion":"No"})
    results.append(m1); scores[m1["configuration"]] = s1

    # 2) HUP + onset, no VEP.
    print("[2/5] HUP + onset (no VEP)")
    m2_model = _train_controlled_hup_model(
        htr_on, hva_on, len(with_onset_cols), None, "Sensitivity_HUP_plus_onset"
    )
    m2, v2, s2 = _evaluate_control_model(m2_model, hva_on, hte_on, hup_test_raw, "HUP + onset")
    m2.update({"configuration":"HUP + onset", "onset":"Yes", "vep_pretraining":"No", "domain_mmd":"No", "fusion":"No"})
    results.append(m2); scores[m2["configuration"]] = s2

    # 3) HUP + VEP, onset removed from BOTH VEP/HUP neural feature spaces.
    print("[3/5] HUP + VEP pretraining (no onset)")
    m3_model = _train_controlled_hup_model(
        htr_no, hva_no, len(no_onset_cols), vep_no_state, "Sensitivity_HUP_plus_VEP_no_onset"
    )
    m3, _, s3 = _evaluate_control_model(m3_model, hva_no, hte_no, hup_test_raw, "HUP + VEP")
    m3.update({"configuration":"HUP + VEP", "onset":"No", "vep_pretraining":"Yes", "domain_mmd":"No", "fusion":"No"})
    results.append(m3); scores[m3["configuration"]] = s3

    # 4) HUP + onset + VEP. Reuse the Stage-1 full-feature VEP state but fine-tune
    # without MMD to isolate initialization from the complete proposed model.
    print("[4/5] HUP + onset + VEP pretraining")
    m4_model = _train_controlled_hup_model(
        htr_on, hva_on, len(with_onset_cols), full_vep_pretrained_state,
        "Sensitivity_HUP_plus_onset_plus_VEP"
    )
    m4, v4, s4 = _evaluate_control_model(m4_model, hva_on, hte_on, hup_test_raw, "HUP + onset + VEP")
    m4.update({"configuration":"HUP + onset + VEP", "onset":"Yes", "vep_pretraining":"Yes", "domain_mmd":"No", "fusion":"No"})
    results.append(m4); scores[m4["configuration"]] = s4

    # 5) Full proposed model: use the already-frozen final test scores. Importantly,
    # no raw onset component is appended here, so onset is not double-counted.
    print("[5/5] pVEP-EZNet Ensemble (complete setting)")
    m5 = evaluate_score_arrays(
        hup_test_raw, full_model_scores, method_name="pVEP-EZNet Ensemble (complete setting)",
        threshold=full_model_threshold
    )
    m5.update({
    "configuration":"pVEP-EZNet Ensemble (complete setting)",
    "onset":"Yes",
    "vep_pretraining":"Yes",
    "domain_mmd":"Yes",
    "fusion":"Learned-score fusion"
})
    results.append(m5); scores[m5["configuration"]] = full_model_scores

    df = pd.DataFrame(results)
    keep = [
        "configuration", "onset", "vep_pretraining", "domain_mmd", "fusion",
        "auc", "average_precision", "mean_mrr", "mean_rank_first_positive",
        "mean_top1_hit", "mean_top3_hit", "mean_top5_hit", "mean_top10_hit",
        "mean_ndcg_at_5", "threshold", "n_parameters",
    ]
    df = df[[c for c in keep if c in df.columns]]
    df.to_csv(os.path.join(OUTPUT_DIR, "HUP_VEP_onset_controlled_sensitivity.csv"), index=False)

    # A compact publication-facing table with exactly the requested four headline metrics.
    sensitivity_table = df[[c for c in [
        "configuration", "auc", "average_precision", "mean_mrr", "mean_rank_first_positive"
    ] if c in df.columns]].copy()
    sensitivity_table = sensitivity_table.rename(columns={
        "auc":"AUC", "average_precision":"AUPRC", "mean_mrr":"MRR",
        "mean_rank_first_positive":"First rank"
    })
    sensitivity_table.to_csv(os.path.join(OUTPUT_DIR, "HUP_VEP_contribution_sensitivity_table.csv"), index=False)

    # Factorial contrasts answer the scientific question directly.
    contrasts = [
        ("Onset contribution without VEP", "HUP + onset", "HUP only (no onset)"),
        ("VEP contribution without onset", "HUP + VEP", "HUP only (no onset)"),
        ("VEP contribution with onset", "HUP + onset + VEP", "HUP + onset"),
        ("Onset contribution with VEP", "HUP + onset + VEP", "HUP + VEP"),
        ("Complete ensemble vs HUP + onset", "pVEP-EZNet Ensemble (complete setting)", "HUP + onset"),
    ]
    sig_rows = []
    for contrast, a, b in contrasts:
        for metric in ["auc", "auprc", "mrr", "first_rank", "hit5", "ndcg5"]:
            rr = _sensitivity_bootstrap_delta(hup_test_raw, scores[a], scores[b], metric=metric)
            rr.update({"contrast": contrast, "configuration_A": a, "configuration_B": b})
            # For first rank lower is better; favorable_delta > 0 always means A is better.
            rr["favorable_delta"] = -rr["delta_A_minus_B"] if metric == "first_rank" else rr["delta_A_minus_B"]
            sig_rows.append(rr)
    pd.DataFrame(sig_rows).to_csv(
        os.path.join(OUTPUT_DIR, "HUP_VEP_onset_controlled_paired_bootstrap.csv"), index=False
    )

    if full_model_weights is not None:
        pd.DataFrame([full_model_weights]).to_csv(
            os.path.join(OUTPUT_DIR, "HUP_VEP_onset_full_model_fusion_weights.csv"), index=False
        )

    _plot_controlled_sensitivity(df)
    print("\nControlled VEP/onset sensitivity results:\n", sensitivity_table.to_string(index=False))
    return df, scores


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    set_seed(RANDOM_SEED)
    ensure_dir(OUTPUT_DIR)
    print("\n====================================================")
    print("HYBRID pVEP-EZNet: VEP -> HUP")
    print("====================================================")
    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Primary objective: {PRIMARY_OBJECTIVE}")
    print(f"Feature columns: {FEATURE_COLUMNS}")

    # Stage 1: VEP pretraining
    vep_graphs = load_vep_cohort(VEP_ROOT)
    vep_train, vep_val, vep_test = split_graphs_patient_level(vep_graphs, test_size=0.20, val_size=0.20)
    vep_scaler = fit_scaler(vep_train)
    vep_train_s = apply_scaler(vep_train, vep_scaler)
    vep_val_s = apply_scaler(vep_val, vep_scaler)
    vep_test_s = apply_scaler(vep_test, vep_scaler)
    global GLOBAL_VEP_DOMAIN_GRAPHS
    GLOBAL_VEP_DOMAIN_GRAPHS = vep_train_s
    print(f"\nVEP split: train={len(vep_train_s)}, val={len(vep_val_s)}, test={len(vep_test_s)}")
    pre_model, pre_hist = train_graph_model(vep_train_s, vep_val_s, epochs=PRETRAIN_EPOCHS, in_dim=len(FEATURE_COLUMNS), init_state=None, stage_name="VEP_pretraining")
    plot_history(pre_hist, "VEP_pretraining")
    vep_metrics = evaluate_graphs(pre_model, vep_test_s, quiet=False)
    pd.DataFrame([vep_metrics]).to_csv(os.path.join(OUTPUT_DIR, "VEP_test_metrics.csv"), index=False)
    save_node_predictions(pre_model, vep_test_s, "VEP_test_node_predictions.csv")
    vep_baselines = run_baselines(vep_test, "VEP")
    vep_baselines.to_csv(os.path.join(OUTPUT_DIR, "VEP_baseline_results.csv"), index=False)
    plot_metric_comparison(vep_baselines, vep_metrics, "VEP_baseline_vs_model_auc.png", "VEP validation: Baseline vs pVEP-EZNet")
    pretrained_state = {k: v.detach().cpu().clone() for k, v in pre_model.state_dict().items()}

    # Stage 2: HUP real supervised validation
    hup_graphs = load_hup_dataset(HUP_ROOT)
    hup_train, hup_val, hup_test = split_graphs_patient_level(hup_graphs, test_size=0.20, val_size=0.20)
    hup_scaler = fit_scaler(hup_train)
    hup_train_s = apply_scaler(hup_train, hup_scaler)
    hup_val_s = apply_scaler(hup_val, hup_scaler)
    hup_test_s = apply_scaler(hup_test, hup_scaler)
    supervised_baseline_df = run_supervised_baselines(hup_train_s, hup_val_s, hup_test, hup_test_s)
    print(f"\nHUP split: train={len(hup_train_s)}, val={len(hup_val_s)}, test={len(hup_test_s)}")
    finetuned_model, fine_hist = train_graph_model(hup_train_s, hup_val_s, epochs=FINETUNE_EPOCHS, in_dim=len(FEATURE_COLUMNS), init_state=pretrained_state, stage_name="HUP_finetuning")
    plot_history(fine_hist, "HUP_finetuning")
    hup_best_threshold, hup_val_best_f1 = find_best_threshold(finetuned_model, hup_val_s)
    print(f"Best HUP validation threshold: {hup_best_threshold:.2f} | val_f1={hup_val_best_f1:.4f}")
    hup_metrics = evaluate_graphs(finetuned_model, hup_test_s, quiet=False, threshold=hup_best_threshold)
    pd.DataFrame([hup_metrics]).to_csv(os.path.join(OUTPUT_DIR, "HUP_test_metrics.csv"), index=False)
    save_node_predictions(finetuned_model, hup_test_s, "HUP_test_node_predictions.csv")
    hup_baselines = run_baselines(hup_test, "HUP")
    hup_baselines.to_csv(os.path.join(OUTPUT_DIR, "HUP_baseline_results.csv"), index=False)
    plot_metric_comparison(hup_baselines, hup_metrics, "HUP_baseline_vs_model_auc.png", "HUP validation: Baseline vs pVEP-EZNet")

    hup_test_vep_scaled = apply_scaler(hup_test, vep_scaler)
    ablation_df, hup_only_model, hup_only_threshold = run_ablation_study(
        hup_train_s, hup_val_s, hup_test_s, hup_test, hup_test_vep_scaled, pre_model, finetuned_model
    )

    # Hybrid validation-tuned ensemble
    print("\nTuning hybrid ranking ensemble on HUP validation set...")
    val_components = {
        "hup_gcn": get_model_score_arrays(hup_only_model, hup_val_s),
        "transfer_gcn": get_model_score_arrays(finetuned_model, hup_val_s),
        "onset": get_feature_score_arrays(hup_val, "onset_score"),
        "energy": get_feature_score_arrays(hup_val, "energy_score"),
        "explainable": get_feature_score_arrays(hup_val, "heuristic_composite_score"),
    }
    best_hybrid = tune_hybrid_weights(hup_val, val_components, component_names=["hup_gcn", "transfer_gcn"])
    print("Best hybrid weights:", best_hybrid["weights"])
    print(f"Best hybrid threshold: {best_hybrid['threshold']:.2f}")
    test_components = {
        "hup_gcn": get_model_score_arrays(hup_only_model, hup_test_s),
        "transfer_gcn": get_model_score_arrays(finetuned_model, hup_test_s),
        "onset": get_feature_score_arrays(hup_test, "onset_score"),
        "energy": get_feature_score_arrays(hup_test, "energy_score"),
        "explainable": get_feature_score_arrays(hup_test, "heuristic_composite_score"),
    }
    hybrid_test_scores = combine_score_components(test_components, best_hybrid["weights"])
    hybrid_metrics = evaluate_score_arrays(hup_test, hybrid_test_scores, method_name="HUP + VEP-transfer score ensemble", threshold=best_hybrid["threshold"])
    paired_rows = []
    for metric_name in ["mrr", "hit1", "hit5", "hit10", "ndcg5", "first_rank"]:
        paired_rows.append(paired_patient_bootstrap(
            hup_test, hybrid_test_scores, test_components["hup_gcn"], metric=metric_name
        ))
    pd.DataFrame(paired_rows).to_csv(os.path.join(OUTPUT_DIR, "HUP_paired_patient_bootstrap_hybrid_vs_hup_only.csv"), index=False)
    print("\nHUP + VEP-transfer ensemble held-out HUP test metrics:")
    for k, v in hybrid_metrics.items():
        if k != "method":
            print(f"{k}: {v:.4f}" if isinstance(v, (float, int, np.floating)) else f"{k}: {v}")
    ablation_df = pd.concat([ablation_df, pd.DataFrame([hybrid_metrics])], ignore_index=True)
    ablation_df.to_csv(os.path.join(OUTPUT_DIR, "HUP_hybrid_ablation_study.csv"), index=False)
    plot_ablation_metric_heatmap(ablation_df)
    plot_topk_hit_curves(ablation_df)
    plot_precision_recall_curves({
        "Hybrid": hybrid_test_scores,
        "HUP-only graph model": test_components["hup_gcn"],
        "VEP-transfer graph model": test_components["transfer_gcn"],
        "Onset only": test_components["onset"],
        "Energy only": test_components["energy"],
        "Heuristic composite only": test_components["explainable"],
    }, hup_test)
    generate_publishable_result_pack(hup_test, hybrid_test_scores, threshold=best_hybrid["threshold"])

    # V6: patient/recording-level uncertainty, paired subject tests, and
    # multi-level explainability. These analyses use only the frozen held-out HUP
    # test predictions; no test information is used for model selection.
    stats_bundle = run_statistics_v6(
        hup_test_raw=hup_test,
        hybrid_scores=hybrid_test_scores,
        hup_only_scores=test_components["hup_gcn"],
        onset_scores=test_components["onset"],
    )

    explainability_bundle = run_explainability_v6(
        transfer_model=finetuned_model,
        hup_only_model=hup_only_model,
        hup_train_raw=hup_train,
        hup_val_raw=hup_val,
        hup_test_raw=hup_test,
        hup_test_scaled=hup_test_s,
        hybrid_scores=hybrid_test_scores,
        hup_only_scores=test_components["hup_gcn"],
        transfer_scores=test_components["transfer_gcn"],
        hybrid_weights=best_hybrid["weights"],
    )

    # Exact publication-oriented five-row VEP contribution sensitivity. Rows 1--4
    # isolate onset and VEP pretraining; row 5 is the complete proposed model.
    sensitivity_df, sensitivity_scores = run_controlled_vep_onset_sensitivity(
        vep_train_raw=vep_train,
        vep_val_raw=vep_val,
        hup_train_raw=hup_train,
        hup_val_raw=hup_val,
        hup_test_raw=hup_test,
        full_vep_pretrained_state=pretrained_state,
        full_model_scores=hybrid_test_scores,
        full_model_threshold=best_hybrid["threshold"],
        full_model_weights=best_hybrid["weights"],
    )

    # Publication-oriented same-split benchmarks: actual EI + plain GCN + GAT +
    # HFO/propagation-GCN-inspired baseline + spatiotemporal graph baseline.
    benchmark_df, benchmark_models = run_benchmarks(
        hup_train_raw=hup_train,
        hup_val_raw=hup_val,
        hup_test_raw=hup_test,
        hup_train_scaled=hup_train_s,
        hup_val_scaled=hup_val_s,
        hup_test_scaled=hup_test_s,
        proposed_transfer_model=finetuned_model,
        proposed_hup_model=hup_only_model,
        proposed_hybrid_scores=hybrid_test_scores,
        proposed_hybrid_threshold=best_hybrid["threshold"],
    )
    literature_comparison_df, literature_comparison_md = generate_literature_comparison_table(
        benchmark_df
    )

    # Save HUP-only structure-aware prediction tables.
    hup_node_structure_df, hup_region_structure_df = save_hup_structure_outputs(
        hup_test, hybrid_test_scores
    )

    run_repeated_group_cv(hup_graphs, pretrained_state)

    # Save model and report
    torch.save({
        "model_state_dict": finetuned_model.state_dict(),
        "hup_only_model_state_dict": hup_only_model.state_dict(),
        "feature_columns": FEATURE_COLUMNS,
        "hup_scaler_mean": hup_scaler.mean_,
        "hup_scaler_scale": hup_scaler.scale_,
        "vep_scaler_mean": vep_scaler.mean_,
        "vep_scaler_scale": vep_scaler.scale_,
        "vep_test_metrics": vep_metrics,
        "hup_test_metrics": hup_metrics,
        "hybrid_hup_test_metrics": hybrid_metrics,
        "hybrid_weights": best_hybrid["weights"],
        "hup_best_threshold": hup_best_threshold,
        "hybrid_best_threshold": best_hybrid["threshold"],
        "settings": {
            "PRIMARY_OBJECTIVE": PRIMARY_OBJECTIVE,
            "USE_LAPLACIAN_PE": USE_LAPLACIAN_PE,
            "LAPLACIAN_PE_DIM": LAPLACIAN_PE_DIM,
            "HIDDEN_DIM": HIDDEN_DIM,
            "DROPOUT": DROPOUT,
            "RANKING_LOSS_WEIGHT": RANKING_LOSS_WEIGHT,
        }
    }, os.path.join(OUTPUT_DIR, "hybrid_pvep_eznet_model.pt"))

    with open(os.path.join(OUTPUT_DIR, "summary_report.txt"), "w", encoding="utf-8") as f:
        f.write("Hybrid pVEP-EZNet Summary\n")
        f.write("==========================\n\n")
        f.write("Framework statement\n")
        f.write("-------------------\n")
        f.write("The proposed framework integrates VEP-based mechanistic graph learning and ")
        f.write("HUP-based supervised SOZ validation within a unified explainable SOZ ")
        f.write("candidate-ranking pipeline.\n\n")
        f.write("VEP Test Metrics\n")
        f.write("----------------\n")
        for k, v in vep_metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\nHUP Test Metrics: VEP-pretrained + HUP-finetuned GCN\n")
        f.write("-----------------------------------------------------\n")
        for k, v in hup_metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\nHybrid HUP Test Metrics\n")
        f.write("-----------------------\n")
        for k, v in hybrid_metrics.items():
            if k != "method":
                f.write(f"{k}: {v:.4f}\n" if isinstance(v, (float, int, np.floating)) else f"{k}: {v}\n")
        f.write("\nHybrid weights\n")
        f.write("--------------\n")
        f.write(str(best_hybrid["weights"]) + "\n")
        f.write("\nControlled VEP / onset sensitivity\n")
        f.write("-----------------------------------\n")
        if not sensitivity_df.empty:
            _cols = [c for c in ["configuration", "auc", "average_precision", "mean_mrr", "mean_rank_first_positive"] if c in sensitivity_df.columns]
            f.write(sensitivity_df[_cols].to_string(index=False) + "\n")
            f.write("Rows 1--4 use the same HUP split and architecture; MMD is disabled so the VEP contrast isolates pretraining. ")
            f.write(
                "The complete ensemble setting uses VEP initialization, "
                "domain alignment, and validation-selected score fusion. "
                "Raw onset is never added again at fusion level.\n"
            )
        else:
            f.write("Sensitivity experiment disabled.\n")
        f.write("\nPublication-oriented same-split HUP benchmarks\n")
        f.write("---------------------------------------------\n")
        if not benchmark_df.empty:
            report_cols = [c for c in [
                "method", "auc", "average_precision", "mean_top1_hit", "mean_top5_hit",
                "mean_top10_hit", "mean_mrr", "mean_rank_first_positive", "n_parameters"
            ] if c in benchmark_df.columns]
            f.write(benchmark_df[report_cols].to_string(index=False) + "\n")
        f.write("\nRelated-work / benchmark comparison\n")
        f.write("------------------------------------\n")
        f.write(literature_comparison_md + "\n")
        f.write("\nImportant Interpretation\n")
        f.write("------------------------\n")
        f.write("The validated result of the framework is the held-out HUP supervised SOZ ")
        f.write("candidate-ranking experiment, supported by the VEP pretraining analyses, ")
        f.write("same-split benchmarks, statistical inference, ablations, and explainability audits.\n")

    print("\n====================================================")
    print("OUTPUT SAVED")
    print("====================================================")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Main files:")
    for name in [
        "VEP_test_metrics.csv",
        "HUP_test_metrics.csv",
        "HUP_ablation_study.csv",
        "HUP_hybrid_ablation_study.csv",
        "HUP_pooled_ROC_PR_curves.png",
        "HUP_confusion_calibration.png",
        "HUP_patient_bootstrap_metric_CI.csv",
        "HUP_patient_bootstrap_all_methods_v6.csv",
        "HUP_recording_level_metrics_v6.csv",
        "HUP_recording_level_cluster_bootstrap_v6.csv",
        "HUP_paired_subject_tests_v6.csv",
        "HUP_explainability_frozen_feature_occlusion_v6.csv",
        "HUP_retrained_feature_ablation_v6.csv",
        "HUP_explainability_top_electrodes_v6.csv",
        "HUP_explainability_feature_attributions_v6.csv",
        "HUP_explainability_neighbor_attributions_v6.csv",
        "HUP_explainability_stability_v6.csv",
        "HUP_clinician_facing_explanation_v6.txt",
        "VEP_loader_audit.csv",
        "HUP_topk_hit_curves.png",
        "HUP_test_channel_structure_predictions.csv",
        "HUP_test_region_predictions.csv",
        "HUP_VEP_onset_controlled_sensitivity.csv",
        "HUP_VEP_contribution_sensitivity_table.csv",
        "HUP_VEP_onset_controlled_paired_bootstrap.csv",
        "HUP_VEP_onset_controlled_sensitivity.png",
        "HUP_benchmark_results.csv",
        "HUP_same_split_manifest.csv",
        "HUP_benchmark_heatmap.png",
        "HUP_paired_bootstrap_significance.csv",
        "Literature_comparison_table.csv",
        "Literature_comparison_table.md",
        "Literature_comparison_table.png",
        "summary_report.txt",
        "hybrid_pvep_eznet_model.pt",
    ]:
        print("  " + name)
    print("\nPipeline completed successfully.")



# ============================================================================
# V4 ADDITIONS
# Same patient-disjoint HUP split for all publication-oriented benchmarks.
# ============================================================================

import time
from scipy.signal import butter, sosfiltfilt, hilbert

RUN_BENCHMARKS = True
BENCHMARK_BASELINE_EPOCHS = 220
BENCHMARK_BASELINE_PATIENCE = 35
BENCHMARK_HIDDEN_DIM = 128
BENCHMARK_DROPOUT = 0.20

# Bartolomei EI settings. EI uses a sliding high/low frequency energy ratio
# and penalises delayed involvement relative to the earliest detected channel.
EI_WINDOW_SEC = 1.0
EI_STEP_SEC = 0.5
EI_TAU_WINDOWS = 1.0
EI_INTEGRATION_SEC = 5.0
EI_DETECTION_Z = 3.0
EI_MIN_CONSEC_WINDOWS = 2
EI_LOW_BAND = (3.4, 12.4)
EI_HIGH_BAND = (12.4, 96.0)

# HFO/propagation baseline. Because the proposed pipeline is deliberately
# low-pass filtered at 120 Hz, this benchmark uses the retained 80-120 Hz
# high-frequency activity as a reproducible HFO/HFA proxy. It is therefore
# labelled an "inspired baseline", not an exact reproduction of the 2022 paper.
PROP_HF_BAND = (80.0, 120.0)
PROP_HF_Z = 3.0
PROP_HF_MIN_DUR_SEC = 0.020
PROP_HF_MAX_DELAY_SEC = 0.250
PROP_SEIZURE_MAX_DELAY_SEC = 5.0

# Spatio-temporal graph baseline settings.
TEMPORAL_NUM_WINDOWS = 12
TEMPORAL_FEATURE_NAMES = [
    "log_rms", "log_line_length", "log_variance", "beta_slow_ratio",
    "gamma_ratio", "hfa_ratio"
]


def _safe_bandpower_segment(sig, sfreq, fmin, fmax):
    """Welch band power for one short segment; returns 0 when unavailable."""
    sig = np.asarray(sig, dtype=float)
    if sig.size < 8 or sfreq <= 0:
        return 0.0
    upper = min(float(fmax), float(sfreq) / 2.0 - 1e-3)
    lower = max(0.0, float(fmin))
    if upper <= lower:
        return 0.0
    nperseg = min(sig.size, max(8, int(round(sfreq))))
    freqs, psd = welch(sig, fs=sfreq, nperseg=nperseg)
    m = (freqs >= lower) & (freqs < upper)
    if not np.any(m):
        return 0.0
    return float(np.trapz(psd[m], freqs[m]))


def compute_bartolomei_ei(data, sfreq):
    """Compute a Bartolomei-style Epileptogenicity Index per contact.

    ER[n] = E_high[n] / E_low[n], using 12.4-96 Hz over 3.4-12.4 Hz.
    A contact's rapid-frequency involvement time Nd is detected from the
    baseline-standardised ER trajectory. The final score follows

        EI_i = sum_{n=Nd_i}^{Nd_i+H} ER_i[n] / (Nd_i - N0 + tau)

    and is normalised within the recording to [0,1]. Contacts with no detected
    rapid-frequency transition receive EI=0.
    """
    data = np.asarray(data, dtype=float)
    n_ch, n_samp = data.shape
    win = max(8, int(round(EI_WINDOW_SEC * sfreq)))
    step = max(1, int(round(EI_STEP_SEC * sfreq)))
    if n_samp < win:
        return np.zeros(n_ch, dtype=float)
    starts = np.arange(0, n_samp - win + 1, step, dtype=int)
    er = np.zeros((n_ch, len(starts)), dtype=float)
    for wi, s in enumerate(starts):
        seg = data[:, s:s + win]
        for ch in range(n_ch):
            low = _safe_bandpower_segment(seg[ch], sfreq, *EI_LOW_BAND)
            high = _safe_bandpower_segment(seg[ch], sfreq, *EI_HIGH_BAND)
            er[ch, wi] = high / (low + 1e-12)
    # The cropped HUP window normally begins ~60 s before clinical seizure onset.
    baseline_end_sample = min(n_samp, int(round(BASELINE_DURATION_SEC * sfreq)))
    baseline_mask = starts + win <= baseline_end_sample
    if baseline_mask.sum() < 3:
        baseline_mask[:min(3, len(starts))] = True
    search_start = int(np.where(~baseline_mask)[0][0]) if np.any(~baseline_mask) else 0
    nd = np.full(n_ch, np.nan)
    for ch in range(n_ch):
        base = er[ch, baseline_mask]
        mu = float(np.mean(base))
        sd = float(np.std(base)) + 1e-8
        z = (er[ch] - mu) / sd
        for t in range(search_start, max(search_start, len(starts) - EI_MIN_CONSEC_WINDOWS + 1)):
            if np.all(z[t:t + EI_MIN_CONSEC_WINDOWS] >= EI_DETECTION_Z):
                nd[ch] = t
                break
    valid = np.isfinite(nd)
    if not np.any(valid):
        return np.zeros(n_ch, dtype=float)
    n0 = int(np.nanmin(nd))
    H = max(1, int(round(EI_INTEGRATION_SEC / EI_STEP_SEC)))
    ei = np.zeros(n_ch, dtype=float)
    for ch in np.where(valid)[0]:
        d = int(nd[ch])
        e = min(len(starts), d + H)
        numerator = float(np.sum(er[ch, d:e]))
        denominator = float(d - n0) + EI_TAU_WINDOWS
        ei[ch] = numerator / max(denominator, 1e-8)
    return normalize_vector(ei)


def extract_temporal_node_features(data, sfreq, n_windows=TEMPORAL_NUM_WINDOWS):
    """Fixed-length per-contact temporal feature sequence for the ST-GNN baseline."""
    data = np.asarray(data, dtype=float)
    n_ch, n_samp = data.shape
    edges = np.linspace(0, n_samp, n_windows + 1).astype(int)
    out = np.zeros((n_ch, n_windows, len(TEMPORAL_FEATURE_NAMES)), dtype=np.float32)
    for t in range(n_windows):
        a, b = int(edges[t]), int(edges[t + 1])
        if b - a < 8:
            b = min(n_samp, a + 8)
        seg = data[:, a:b]
        for ch in range(n_ch):
            x = np.nan_to_num(seg[ch])
            rms = np.sqrt(np.mean(x ** 2) + 1e-12)
            ll = np.mean(np.abs(np.diff(x))) if len(x) > 1 else 0.0
            var = np.var(x)
            theta = _safe_bandpower_segment(x, sfreq, 4, 8)
            alpha = _safe_bandpower_segment(x, sfreq, 8, 13)
            beta = _safe_bandpower_segment(x, sfreq, 13, 30)
            gamma = _safe_bandpower_segment(x, sfreq, 30, 80)
            hfa = _safe_bandpower_segment(x, sfreq, HFA_LOW_FREQ, HFA_HIGH_FREQ)
            slow = theta + alpha + 1e-12
            total = theta + alpha + beta + gamma + hfa + 1e-12
            out[ch, t] = [
                np.log1p(rms), np.log1p(ll), np.log1p(var),
                beta / slow, gamma / total, hfa / total,
            ]
    return np.nan_to_num(out).astype(np.float32)


def _detect_high_frequency_events(data, sfreq):
    """Detect 80-120 Hz HFA/HFO-proxy events and return rate and first-event times."""
    data = np.asarray(data, dtype=float)
    n_ch, n_samp = data.shape
    low = PROP_HF_BAND[0]
    high = min(PROP_HF_BAND[1], sfreq / 2.0 - 1.0)
    if high <= low + 1.0 or n_samp < 32:
        return np.zeros(n_ch), np.full(n_ch, np.nan), np.zeros(n_ch)
    sos = butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
    try:
        hf = sosfiltfilt(sos, data, axis=1)
    except Exception:
        return np.zeros(n_ch), np.full(n_ch, np.nan), np.zeros(n_ch)
    env = np.abs(hilbert(hf, axis=1))
    baseline_n = min(n_samp, max(16, int(round(BASELINE_DURATION_SEC * sfreq))))
    min_len = max(1, int(round(PROP_HF_MIN_DUR_SEC * sfreq)))
    rates = np.zeros(n_ch, dtype=float)
    first = np.full(n_ch, np.nan)
    energy = np.mean(hf ** 2, axis=1)
    duration_min = max((n_samp / sfreq) / 60.0, 1e-6)
    for ch in range(n_ch):
        base = env[ch, :baseline_n]
        thr = float(np.mean(base) + PROP_HF_Z * (np.std(base) + 1e-8))
        active = env[ch] > thr
        padded = np.pad(active.astype(np.int8), (1, 1))
        d = np.diff(padded)
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        valid = [(s, e) for s, e in zip(starts, ends) if e - s >= min_len]
        rates[ch] = len(valid) / duration_min
        if valid:
            first[ch] = valid[0][0] / sfreq
    return normalize_vector(rates), first, normalize_vector(np.log1p(energy))


def build_hfo_propagation_representation(data, sfreq, raw_df):
    """Create a propagation-network representation inspired by the 2022 HFO/seizure GCN.

    The published method combines pathological-HFO and seizure-propagation networks. Here we
    implement a same-cohort reproducible adaptation using 80-120 Hz high-frequency events
    (because the main preprocessing is capped at 120 Hz) plus seizure-onset propagation.
    """
    n = data.shape[0]
    hf_rate, hf_first, hf_energy = _detect_high_frequency_events(data, sfreq)
    onset = np.asarray(raw_df["onset_time_sec"].values, dtype=float)
    onset_z = normalize_vector(raw_df["onset_zmax"].values)
    A_hf = np.zeros((n, n), dtype=float)
    A_sz = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.isfinite(hf_first[i]) and np.isfinite(hf_first[j]):
                d = hf_first[j] - hf_first[i]
                if 0.0 < d <= PROP_HF_MAX_DELAY_SEC:
                    A_hf[i, j] = np.exp(-d / max(PROP_HF_MAX_DELAY_SEC, 1e-6))
            if np.isfinite(onset[i]) and np.isfinite(onset[j]):
                d = onset[j] - onset[i]
                if 0.0 < d <= PROP_SEIZURE_MAX_DELAY_SEC:
                    A_sz[i, j] = np.exp(-d / max(PROP_SEIZURE_MAX_DELAY_SEC, 1e-6))
    A_dir = 0.5 * normalize_matrix(A_hf) + 0.5 * normalize_matrix(A_sz)
    A_sym = normalize_matrix(A_dir + A_dir.T)
    if np.max(A_sym) <= 0:
        A_sym = build_functional_connectivity(data)
    out_strength = normalize_vector(A_dir.sum(axis=1))
    in_strength = normalize_vector(A_dir.sum(axis=0))
    onset_score = np.zeros(n)
    valid = np.isfinite(onset)
    if valid.sum() > 1:
        mn, mx = np.min(onset[valid]), np.max(onset[valid])
        onset_score[valid] = 1.0 - (onset[valid] - mn) / (mx - mn + 1e-8)
    elif valid.sum() == 1:
        onset_score[np.where(valid)[0][0]] = 1.0
    X = np.column_stack([
        hf_rate, hf_energy, onset_score, onset_z, out_strength, in_strength
    ]).astype(np.float32)
    return X, A_sym.astype(np.float32)


def load_hup_dataset(root):
    """Journal-V4 HUP loader with EI, propagation, and temporal benchmark representations."""
    print("\n====================================================")
    print("STAGE 2: Loading HUP real iEEG dataset (evaluation-v4)")
    print("====================================================")
    if not os.path.exists(root):
        raise FileNotFoundError(f"HUP path not found: {root}")
    rec_files = find_hup_recordings(root)
    if not rec_files:
        raise RuntimeError("No HUP recording files found. Expected EDF/BDF/VHDR/FIF/SET.")
    print(f"Found HUP recording files: {len(rec_files)}")
    subject_to_files = {}
    for f in rec_files:
        sid = get_subject_id_from_path(f)
        subject_to_files.setdefault(sid, []).append(f)
    subject_ids = sorted(subject_to_files.keys())
    if MAX_HUP_PATIENTS is not None:
        subject_ids = subject_ids[:MAX_HUP_PATIENTS]
    patients = []
    for sid in subject_ids:
        files_all = sorted(subject_to_files[sid])
        if HUP_ICTAL_ONLY:
            files_all = [f for f in files_all if "task-ictal" in os.path.basename(f).lower()]
        for rec_path in files_all[:MAX_FILES_PER_HUP_PATIENT]:
            try:
                tmin, tmax = get_seizure_window_from_events(rec_path)
                data, sfreq, ch_names, selected_idx = load_and_preprocess_recording(
                    rec_path, max_channels=MAX_CHANNELS_PER_RECORDING, tmin=tmin, tmax=tmax
                )
                y = infer_soz_labels_from_channels(ch_names, rec_path)
                if y is None or len(y) != len(ch_names) or np.sum(y) == 0:
                    continue
                W_func = build_functional_connectivity(data)
                raw_df = extract_signal_features(data, sfreq, ch_names)
                coords = align_electrode_coords_to_channels(ch_names, rec_path)
                W, W_spatial = combine_functional_spatial_adjacency(W_func, coords)
                X_df = convert_raw_features_to_node_features(raw_df, W)
                ei_score = compute_bartolomei_ei(data, sfreq)
                temporal_X = extract_temporal_node_features(data, sfreq)
                prop_X, prop_W = build_hfo_propagation_representation(data, sfreq, raw_df)
                pid = f"{sid}_{os.path.splitext(os.path.basename(rec_path))[0]}"
                g = make_patient_graph(pid, W, X_df, y, ch_names, source="HUP")
                g["recording_path"] = rec_path
                g["raw_features"] = raw_df
                g["W_functional"] = W_func
                g["W_spatial"] = W_spatial
                g["bartolomei_ei"] = ei_score.astype(np.float32)
                g["temporal_X"] = temporal_X.astype(np.float32)
                g["prop_X"] = prop_X.astype(np.float32)
                g["prop_W"] = prop_W.astype(np.float32)
                g["X_df"]["bartolomei_ei"] = ei_score
                g = add_structural_annotations_to_graph(g, coords=coords)
                patients.append(g)
                print(
                    f"Loaded HUP: {os.path.basename(rec_path)} | channels={len(ch_names)} "
                    f"| SOZ={int(np.sum(y))} | EImax={float(np.max(ei_score)):.2f}"
                )
            except Exception as e:
                print(f"Skipping HUP recording {rec_path}: {e}")
    if not patients:
        raise RuntimeError("No labelled HUP patient graphs were loaded.")
    print(f"Loaded labelled HUP graphs: {len(patients)}")
    print(f"Unique HUP patients: {len(set(get_group_id_for_split(g) for g in patients))}")
    return patients


def fit_extra_scaler(graphs, key):
    rows = []
    for g in graphs:
        arr = np.asarray(g[key], dtype=float)
        rows.append(arr.reshape(-1, arr.shape[-1]))
    scaler = StandardScaler()
    scaler.fit(np.vstack(rows))
    return scaler


def apply_extra_scaler(graphs, scaler, key, out_key):
    out = []
    for g in graphs:
        g2 = dict(g)
        arr = np.asarray(g[key], dtype=float)
        shp = arr.shape
        flat = scaler.transform(arr.reshape(-1, shp[-1]))
        g2[out_key] = flat.reshape(shp).astype(np.float32)
        out.append(g2)
    return out


class PlainGCNRanker(nn.Module):
    def __init__(self, in_dim, hidden=BENCHMARK_HIDDEN_DIM, dropout=BENCHMARK_DROPOUT):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.g1 = DenseGCNLayer(hidden, hidden)
        self.g2 = DenseGCNLayer(hidden, hidden)
        self.g3 = DenseGCNLayer(hidden, hidden)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.dropout = dropout
    def forward(self, X, A):
        An = graph_normalized_tensor(A)
        h0 = F.gelu(self.inp(X))
        h1 = F.gelu(self.g1(h0, An))
        h1 = F.dropout(h1, self.dropout, self.training)
        h2 = F.gelu(self.g2(h1, An) + h0)
        h2 = F.dropout(h2, self.dropout, self.training)
        h3 = F.gelu(self.g3(h2, An) + h1)
        return self.out(h3).squeeze(-1)


class GATRanker(nn.Module):
    def __init__(self, in_dim, hidden=BENCHMARK_HIDDEN_DIM, dropout=BENCHMARK_DROPOUT):
        super().__init__()
        if hidden % ATTENTION_HEADS != 0:
            hidden = ATTENTION_HEADS * max(1, hidden // ATTENTION_HEADS)
        self.inp = nn.Linear(in_dim, hidden)
        self.att1 = WeightedGraphAttention(hidden, ATTENTION_HEADS, dropout)
        self.att2 = WeightedGraphAttention(hidden, ATTENTION_HEADS, dropout)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, X, A):
        h = F.gelu(self.inp(X))
        h, _ = self.att1(h, A)
        h, _ = self.att2(h, A)
        return self.out(h).squeeze(-1)


class SpatioTemporalGraphRanker(nn.Module):
    def __init__(self, temporal_dim=len(TEMPORAL_FEATURE_NAMES), hidden=BENCHMARK_HIDDEN_DIM,
                 dropout=BENCHMARK_DROPOUT):
        super().__init__()
        if hidden % ATTENTION_HEADS != 0:
            hidden = ATTENTION_HEADS * max(1, hidden // ATTENTION_HEADS)
        self.gru = nn.GRU(temporal_dim, hidden, batch_first=True, bidirectional=False)
        self.att1 = WeightedGraphAttention(hidden, ATTENTION_HEADS, dropout)
        self.att2 = WeightedGraphAttention(hidden, ATTENTION_HEADS, dropout)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, temporal_X, A):
        _, h = self.gru(temporal_X)
        z = h[-1]
        z, _ = self.att1(z, A)
        z, _ = self.att2(z, A)
        return self.out(z).squeeze(-1)


def _benchmark_forward(model, g, mode):
    y = torch.tensor(g["y"], dtype=torch.float32, device=DEVICE)
    if mode == "static":
        X = torch.tensor(g["X"], dtype=torch.float32, device=DEVICE)
        A = torch.tensor(sparsify_adjacency(g["W"]), dtype=torch.float32, device=DEVICE)
        return model(X, A), y
    if mode == "prop":
        X = torch.tensor(g["prop_X_scaled"], dtype=torch.float32, device=DEVICE)
        A = torch.tensor(sparsify_adjacency(g["prop_W"]), dtype=torch.float32, device=DEVICE)
        return model(X, A), y
    if mode == "temporal":
        T = torch.tensor(g["temporal_X_scaled"], dtype=torch.float32, device=DEVICE)
        A = torch.tensor(sparsify_adjacency(g["W"]), dtype=torch.float32, device=DEVICE)
        return model(T, A), y
    raise ValueError(mode)


def _benchmark_predict(model, graphs, mode):
    model.eval()
    out = []
    with torch.no_grad():
        for g in graphs:
            logits, _ = _benchmark_forward(model, g, mode)
            out.append(torch.sigmoid(logits).detach().cpu().numpy())
    return out


def train_benchmark_baseline(model, train_graphs, val_graphs, mode, stage_name):
    model = model.to(DEVICE)
    loss_fn = RankingFirstLossV2(pos_weight=compute_pos_weight(train_graphs))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=8)
    best_score, best_state, wait = -1e9, None, 0
    history = []
    for ep in range(1, BENCHMARK_BASELINE_EPOCHS + 1):
        model.train()
        losses = []
        for idx in np.random.permutation(len(train_graphs)):
            g = train_graphs[idx]
            opt.zero_grad()
            logits, y = _benchmark_forward(model, g, mode)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.item()))
        val_scores = _benchmark_predict(model, val_graphs, mode)
        vm = evaluate_score_arrays(val_graphs, val_scores, stage_name, threshold=0.5)
        score = choose_selection_score(vm)
        scheduler.step(score)
        history.append({
            "epoch": ep, "train_loss": float(np.mean(losses)), "selection_score": score,
            "val_auc": vm.get("auc", np.nan), "val_auprc": vm.get("average_precision", np.nan),
            "val_mrr": vm.get("mean_mrr", np.nan), "val_hit1": vm.get("mean_top1_hit", np.nan),
            "val_hit5": vm.get("mean_top5_hit", np.nan), "val_first_rank": vm.get("mean_rank_first_positive", np.nan),
        })
        if score > best_score + 1e-6:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if ep == 1 or ep % 20 == 0:
            print(f"{stage_name} | ep={ep:03d} loss={np.mean(losses):.4f} MRR={vm.get('mean_mrr', np.nan):.3f} Hit1={vm.get('mean_top1_hit', np.nan):.3f}")
        if wait >= BENCHMARK_BASELINE_PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    hist = pd.DataFrame(history)
    hist.to_csv(os.path.join(OUTPUT_DIR, f"{stage_name}_history.csv"), index=False)
    return model


def _evaluate_trained_benchmark_model(model, val_graphs, test_graphs, mode, method_name):
    val_scores = _benchmark_predict(model, val_graphs, mode)
    thr, _ = find_best_threshold_for_score_arrays(val_graphs, val_scores)
    test_scores = _benchmark_predict(model, test_graphs, mode)
    metrics = evaluate_score_arrays(test_graphs, test_scores, method_name=method_name, threshold=thr)
    metrics["threshold"] = float(thr)
    metrics["n_parameters"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return metrics, test_scores


def run_benchmarks(hup_train_raw, hup_val_raw, hup_test_raw,
                            hup_train_scaled, hup_val_scaled, hup_test_scaled,
                            proposed_transfer_model, proposed_hup_model,
                            proposed_hybrid_scores, proposed_hybrid_threshold):
    """Train/evaluate all publication-oriented methods on the exact same HUP split."""
    if not RUN_BENCHMARKS:
        return pd.DataFrame(), {}
    print("\n====================================================")
    print("SAME-SPLIT HUP BENCHMARKS")
    print("====================================================")
    rows, models = [], {}
    split_rows = []
    for split_name, graphs in [("train", hup_train_raw), ("validation", hup_val_raw), ("test", hup_test_raw)]:
        for g in graphs:
            split_rows.append({
                "split": split_name,
                "patient_group": get_group_id_for_split(g),
                "graph_id": g.get("patient_id", ""),
                "n_contacts": int(len(g.get("y", []))),
                "n_soz_contacts": int(np.sum(g.get("y", []))),
            })
    pd.DataFrame(split_rows).to_csv(os.path.join(OUTPUT_DIR, "HUP_same_split_manifest.csv"), index=False)

    # 1) Actual Bartolomei-style EI: no fitting and no test-set tuning.
    ei_val = [np.asarray(g["bartolomei_ei"], dtype=float) for g in hup_val_raw]
    ei_test = [np.asarray(g["bartolomei_ei"], dtype=float) for g in hup_test_raw]
    ei_thr, _ = find_best_threshold_for_score_arrays(hup_val_raw, ei_val)
    m = evaluate_score_arrays(hup_test_raw, ei_test, "Traditional EI (Bartolomei-style)", threshold=ei_thr)
    m["threshold"] = ei_thr
    m["n_parameters"] = 0
    rows.append(m)

    # 2) Plain GCN baseline using the same node features and graph as the proposed method.
    t0 = time.time()
    gcn = PlainGCNRanker(len(FEATURE_COLUMNS))
    gcn = train_benchmark_baseline(gcn, hup_train_scaled, hup_val_scaled, "static", "Benchmark_PlainGCN")
    m, gcn_scores = _evaluate_trained_benchmark_model(gcn, hup_val_scaled, hup_test_scaled, "static", "Plain GCN baseline")
    m["train_seconds"] = time.time() - t0
    rows.append(m); models["gcn"] = gcn

    # 3) GAT baseline: representative attention-GNN benchmark under the same HUP protocol.
    t0 = time.time()
    gat = GATRanker(len(FEATURE_COLUMNS))
    gat = train_benchmark_baseline(gat, hup_train_scaled, hup_val_scaled, "static", "Benchmark_GAT")
    m, gat_scores = _evaluate_trained_benchmark_model(gat, hup_val_scaled, hup_test_scaled, "static", "GAT baseline")
    m["train_seconds"] = time.time() - t0
    rows.append(m); models["gat"] = gat

    # 4) HFO/propagation GCN-inspired baseline with its own train-only scaler.
    prop_scaler = fit_extra_scaler(hup_train_raw, "prop_X")
    ptrain = apply_extra_scaler(hup_train_raw, prop_scaler, "prop_X", "prop_X_scaled")
    pval = apply_extra_scaler(hup_val_raw, prop_scaler, "prop_X", "prop_X_scaled")
    ptest = apply_extra_scaler(hup_test_raw, prop_scaler, "prop_X", "prop_X_scaled")
    t0 = time.time()
    prop_model = PlainGCNRanker(in_dim=6)
    prop_model = train_benchmark_baseline(prop_model, ptrain, pval, "prop", "Benchmark_HFOPropagationGCN")
    m, prop_scores = _evaluate_trained_benchmark_model(prop_model, pval, ptest, "prop", "HFO/propagation GCN-inspired")
    m["train_seconds"] = time.time() - t0
    rows.append(m); models["hfo_prop_gcn"] = prop_model

    # 5) Spatio-temporal graph baseline: temporal GRU + graph attention.
    temp_scaler = fit_extra_scaler(hup_train_raw, "temporal_X")
    ttrain = apply_extra_scaler(hup_train_raw, temp_scaler, "temporal_X", "temporal_X_scaled")
    tval = apply_extra_scaler(hup_val_raw, temp_scaler, "temporal_X", "temporal_X_scaled")
    ttest = apply_extra_scaler(hup_test_raw, temp_scaler, "temporal_X", "temporal_X_scaled")
    t0 = time.time()
    st_model = SpatioTemporalGraphRanker()
    st_model = train_benchmark_baseline(st_model, ttrain, tval, "temporal", "Benchmark_SpatioTemporalGNN")
    m, st_scores = _evaluate_trained_benchmark_model(st_model, tval, ttest, "temporal", "Spatio-temporal graph baseline")
    m["train_seconds"] = time.time() - t0
    rows.append(m); models["st_gnn"] = st_model

    # 6) Proposed learned variants, evaluated on the identical held-out patients.
    transfer_scores = get_model_score_arrays(proposed_transfer_model, hup_test_scaled)
    # threshold was selected from validation in main for the transfer model.
    transfer_val_scores = get_model_score_arrays(proposed_transfer_model, hup_val_scaled)
    transfer_thr, _ = find_best_threshold_for_score_arrays(hup_val_raw, transfer_val_scores)
    m = evaluate_score_arrays(hup_test_raw, transfer_scores, "Proposed VEP-pretrained graph ranker", transfer_thr)
    m["threshold"] = transfer_thr
    m["n_parameters"] = int(
        sum(
            p.numel()
            for p in proposed_transfer_model.parameters()
            if p.requires_grad
        )
    )
    rows.append(m)

    hup_scores = get_model_score_arrays(proposed_hup_model, hup_test_scaled)
    hup_val_scores = get_model_score_arrays(proposed_hup_model, hup_val_scaled)
    hup_thr, _ = find_best_threshold_for_score_arrays(hup_val_raw, hup_val_scores)
    m = evaluate_score_arrays(hup_test_raw, hup_scores, "Proposed HUP-only graph ranker", hup_thr)
    m["threshold"] = hup_thr
    m["n_parameters"] = int(sum(p.numel() for p in proposed_hup_model.parameters() if p.requires_grad))
    rows.append(m)

    m = evaluate_score_arrays(
        hup_test_raw, proposed_hybrid_scores, "Proposed HUP+VEP ensemble",
        threshold=proposed_hybrid_threshold
    )
    m["threshold"] = proposed_hybrid_threshold
    m["n_parameters"] = int(
        sum(
            p.numel()
            for p in proposed_transfer_model.parameters()
            if p.requires_grad
        )
        +
        sum(
            p.numel()
            for p in proposed_hup_model.parameters()
            if p.requires_grad
        )
    )
    rows.append(m)

    df = pd.DataFrame(rows)
    order_cols = [
        "method", "auc", "average_precision", "mean_top1_hit", "mean_top3_hit",
        "mean_top5_hit", "mean_top10_hit", "mean_mrr", "mean_ndcg_at_5",
        "mean_rank_first_positive", "threshold", "n_parameters", "train_seconds",
    ]
    df = df[[c for c in order_cols if c in df.columns]]
    df.to_csv(os.path.join(OUTPUT_DIR, "HUP_benchmark_results.csv"), index=False)
    # Compact evaluation figure using the same held-out test patients.
    plot_cols = [c for c in ["auc", "average_precision", "mean_top5_hit", "mean_mrr"] if c in df.columns]
    if plot_cols:
        mat = df[plot_cols].astype(float).to_numpy()
        fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.55 * len(df))))
        im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_yticks(np.arange(len(df))); ax.set_yticklabels(df["method"].tolist(), fontsize=8)
        ax.set_xticks(np.arange(len(plot_cols))); ax.set_xticklabels([
            {"auc":"AUC", "average_precision":"AUPRC", "mean_top5_hit":"Hit@5", "mean_mrr":"MRR"}.get(c,c)
            for c in plot_cols
        ])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        ax.set_title("Same-split HUP benchmarks")
        save_fig("HUP_benchmark_heatmap.png")

    # Patient-level paired bootstrap: proposed ensemble versus every same-split comparator.
    comparator_scores = {
        "Traditional EI": ei_test,
        "Plain GCN": gcn_scores,
        "GAT": gat_scores,
        "HFO/propagation GCN-inspired": prop_scores,
        "Spatio-temporal graph": st_scores,
    }
    sig_rows = []
    for cname, cscores in comparator_scores.items():
        for metric_name in ["mrr", "hit1", "hit5", "ndcg5", "first_rank"]:
            rr = paired_patient_bootstrap(
                hup_test_raw, proposed_hybrid_scores, cscores, metric=metric_name
            )
            rr["comparator"] = cname
            sig_rows.append(rr)
    pd.DataFrame(sig_rows).to_csv(
        os.path.join(OUTPUT_DIR, "HUP_paired_bootstrap_significance.csv"), index=False
    )
    print("\nBenchmark results:\n", df.to_string(index=False))
    return df, models


def _metric_cell(df, method):
    row = df[df["method"] == method]
    if row.empty:
        return "not run"
    r = row.iloc[0]
    auc = r.get("auc", np.nan); ap = r.get("average_precision", np.nan)
    h5 = r.get("mean_top5_hit", np.nan); mrr = r.get("mean_mrr", np.nan)
    return f"AUC {auc:.3f}; AUPRC {ap:.3f}; Hit@5 {h5:.3f}; MRR {mrr:.3f}"


def generate_literature_comparison_table(benchmark_df):
    """Create a defensible table separating literature methods from our same-split adaptations."""
    rows = [
        {
            "Method": "Traditional Epileptogenicity Index",
            "Year": "2008", "Data": "Human SEEG", "Graph": "No",
            "Mechanistic prior": "No", "Real iEEG": "Yes", "Explainability": "Yes",
            "Ranking": "Yes", "Same-split HUP benchmark": "Bartolomei-style EI",
            "AUC/AUPRC/Top-k": _metric_cell(benchmark_df, "Traditional EI (Bartolomei-style)"),
        },
        {
            "Method": "Seizure localisation with attention-based GNN",
            "Year": "2022", "Data": "Human iEEG + simulation", "Graph": "Yes",
            "Mechanistic prior": "No", "Real iEEG": "Yes", "Explainability": "Attention",
            "Ranking": "Localization via attention", "Same-split HUP benchmark": "GAT baseline",
            "AUC/AUPRC/Top-k": _metric_cell(benchmark_df, "GAT baseline"),
        },
        {
            "Method": "Epilepsy propagation network + GCN",
            "Year": "2022", "Data": "Human iEEG", "Graph": "Yes",
            "Mechanistic prior": "No", "Real iEEG": "Yes", "Explainability": "Network-based",
            "Ranking": "Confidence/localization", "Same-split HUP benchmark": "HFO/propagation GCN-inspired",
            "AUC/AUPRC/Top-k": _metric_cell(benchmark_df, "HFO/propagation GCN-inspired"),
        },
        {
            "Method": "Structural-connectivity + spatiotemporal EZ localization (ST-CANet)",
            "Year": "2025", "Data": "SEEG + CT/MRI/DTI", "Graph": "Structural connectivity",
            "Mechanistic prior": "No", "Real iEEG": "Yes", "Explainability": "Partial",
            "Ranking": "Electrode classification", "Same-split HUP benchmark": "Spatio-temporal graph baseline",
            "AUC/AUPRC/Top-k": _metric_cell(benchmark_df, "Spatio-temporal graph baseline"),
        },
        {
            "Method": "Plain GCN baseline",
            "Year": "baseline", "Data": "HUP iEEG", "Graph": "Yes",
            "Mechanistic prior": "No", "Real iEEG": "Yes", "Explainability": "No",
            "Ranking": "Yes", "Same-split HUP benchmark": "Exact same split",
            "AUC/AUPRC/Top-k": _metric_cell(benchmark_df, "Plain GCN baseline"),
        },
        {
            "Method": "Proposed pVEP-EZNet",
            "Year": "2026", "Data": "HUP + VEP", "Graph": "Yes",
            "Mechanistic prior": "Yes", "Real iEEG": "Yes", "Explainability": "Component/attention",
            "Ranking": "Yes", "Same-split HUP benchmark": "Proposed HUP+VEP ensemble",
            "AUC/AUPRC/Top-k": _metric_cell(benchmark_df, "Proposed HUP+VEP ensemble"),
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "Literature_comparison_table.csv"), index=False)
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    md = "\n".join(lines)
    with open(os.path.join(OUTPUT_DIR, "Literature_comparison_table.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    # Render a compact manuscript-ready PNG.
    fig, ax = plt.subplots(figsize=(18, 4.8))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1.0, 1.55)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
        if r > 0 and df.iloc[r - 1]["Method"] == "Proposed pVEP-EZNet":
            cell.set_text_props(weight="bold")
    ax.set_title("Published SOZ/EZ graph methods and same-split HUP representative benchmarks", fontsize=11)
    save_fig("Literature_comparison_table.png")
    return df, md



# ============================================================================
# V6: PATIENT/RECORDING STATISTICS + MULTI-LEVEL GNN EXPLAINABILITY
# ============================================================================
from scipy.stats import wilcoxon, spearmanr

RUN_STATS_V6 = True
RUN_EXPLAINABILITY_V6 = True
RUN_RETRAINED_FEATURE_ABLATION_V6 = True

RECORDING_BOOTSTRAP_N_V6 = 2000
PAIRED_BOOTSTRAP_N_V6 = 5000
EXPLAIN_IG_STEPS_V6 = 24
EXPLAIN_STABILITY_IG_STEPS_V6 = 8
EXPLAIN_STABILITY_REPEATS_V6 = 10
EXPLAIN_NOISE_STD_V6 = 0.02
EXPLAIN_TOP_NEIGHBORS_V6 = 5
EXPLAIN_MAX_RECORDINGS_V6 = None  # None = explain every held-out recording


def _recording_metric_row_v6(g, scores, method):
    """Metrics for one seizure recording; no patient aggregation here."""
    y = np.asarray(g["y"], dtype=int)
    p = normalize_vector(np.asarray(scores, dtype=float))
    row = {
        "method": method,
        "patient_group": get_group_id_for_split(g),
        "recording_id": str(g.get("patient_id", "")),
        "n_contacts": int(len(y)),
        "n_soz_contacts": int(np.sum(y)),
        "auc": safe_auc(y, p),
        "auprc": safe_ap(y, p),
        "mrr": reciprocal_rank_first_positive(y, p),
        "first_rank": mean_rank_first_positive(y, p),
        "ndcg5": ndcg_at_k(y, p, 5),
    }
    for k in TOP_K_LIST:
        row[f"top{k}_hit"] = topk_hit(y, p, k)
        row[f"recall_at_{k}"] = recall_at_k(y, p, k)
    return row


def recording_level_metrics_v6(graphs_raw, score_arrays, method):
    rows = [_recording_metric_row_v6(g, s, method) for g, s in zip(graphs_raw, score_arrays)]
    return pd.DataFrame(rows)


def recording_cluster_bootstrap_v6(recording_df, metric, n_boot=RECORDING_BOOTSTRAP_N_V6, seed=RANDOM_SEED):
    """Recording-level metric with subject-cluster bootstrap.

    Entire subjects are resampled, and all of their recordings are retained.
    This reports recording-level Top-k/MRR while respecting within-subject dependence.
    """
    if recording_df.empty or metric not in recording_df.columns:
        return np.nan, np.nan, np.nan
    groups = sorted(recording_df["patient_group"].astype(str).unique())
    if not groups:
        return np.nan, np.nan, np.nan
    by_group = {g: recording_df[recording_df["patient_group"].astype(str) == g] for g in groups}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        chunks = [by_group[g] for g in sampled]
        x = pd.concat(chunks, ignore_index=True)[metric].astype(float).to_numpy()
        x = x[np.isfinite(x)]
        if len(x):
            vals.append(float(np.mean(x)))
    if not vals:
        return np.nan, np.nan, np.nan
    vals = np.asarray(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _patient_metric_vectors_v6(graphs_raw, scores_a, scores_b, metric):
    pa = {o["patient_id"]: o for o in aggregate_predictions_by_patient(graphs_raw, scores_a)}
    pb = {o["patient_id"]: o for o in aggregate_predictions_by_patient(graphs_raw, scores_b)}
    ids = sorted(set(pa).intersection(pb))
    va, vb, used = [], [], []
    for sid in ids:
        def one(o):
            y, p = o["y"], o["p"]
            if metric == "mrr":
                return reciprocal_rank_first_positive(y, p)
            if metric == "first_rank":
                return mean_rank_first_positive(y, p)
            if metric == "hit1":
                return topk_hit(y, p, 1)
            if metric == "hit5":
                return topk_hit(y, p, 5)
            if metric == "hit10":
                return topk_hit(y, p, 10)
            if metric == "ndcg5":
                return ndcg_at_k(y, p, 5)
            raise KeyError(metric)
        a, b = one(pa[sid]), one(pb[sid])
        if np.isfinite(a) and np.isfinite(b):
            va.append(float(a)); vb.append(float(b)); used.append(sid)
    return np.asarray(va), np.asarray(vb), used


def paired_subject_wilcoxon_v6(graphs_raw, scores_a, scores_b, metric, comparison):
    a, b, ids = _patient_metric_vectors_v6(graphs_raw, scores_a, scores_b, metric)
    if len(a) == 0:
        return {"comparison": comparison, "metric": metric, "n_subjects": 0,
                "mean_A": np.nan, "mean_B": np.nan, "mean_delta_A_minus_B": np.nan,
                "median_delta_A_minus_B": np.nan, "favorable_delta": np.nan,
                "wilcoxon_statistic": np.nan, "wilcoxon_p_two_sided": np.nan}
    d = a - b
    if np.allclose(d, 0):
        stat, p = 0.0, 1.0
    else:
        try:
            res = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox", correction=False, mode="auto")
            stat, p = float(res.statistic), float(res.pvalue)
        except Exception:
            stat, p = np.nan, np.nan
    mean_delta = float(np.mean(d))
    favorable = -mean_delta if metric == "first_rank" else mean_delta
    return {
        "comparison": comparison,
        "metric": metric,
        "n_subjects": int(len(a)),
        "mean_A": float(np.mean(a)),
        "mean_B": float(np.mean(b)),
        "mean_delta_A_minus_B": mean_delta,
        "median_delta_A_minus_B": float(np.median(d)),
        "favorable_delta": float(favorable),
        "wilcoxon_statistic": stat,
        "wilcoxon_p_two_sided": p,
    }


def run_statistics_v6(hup_test_raw, hybrid_scores, hup_only_scores, onset_scores):
    """Implement the patient/recording-level statistical analyses."""
    if not RUN_STATS_V6:
        return {}
    print("\n====================================================")
    print("V6 PATIENT / RECORDING STATISTICS")
    print("====================================================")

    methods = {
        "Hybrid pVEP-EZNet": hybrid_scores,
        "HUP-only graph model": hup_only_scores,
        "Onset-only": onset_scores,
    }

    # 1) Patient-level bootstrap CIs: subjects, never contacts, are resampled.
    ci_rows = []
    for method, scores in methods.items():
        for label, key in [
            ("AUC", "auc"), ("AUPRC", "auprc"), ("MRR", "mrr"),
            ("First-SOZ rank", "first_rank"), ("Hit@1", "hit1"),
            ("Hit@5", "hit5"), ("Hit@10", "hit10"), ("nDCG@5", "ndcg5")
        ]:
            mean, lo, hi = patient_level_bootstrap_ci(
                hup_test_raw, scores, metric=key, n_boot=BOOTSTRAP_N, seed=RANDOM_SEED
            )
            ci_rows.append({"method": method, "metric": label, "estimate_bootstrap_mean": mean,
                            "ci_low_95": lo, "ci_high_95": hi, "resampling_unit": "subject"})
    patient_ci_df = pd.DataFrame(ci_rows)
    patient_ci_df.to_csv(os.path.join(OUTPUT_DIR, "HUP_patient_bootstrap_all_methods_v6.csv"), index=False)

    # 2) Per-recording Top-k and ranking metrics, with subject-cluster CIs.
    rec_frames = [recording_level_metrics_v6(hup_test_raw, s, m) for m, s in methods.items()]
    rec_df = pd.concat(rec_frames, ignore_index=True)
    rec_df.to_csv(os.path.join(OUTPUT_DIR, "HUP_recording_level_metrics_v6.csv"), index=False)

    rec_ci_rows = []
    for method in methods:
        d = rec_df[rec_df["method"] == method].copy()
        for label, key in [
            ("Hit@1", "top1_hit"), ("Hit@3", "top3_hit"), ("Hit@5", "top5_hit"),
            ("Hit@10", "top10_hit"), ("MRR", "mrr"),
            ("First-SOZ rank", "first_rank"), ("nDCG@5", "ndcg5")
        ]:
            mean, lo, hi = recording_cluster_bootstrap_v6(d, key)
            observed = float(np.nanmean(d[key].astype(float))) if key in d and len(d) else np.nan
            rec_ci_rows.append({"method": method, "metric": label,
                                "observed_recording_mean": observed,
                                "cluster_bootstrap_mean": mean, "ci_low_95": lo, "ci_high_95": hi,
                                "n_recordings": int(len(d)),
                                "n_subjects": int(d["patient_group"].nunique()),
                                "bootstrap_cluster": "subject"})
    rec_ci_df = pd.DataFrame(rec_ci_rows)
    rec_ci_df.to_csv(os.path.join(OUTPUT_DIR, "HUP_recording_level_cluster_bootstrap_v6.csv"), index=False)

    # 3) Paired subject-level differences and Wilcoxon tests.
    paired_rows = []
    comparisons = [
        ("Hybrid vs HUP-only graph model", hybrid_scores, hup_only_scores),
        ("Hybrid vs onset-only", hybrid_scores, onset_scores),
    ]
    for comp, a, b in comparisons:
        for metric in ["mrr", "first_rank", "hit1", "hit5", "hit10", "ndcg5"]:
            boot = paired_patient_bootstrap(
                hup_test_raw, a, b, metric=metric, n_boot=PAIRED_BOOTSTRAP_N_V6, seed=RANDOM_SEED
            )
            wx = paired_subject_wilcoxon_v6(hup_test_raw, a, b, metric, comp)
            row = dict(wx)
            row.update({
                "bootstrap_delta_A_minus_B": boot.get("delta", np.nan),
                "bootstrap_ci_low_95": boot.get("ci_low", np.nan),
                "bootstrap_ci_high_95": boot.get("ci_high", np.nan),
                "bootstrap_p_two_sided": min(1.0, float(boot.get("bootstrap_p_two_sided", np.nan)))
                    if np.isfinite(boot.get("bootstrap_p_two_sided", np.nan)) else np.nan,
            })
            paired_rows.append(row)
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(os.path.join(OUTPUT_DIR, "HUP_paired_subject_tests_v6.csv"), index=False)

    # Paper-friendly recording Top-k summary.
    summary = rec_df.groupby("method", as_index=False).agg(
        recordings=("recording_id", "count"),
        subjects=("patient_group", "nunique"),
        Hit1=("top1_hit", "mean"), Hit3=("top3_hit", "mean"),
        Hit5=("top5_hit", "mean"), Hit10=("top10_hit", "mean"),
        MRR=("mrr", "mean"), First_rank=("first_rank", "mean"), nDCG5=("ndcg5", "mean")
    )
    summary.to_csv(os.path.join(OUTPUT_DIR, "HUP_recording_level_topk_summary_v6.csv"), index=False)

    print("Patient-level bootstrap and recording-level cluster bootstrap saved.")
    print("Paired subject comparisons saved for Hybrid vs HUP-only and Hybrid vs onset-only.")
    return {"patient_ci": patient_ci_df, "recording": rec_df,
            "recording_ci": rec_ci_df, "paired": paired_df, "recording_summary": summary}


# --------------------------- Explainability ---------------------------

FEATURE_GROUPS_V6 = {
    "onset": ["onset_score", "onset_z_score"],
    "energy": ["energy_score"],
    "HFA": ["hfa_score"],
    "gamma": ["gamma_score"],
    "centrality": ["centrality_score"],
    "Laplacian_PE": list(LAPLACIAN_PE_COLUMNS),
}


def _feature_indices_v6(names, columns=None):
    columns = list(FEATURE_COLUMNS if columns is None else columns)
    return [columns.index(c) for c in names if c in columns]


def integrated_gradients_node_v6(model, graph_scaled, target_idx, steps=EXPLAIN_IG_STEPS_V6):
    """Node-targeted integrated gradients over every source-node feature.

    Zero is a principled baseline because graph_scaled is standardized using the
    training-set scaler, so zero corresponds approximately to the training mean.
    """
    model.eval()
    X, A, _ = graph_to_tensors(graph_scaled)
    X = X.detach()
    baseline = torch.zeros_like(X)
    total_grad = torch.zeros_like(X)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=X.device)[1:]
    for alpha in alphas:
        Xi = (baseline + alpha * (X - baseline)).detach().requires_grad_(True)
        logits = model(Xi, A)
        target = logits[int(target_idx)]
        grad = torch.autograd.grad(target, Xi, retain_graph=False, create_graph=False)[0]
        total_grad += grad.detach()
    ig = (X - baseline) * (total_grad / float(len(alphas)))
    with torch.no_grad():
        logits, _, attn = model(X, A, return_embedding=True)
        prob = torch.sigmoid(logits[int(target_idx)]).item()
    return ig.detach().cpu().numpy(), (attn.detach().cpu().numpy() if attn is not None else None), float(prob), A.detach().cpu().numpy()


def edge_occlusion_for_target_v6(model, graph_scaled, target_idx, candidate_neighbors):
    model.eval()
    X, A, _ = graph_to_tensors(graph_scaled)
    with torch.no_grad():
        base_prob = float(torch.sigmoid(model(X, A)[int(target_idx)]).item())
    rows = []
    for j in candidate_neighbors:
        A2 = A.clone()
        A2[int(target_idx), int(j)] = 0.0
        A2[int(j), int(target_idx)] = 0.0
        with torch.no_grad():
            p2 = float(torch.sigmoid(model(X, A2)[int(target_idx)]).item())
        rows.append((int(j), base_prob, p2, base_prob - p2))
    return rows


def frozen_feature_group_occlusion_v6(model, test_scaled, test_raw, baseline_scores, model_label):
    """Post-hoc group occlusion on the frozen model; zero = train-mean after scaling."""
    base = evaluate_score_arrays(test_raw, baseline_scores, method_name=model_label)
    rows = []
    for group, cols in FEATURE_GROUPS_V6.items():
        idx = _feature_indices_v6(cols)
        altered = []
        for g in test_scaled:
            g2 = dict(g)
            X = np.asarray(g["X"], dtype=np.float32).copy()
            if idx:
                X[:, idx] = 0.0
            g2["X"] = X
            altered.append(g2)
        scores = get_model_score_arrays(model, altered)
        m = evaluate_score_arrays(test_raw, scores, method_name=f"{model_label} minus {group}")
        rows.append({
            "model": model_label, "occluded_group": group,
            "AUC": m.get("auc", np.nan), "AUPRC": m.get("average_precision", np.nan),
            "MRR": m.get("mean_mrr", np.nan), "First_rank": m.get("mean_rank_first_positive", np.nan),
            "Hit5": m.get("mean_top5_hit", np.nan), "nDCG5": m.get("mean_ndcg_at_5", np.nan),
            "delta_AUC": m.get("auc", np.nan) - base.get("auc", np.nan),
            "delta_AUPRC": m.get("average_precision", np.nan) - base.get("average_precision", np.nan),
            "delta_MRR": m.get("mean_mrr", np.nan) - base.get("mean_mrr", np.nan),
            "delta_First_rank": m.get("mean_rank_first_positive", np.nan) - base.get("mean_rank_first_positive", np.nan),
            "delta_Hit5": m.get("mean_top5_hit", np.nan) - base.get("mean_top5_hit", np.nan),
            "analysis_type": "frozen-model feature-group occlusion",
        })
    return pd.DataFrame(rows)


def _keep_columns_without_group_v6(group):
    removed = set(FEATURE_GROUPS_V6[group])
    return [c for c in FEATURE_COLUMNS if c not in removed]


def run_retrained_feature_ablation_v6(hup_train_raw, hup_val_raw, hup_test_raw,
                                       hup_only_model, hup_test_scaled,
                                       transfer_model, transfer_scores):
    """True clinical feature-group ablation by retraining the HUP model.

    Each ablation is trained from scratch on the identical patient split with the
    selected feature group entirely removed from train/validation/test inputs.
    VEP-prior removal is reported separately by comparing the transfer model to
    the full-feature HUP-only model already trained in the main pipeline.
    """
    if not RUN_RETRAINED_FEATURE_ABLATION_V6:
        return pd.DataFrame()
    global GLOBAL_VEP_DOMAIN_GRAPHS
    print("\n====================================================")
    print("RETRAINED CLINICAL FEATURE-GROUP ABLATION")
    print("====================================================")

    full_hup_scores = get_model_score_arrays(hup_only_model, hup_test_scaled)
    full_hup = evaluate_score_arrays(hup_test_raw, full_hup_scores, method_name="HUP-only full features")
    transfer_m = evaluate_score_arrays(hup_test_raw, transfer_scores, method_name="VEP-transfer full features")
    rows = [{
        "ablation": "none (HUP-only reference)", "removed": "none", "reference": "HUP-only full features",
        "AUC": full_hup.get("auc", np.nan), "AUPRC": full_hup.get("average_precision", np.nan),
        "MRR": full_hup.get("mean_mrr", np.nan), "First_rank": full_hup.get("mean_rank_first_positive", np.nan),
        "Hit5": full_hup.get("mean_top5_hit", np.nan), "nDCG5": full_hup.get("mean_ndcg_at_5", np.nan),
        "delta_MRR_vs_reference": 0.0, "delta_First_rank_vs_reference": 0.0,
        "comparison_type": "clinical feature ablation"
    }]

    saved_pool = GLOBAL_VEP_DOMAIN_GRAPHS
    GLOBAL_VEP_DOMAIN_GRAPHS = None  # ensure ablations are pure HUP controls
    try:
        for group in FEATURE_GROUPS_V6:
            keep = _keep_columns_without_group_v6(group)
            tr, va, te, _ = _prepare_control_split(hup_train_raw, hup_val_raw, hup_test_raw, keep)
            stage = "ExplainAblation_no_" + re.sub(r"[^A-Za-z0-9]+", "_", group)
            set_seed(RANDOM_SEED)
            model, hist = train_graph_model(
                tr, va, epochs=FINETUNE_EPOCHS, in_dim=len(keep), init_state=None,
                stage_name=stage, ablation="full"
            )
            hist.to_csv(os.path.join(OUTPUT_DIR, f"{stage}_history.csv"), index=False)
            thr, _ = find_best_threshold(model, va)
            scores = get_model_score_arrays(model, te)
            m = evaluate_score_arrays(hup_test_raw, scores, method_name=stage, threshold=thr)
            rows.append({
                "ablation": f"remove {group}", "removed": ",".join(FEATURE_GROUPS_V6[group]),
                "reference": "HUP-only full features",
                "AUC": m.get("auc", np.nan), "AUPRC": m.get("average_precision", np.nan),
                "MRR": m.get("mean_mrr", np.nan), "First_rank": m.get("mean_rank_first_positive", np.nan),
                "Hit5": m.get("mean_top5_hit", np.nan), "nDCG5": m.get("mean_ndcg_at_5", np.nan),
                "delta_MRR_vs_reference": m.get("mean_mrr", np.nan) - full_hup.get("mean_mrr", np.nan),
                "delta_First_rank_vs_reference": m.get("mean_rank_first_positive", np.nan) - full_hup.get("mean_rank_first_positive", np.nan),
                "comparison_type": "clinical feature ablation"
            })
    finally:
        GLOBAL_VEP_DOMAIN_GRAPHS = saved_pool

    # Mechanistic-prior ablation: full VEP-transfer learned model versus the
    # architecturally matched HUP-only full-feature model.
    rows.append({
        "ablation": "none (VEP-transfer reference)", "removed": "none",
        "reference": "VEP-transfer full features",
        "AUC": transfer_m.get("auc", np.nan), "AUPRC": transfer_m.get("average_precision", np.nan),
        "MRR": transfer_m.get("mean_mrr", np.nan), "First_rank": transfer_m.get("mean_rank_first_positive", np.nan),
        "Hit5": transfer_m.get("mean_top5_hit", np.nan), "nDCG5": transfer_m.get("mean_ndcg_at_5", np.nan),
        "delta_MRR_vs_reference": 0.0, "delta_First_rank_vs_reference": 0.0,
        "comparison_type": "mechanistic-prior ablation"
    })
    rows.append({
        "ablation": "remove VEP prior", "removed": "VEP initialization + VEP-domain MMD",
        "reference": "VEP-transfer full features",
        "AUC": full_hup.get("auc", np.nan), "AUPRC": full_hup.get("average_precision", np.nan),
        "MRR": full_hup.get("mean_mrr", np.nan), "First_rank": full_hup.get("mean_rank_first_positive", np.nan),
        "Hit5": full_hup.get("mean_top5_hit", np.nan), "nDCG5": full_hup.get("mean_ndcg_at_5", np.nan),
        "delta_MRR_vs_reference": full_hup.get("mean_mrr", np.nan) - transfer_m.get("mean_mrr", np.nan),
        "delta_First_rank_vs_reference": full_hup.get("mean_rank_first_positive", np.nan) - transfer_m.get("mean_rank_first_positive", np.nan),
        "comparison_type": "mechanistic-prior ablation"
    })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "HUP_retrained_feature_ablation_v6.csv"), index=False)
    return df


def _topk_set_v6(x, k, exclude=None):
    x = np.asarray(x, dtype=float).copy()
    if exclude is not None and 0 <= int(exclude) < len(x):
        x[int(exclude)] = -np.inf
    k = min(int(k), max(0, len(x) - (1 if exclude is not None else 0)))
    return set(np.argsort(x)[::-1][:k].tolist()) if k > 0 else set()


def explanation_stability_v6(model, graph_scaled, target_idx, base_ig,
                              repeats=EXPLAIN_STABILITY_REPEATS_V6, seed=RANDOM_SEED):
    base_feat = np.sum(np.abs(base_ig), axis=0)
    base_node = np.sum(np.abs(base_ig), axis=1)
    _, A0, _ = graph_to_tensors(graph_scaled)
    A0 = A0.detach().cpu().numpy()
    direct = [j for j in range(len(base_node)) if j != int(target_idx) and A0[int(target_idx), j] > 0]
    base_top = set(sorted(direct, key=lambda j: base_node[j], reverse=True)[:EXPLAIN_TOP_NEIGHBORS_V6])
    feat_rhos, neigh_jacc = [], []
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        g2 = dict(graph_scaled)
        X = np.asarray(graph_scaled["X"], dtype=np.float32).copy()
        X += rng.normal(0.0, EXPLAIN_NOISE_STD_V6, size=X.shape).astype(np.float32)
        g2["X"] = X
        ig2, _, _, _ = integrated_gradients_node_v6(
            model, g2, target_idx, steps=EXPLAIN_STABILITY_IG_STEPS_V6
        )
        f2 = np.sum(np.abs(ig2), axis=0)
        n2 = np.sum(np.abs(ig2), axis=1)
        rho = spearmanr(base_feat, f2).statistic if len(base_feat) > 1 else np.nan
        if np.isfinite(rho):
            feat_rhos.append(float(rho))
        t2 = set(sorted(direct, key=lambda j: n2[j], reverse=True)[:EXPLAIN_TOP_NEIGHBORS_V6])
        union = base_top | t2
        neigh_jacc.append(float(len(base_top & t2) / len(union)) if union else 1.0)
    return {
        "feature_rank_spearman_mean": float(np.mean(feat_rhos)) if feat_rhos else np.nan,
        "feature_rank_spearman_min": float(np.min(feat_rhos)) if feat_rhos else np.nan,
        "top_neighbor_jaccard_mean": float(np.mean(neigh_jacc)) if neigh_jacc else np.nan,
        "perturbation_sigma_scaled": EXPLAIN_NOISE_STD_V6,
        "repeats": int(repeats),
    }


def run_local_explanation_audit_v6(transfer_model, hup_only_model, hup_test_scaled, hup_test_raw,
                                   hybrid_scores, hup_only_scores, transfer_scores, hybrid_weights):
    """Explain final top-ranked electrodes at feature, node, edge and ensemble levels."""
    n_graphs = len(hup_test_raw) if EXPLAIN_MAX_RECORDINGS_V6 is None else min(len(hup_test_raw), EXPLAIN_MAX_RECORDINGS_V6)
    top_rows, feat_rows, neigh_rows, stab_rows = [], [], [], []
    feature_matrix_rows = []
    text_lines = [
        "Hybrid pVEP-EZNet clinician-facing explanation audit",
        "====================================================",
        "This report is a post-hoc decision audit, not a validated clinical usability study.",
        "Integrated gradients explain the learned transfer model; attention is reported as contextual evidence,",
        "and edge occlusion provides a direct counterfactual test of selected graph-neighbor connections.", ""
    ]

    w_h = float(hybrid_weights.get("hup_gcn", 0.0)) if hybrid_weights else 0.0
    w_t = float(hybrid_weights.get("transfer_gcn", 0.0)) if hybrid_weights else 0.0

    for gi in range(n_graphs):
        raw_g, sc_g = hup_test_raw[gi], hup_test_scaled[gi]
        final = normalize_vector(hybrid_scores[gi])
        hups = normalize_vector(hup_only_scores[gi])
        trs = normalize_vector(transfer_scores[gi])
        target = int(np.argmax(final))
        ch_names = list(raw_g.get("channel_names", [f"node_{i}" for i in range(len(final))]))
        target_name = ch_names[target] if target < len(ch_names) else f"node_{target}"
        y = np.asarray(raw_g.get("y", np.zeros(len(final))), dtype=int)

        ig, attn, transfer_prob, A_used = integrated_gradients_node_v6(transfer_model, sc_g, target)
        feature_abs = np.sum(np.abs(ig), axis=0)
        feature_signed = np.sum(ig, axis=0)
        node_influence = np.sum(np.abs(ig), axis=1)
        # Restrict graph-neighbor explanations to direct edges incident on the target.
        # Integrated gradients may also identify multi-hop source nodes, but direct
        # neighbors are the appropriate set for attention and edge-occlusion claims.
        direct_neighbors = [j for j in range(len(node_influence))
                            if j != target and A_used[target, j] > 0]
        top_neigh = sorted(direct_neighbors, key=lambda j: node_influence[j], reverse=True)[:EXPLAIN_TOP_NEIGHBORS_V6]
        edge_rows = edge_occlusion_for_target_v6(transfer_model, sc_g, target, top_neigh)
        edge_map = {j: (bp, op, d) for j, bp, op, d in edge_rows}
        stability = explanation_stability_v6(transfer_model, sc_g, target, ig, seed=RANDOM_SEED + gi)

        # HUP-only model score at the same final-ensemble target.
        Xh, Ah, _ = graph_to_tensors(sc_g)
        with torch.no_grad():
            hup_prob = float(torch.sigmoid(hup_only_model(Xh, Ah)[target]).item())

        top_rows.append({
            "patient_group": get_group_id_for_split(raw_g), "recording_id": raw_g.get("patient_id", ""),
            "top_electrode": target_name, "top_index": target, "true_SOZ": int(y[target]) if target < len(y) else 0,
            "final_hybrid_score": float(final[target]),
            "HUP_component": float(w_h * hups[target]), "VEP_transfer_component": float(w_t * trs[target]),
            "HUP_model_probability": hup_prob, "transfer_model_probability": transfer_prob,
            **stability,
        })

        # Feature attributions.
        orderf = np.argsort(feature_abs)[::-1]
        for rank, fj in enumerate(orderf, start=1):
            fname = FEATURE_COLUMNS[fj] if fj < len(FEATURE_COLUMNS) else f"feature_{fj}"
            feat_rows.append({
                "patient_group": get_group_id_for_split(raw_g), "recording_id": raw_g.get("patient_id", ""),
                "top_electrode": target_name, "feature_rank": rank, "feature": fname,
                "abs_integrated_gradient": float(feature_abs[fj]),
                "signed_integrated_gradient": float(feature_signed[fj]),
            })
        feature_matrix_rows.append([float(v) for v in feature_abs])

        # Neighbor evidence: IG influence + attention + edge counterfactual.
        for rank, j in enumerate(top_neigh, start=1):
            bp, op, drop = edge_map.get(j, (transfer_prob, np.nan, np.nan))
            neigh_rows.append({
                "patient_group": get_group_id_for_split(raw_g), "recording_id": raw_g.get("patient_id", ""),
                "top_electrode": target_name, "neighbor_rank": rank,
                "neighbor": ch_names[j] if j < len(ch_names) else f"node_{j}",
                "integrated_gradient_node_influence": float(node_influence[j]),
                "attention_target_to_neighbor": float(attn[target, j]) if attn is not None else np.nan,
                "graph_edge_weight": float(A_used[target, j]),
                "target_probability_before_edge_occlusion": float(bp),
                "target_probability_after_edge_occlusion": float(op),
                "edge_occlusion_probability_drop": float(drop),
            })

        stab_rows.append({"patient_group": get_group_id_for_split(raw_g),
                          "recording_id": raw_g.get("patient_id", ""),
                          "top_electrode": target_name, **stability})

        top3f = [FEATURE_COLUMNS[j] for j in orderf[:3]]
        top3n = [ch_names[j] if j < len(ch_names) else f"node_{j}" for j in top_neigh[:3]]
        text_lines.extend([
            f"Recording: {raw_g.get('patient_id','')}",
            f"  Top electrode: {target_name} | final score={final[target]:.3f} | true SOZ={int(y[target]) if target < len(y) else 0}",
            f"  Ensemble evidence: HUP component={w_h*hups[target]:.3f}, VEP-transfer component={w_t*trs[target]:.3f}",
            f"  Strongest transfer-model features: {', '.join(top3f)}",
            f"  Most influential graph nodes: {', '.join(top3n) if top3n else 'none'}",
            f"  Explanation stability: feature Spearman={stability['feature_rank_spearman_mean']:.3f}, neighbor Jaccard={stability['top_neighbor_jaccard_mean']:.3f}",
            ""
        ])

    top_df = pd.DataFrame(top_rows)
    feat_df = pd.DataFrame(feat_rows)
    neigh_df = pd.DataFrame(neigh_rows)
    stab_df = pd.DataFrame(stab_rows)
    top_df.to_csv(os.path.join(OUTPUT_DIR, "HUP_explainability_top_electrodes_v6.csv"), index=False)
    feat_df.to_csv(os.path.join(OUTPUT_DIR, "HUP_explainability_feature_attributions_v6.csv"), index=False)
    neigh_df.to_csv(os.path.join(OUTPUT_DIR, "HUP_explainability_neighbor_attributions_v6.csv"), index=False)
    stab_df.to_csv(os.path.join(OUTPUT_DIR, "HUP_explainability_stability_v6.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "HUP_clinician_facing_explanation_v6.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))

    # Global feature-attribution summary across top-ranked electrodes.
    if feature_matrix_rows:
        mat = np.asarray(feature_matrix_rows, dtype=float)
        mean_imp = np.mean(mat, axis=0)
        gdf = pd.DataFrame({"feature": FEATURE_COLUMNS[:len(mean_imp)], "mean_abs_integrated_gradient": mean_imp})
        gdf = gdf.sort_values("mean_abs_integrated_gradient", ascending=False)
        gdf.to_csv(os.path.join(OUTPUT_DIR, "HUP_explainability_global_feature_importance_v6.csv"), index=False)
        plt.figure(figsize=(9, 4.8))
        plt.bar(np.arange(len(gdf)), gdf["mean_abs_integrated_gradient"].values)
        plt.xticks(np.arange(len(gdf)), gdf["feature"].values, rotation=55, ha="right")
        plt.ylabel("Mean absolute integrated gradient")
        plt.title("Global feature attribution across held-out top-ranked electrodes")
        save_fig("HUP_explainability_global_feature_importance_v6.png")

        # Recording x feature heatmap for manuscript/supplement.
        plt.figure(figsize=(11, max(4.5, 0.35 * len(mat))))
        plt.imshow(mat, aspect="auto")
        plt.xticks(np.arange(mat.shape[1]), FEATURE_COLUMNS[:mat.shape[1]], rotation=55, ha="right", fontsize=8)
        ylabels = [r.get("top_electrode", "") for r in top_rows]
        plt.yticks(np.arange(len(ylabels)), ylabels, fontsize=7)
        plt.colorbar(label="Absolute integrated gradient")
        plt.title("Per-recording feature attribution for the final top-ranked electrode")
        save_fig("HUP_explainability_feature_heatmap_v6.png")

    return {"top": top_df, "features": feat_df, "neighbors": neigh_df, "stability": stab_df}


def run_explainability_v6(transfer_model, hup_only_model,
                          hup_train_raw, hup_val_raw, hup_test_raw, hup_test_scaled,
                          hybrid_scores, hup_only_scores, transfer_scores, hybrid_weights):
    if not RUN_EXPLAINABILITY_V6:
        return {}
    print("\n====================================================")
    print("V6 MULTI-LEVEL GRAPH EXPLAINABILITY")
    print("====================================================")

    # A. Frozen-model feature-group occlusion: direct reliance audit.
    frozen_df = frozen_feature_group_occlusion_v6(
        transfer_model, hup_test_scaled, hup_test_raw, transfer_scores,
        model_label="VEP-transfer pVEP-EZNet"
    )
    frozen_df.to_csv(os.path.join(OUTPUT_DIR, "HUP_explainability_frozen_feature_occlusion_v6.csv"), index=False)

    # B. Retrained feature ablation: tests whether performance is reproducibly
    # affected when an entire clinical feature group is unavailable during training.
    retrain_df = run_retrained_feature_ablation_v6(
        hup_train_raw, hup_val_raw, hup_test_raw,
        hup_only_model=hup_only_model, hup_test_scaled=hup_test_scaled,
        transfer_model=transfer_model, transfer_scores=transfer_scores
    )

    # C. Local explanations of the final top-ranked electrodes.
    local = run_local_explanation_audit_v6(
        transfer_model, hup_only_model, hup_test_scaled, hup_test_raw,
        hybrid_scores, hup_only_scores, transfer_scores, hybrid_weights
    )

    # Compact paper-facing explanation summary.
    summary_rows = []
    if not frozen_df.empty:
        for _, r in frozen_df.iterrows():
            summary_rows.append({
                "analysis": "Frozen feature occlusion", "factor": r["occluded_group"],
                "delta_MRR": r.get("delta_MRR", np.nan),
                "delta_First_rank": r.get("delta_First_rank", np.nan),
                "delta_Hit5": r.get("delta_Hit5", np.nan),
            })
    if not local.get("stability", pd.DataFrame()).empty:
        st = local["stability"]
        summary_rows.append({
            "analysis": "Explanation stability", "factor": "held-out top electrodes",
            "delta_MRR": np.nan, "delta_First_rank": np.nan, "delta_Hit5": np.nan,
            "feature_spearman_mean": float(st["feature_rank_spearman_mean"].mean()),
            "neighbor_jaccard_mean": float(st["top_neighbor_jaccard_mean"].mean()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "HUP_explainability_summary_v6.csv"), index=False)

    return {"frozen_occlusion": frozen_df, "retrained_ablation": retrain_df, **local, "summary": summary_df}


if __name__ == "__main__":
    main()
