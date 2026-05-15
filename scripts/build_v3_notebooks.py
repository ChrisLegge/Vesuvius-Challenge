#!/usr/bin/env python3
"""
Vesuvius Challenge Surface Detection - Competition Notebook Builder v3.0
Generates 3 training notebooks (A, B, C) + 1 prediction/ensemble notebook.

v3.0 Metric-Aligned Rewrite:
  Target: Score = 0.30*TopoScore + 0.35*SurfaceDice@tau + 0.35*VOI_score

  A) Architecture: Aux heads (SDF + topology guidance) on decoder
  B) Loss: SurfDistLoss, SDF regression, clDice, CompReg, MultiScale TopoBridge,
     TopoGuidance — phase-aware scheduling (early/mid/late)
  C) Dataset: Fixed hard mining, bridge-risk sampling, SDF/topo GT precompute,
     class-conditional label smoothing, boundary-fraction rejection
  D) Training: 3-phase curriculum (early/mid/late), LR restart at transitions,
     SWA in final 10%, high-res interleaving in mid phase
  E) Validation: Mini-leaderboard with SurfaceDice proxy, VOI proxy, TopoScore proxy
  F) Postproc: 26-CC everywhere, smart dust removal, multi-scale bridge killer,
     cautious hole filling, adaptive component guardrails
  G) Calibration: Score-aligned metric (0.30*topo+0.35*surfdice+0.35*voi),
     ensemble-level calibration on fused logits
  H) Ensemble: A=balanced, B=topology specialist, C=surface specialist,
     risk-aware fusion, trimmed mean option
"""

import json
import os

OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Documents", "Vesuvius Notebooks")

# ============================================================
# Notebook JSON helpers
# ============================================================

def make_nb(cells):
    nb_cells = []
    for src in cells:
        lines = src.split('\n')
        source = [l + '\n' for l in lines[:-1]]
        if lines:
            source.append(lines[-1])
        nb_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"trusted": True},
            "outputs": [],
            "source": source,
        })
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

def save_nb(nb, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    n = len(nb["cells"])
    sz = os.path.getsize(path)
    print(f"  Saved: {os.path.basename(path)} ({n} cells, {sz/1024:.0f} KB)")


# ============================================================
# CELL SOURCES - TRAINING
# ============================================================

CELL_TRAIN_ENV = r'''
# ============================================================
# Environment + Imports
# ============================================================
import subprocess, sys
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "imagecodecs"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except Exception:
    print("[WARN] imagecodecs install failed — tifffile/PIL fallback will be used")

import os, time, gc, warnings, json, random, math, traceback, hashlib, shutil, copy
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt_fn
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

import scipy.ndimage as ndi
from scipy.ndimage import (distance_transform_edt, maximum_filter,
                           generate_binary_structure, label as cc_label,
                           binary_erosion, binary_dilation, binary_opening)

try:
    import tifffile
    def read_tif(path):
        return tifffile.imread(path)
except ImportError:
    from PIL import Image
    def read_tif(path):
        img = Image.open(path)
        frames = []
        try:
            while True:
                frames.append(np.array(img))
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        if len(frames) == 0:
            raise RuntimeError(f"Empty TIF: {path}")
        return np.stack(frames, axis=0)

# NEVER use expandable_segments:True — CUDA VMM corrupts T4/Turing context after ~4 large ops.
# max_split_size_mb:9000 — blocks >9GB never split; medium allocs (activations <3GB) reuse freely.
# Matches predict notebook allocator config (confirmed stable).
import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:9000")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"[ENV] PyTorch {torch.__version__}, Device: {DEVICE}, GPUs: {N_GPUS}")
if DEVICE.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    VRAM_GB = props.total_memory / 1e9
    print(f"[ENV] GPU0: {props.name}, VRAM: {VRAM_GB:.1f} GB")
    if N_GPUS > 1:
        for _gi in range(1, N_GPUS):
            _p = torch.cuda.get_device_properties(_gi)
            print(f"[ENV] GPU{_gi}: {_p.name}, VRAM: {_p.total_memory/1e9:.1f} GB")
else:
    VRAM_GB = 0

T_START = time.time()
def elapsed_h():
    return (time.time() - T_START) / 3600.0
def budget_ok(max_h):
    return elapsed_h() < max_h

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("[ENV] TF32 + cudnn.benchmark enabled")

def log_resources(tag=""):
    try:
        import psutil
        ram = psutil.virtual_memory()
        msg = f"[RES {tag}] RAM={ram.used/1e9:.1f}/{ram.total/1e9:.1f}GB ({ram.percent}%)"
    except ImportError:
        msg = f"[RES {tag}]"
    if DEVICE.type == "cuda":
        alloc = torch.cuda.memory_allocated() / 1e9
        resv = torch.cuda.memory_reserved() / 1e9
        msg += f" GPU={alloc:.1f}/{resv:.1f}GB alloc/resv"
    print(msg, flush=True)

def worker_init_fn(worker_id):
    # Epoch-aware seed: prevents cross-epoch sampling repetition.
    # persistent_workers=False means workers are re-forked each epoch, so
    # wi.dataset._epoch reflects the epoch set by the main process.
    wi = torch.utils.data.get_worker_info()
    epoch = int(wi.dataset._epoch) if (wi is not None and hasattr(wi.dataset, "_epoch")) else 0
    seed_val = hash((SEED, epoch, worker_id)) & 0x7FFFFFFF
    np.random.seed(seed_val)
    random.seed(seed_val)

log_resources("startup")
'''


def cell_train_config(model_name, seed, fold_idx, patch_size,
                      w_ce, w_dice, w_msr, w_bnd, w_topo=0.3,
                      w_surfdist=0.4, w_sdf=0.4, w_cldice=0.0,
                      w_compreg=0.0, w_topoguide=0.0,
                      w_gapneg=0.5, w_tversky=0.0,
                      tversky_alpha=0.3, tversky_beta=0.7, tversky_gamma=1.0,
                      epochs_budget=300, grad_accum=2, ema_decay=0.999,
                      warmup_epochs=5, lr_floor=3e-4, label_smooth=0.02,
                      aug_scale=1.0, ring_width=3, ignore_reject=0.70,
                      n_epoch_vols=8, initial_lr=0.01, grad_clip=5.0,
                      p_boundary=0.40, p_ring=0.20, p_fg=0.25, p_bg=0.10,
                      p_hard=0.05, p_bridge_risk=0.05,
                      p_thin_risk=0.05, p_curvature=0.05, model_role="balanced"):
    pD, pH, pW = patch_size
    noise_p = round(0.10 * aug_scale, 3)
    bright_p = round(0.20 * aug_scale, 3)
    contrast_p = round(0.20 * aug_scale, 3)
    gamma_p = round(0.15 * aug_scale, 3)
    return f'''
# ============================================================
# Configuration - {model_name.upper()} (role: {model_role})
# v3.0: Metric-aligned (Score = 0.30*Topo + 0.35*SurfDice + 0.35*VOI)
# ============================================================
MODEL_NAME   = "{model_name}"
MODEL_ROLE   = "{model_role}"
SEED         = {seed}
FOLD_IDX     = {fold_idx}
NUM_FOLDS    = 5

# Architecture (nnUNet v2 3d_fullres plan + aux heads)
NUM_CLASSES  = 2
IGNORE_LABEL = 255
IN_CHANNELS  = 1
FEATURES     = [32, 64, 128, 256, 320, 320]
BLOCKS       = [1, 3, 4, 6, 6, 6]
STRIDES      = [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]]
DEC_CONVS    = [1, 1, 1, 1, 1]
PATCH_SIZE       = ({pD}, {pH}, {pW})
PATCH_SIZE_EARLY = (min(128, {pD}), min(128, {pH}), min(128, {pW}))

# v3: Three-phase curriculum (time fractions)
# Brain doc: reserve >= 35% for high-res polish (the stage that matters)
PHASE_EARLY_END = 0.15   # Early phase: 0-15% time (was 0.25 — saves 0.85h for mid/late)
PHASE_MID_END   = 0.55   # Mid phase: 15-55% time
                          # Late/polish: 55-100% (45% of budget at full res)

# Training
BATCH_SIZE       = 1
ITERS_PER_EPOCH  = 200   # was 250 — 20% faster epochs = ~3-4 more epochs in 8.5h budget
EPOCHS_BUDGET    = {epochs_budget}
INITIAL_LR       = {initial_lr}
WEIGHT_DECAY     = 3e-5
MOMENTUM         = 0.99
MAX_TRAIN_HOURS  = 8.5
GRAD_CKPT        = True
GRAD_CLIP        = {grad_clip}

# LR schedule
WARMUP_EPOCHS    = {warmup_epochs}
LR_FLOOR         = {lr_floor}
LR_RESTART_FRAC_MID  = 0.40  # v3: restart LR at 40% of initial at mid transition
LR_RESTART_FRAC_LATE = 0.30  # v3: restart LR at 30% of initial at late transition (brain doc: keep alive!)

# Gradient accumulation (effective BS = BATCH_SIZE * GRAD_ACCUM)
GRAD_ACCUM       = {grad_accum}

# EMA
EMA_DECAY        = {ema_decay}

# v3: SWA — start at 80% (was 90%) to accumulate more checkpoints for averaging.
# 90% → 80%: gives ~3× more averaged snapshots in 8.5h; 1.7h of SWA @ 8.5h total.
SWA_START_FRAC   = 0.80
SWA_LR           = 1e-4

# Label smoothing
LABEL_SMOOTH     = {label_smooth}

# v3: Loss weights (6 core + 4 optional)
# Brain doc: CE+Dice+SDF+SurfDist+MSR+Topo = 6 stable, phase-gated
W_CE       = {w_ce}
W_DICE     = {w_dice}
W_SDF      = {w_sdf}        # v3: SDF regression aux head (best math-to-metric trick)
W_SURFDIST = {w_surfdist}   # v3: Surface distance loss (SurfaceDice@tau alignment)
W_MSR      = {w_msr}        # v3: Medial surface recall (anti-split)
W_TOPO     = {w_topo}       # v3: Multi-scale topo bridge (anti-merge)
W_GAPNEG   = {w_gapneg}     # Brain doc: Gap-negative anti-bridge loss (THE weapon)
W_TVERSKY  = {w_tversky}    # Brain doc: Tversky loss (specialist FN/FP control)
TVERSKY_ALPHA = {tversky_alpha}  # FP weight (high = anti-merge)
TVERSKY_BETA  = {tversky_beta}   # FN weight (high = anti-split)
TVERSKY_GAMMA = {tversky_gamma}  # Focal exponent
W_BND      = {w_bnd}        # Optional: boundary-weighted CE
W_CLDICE   = {w_cldice}     # Optional: Centerline Dice (0 by default)
W_COMPREG  = {w_compreg}    # Optional: Component count reg (0 by default)
W_TOPOGUIDE = {w_topoguide} # Optional: Topology guidance aux (0 by default)

# Deep supervision weights (nnUNet style: last=0)
DS_WEIGHTS_RAW = [1.0, 0.5, 0.25, 0.125, 0.0]
_s = sum(DS_WEIGHTS_RAW)
DS_WEIGHTS = [w / _s for w in DS_WEIGHTS_RAW]

# Augmentation (scaled per model for diversity)
AUG_FLIP_PROB     = 0.5
AUG_ROT90_PROB    = 0.3
AUG_NOISE_PROB    = {noise_p}
AUG_NOISE_STD     = 0.07
AUG_BRIGHT_PROB   = {bright_p}
AUG_BRIGHT_RANGE  = 0.08
AUG_CONTRAST_PROB = {contrast_p}
AUG_GAMMA_PROB    = {gamma_p}

# Sampling (v3: added bridge-risk)
P_BOUNDARY    = {p_boundary}
P_RING        = {p_ring}
P_FG          = {p_fg}
P_BG          = {p_bg}
P_HARD        = {p_hard}
P_BRIDGE_RISK = {p_bridge_risk}  # v3: bridge-risk region sampling (anti-merge)
P_THIN_RISK   = {p_thin_risk}   # v3: thin-region sampling (anti-split, brain doc)
P_CURVATURE   = {p_curvature}   # Playbook: high-curvature sheet bends (splits/holes born here)
HARD_CACHE_SIZE = 512
BOUNDARY_DILATE = 3
RING_WIDTH      = {ring_width}
IGNORE_REJECT   = {ignore_reject}
MAX_REJECT_ATTEMPTS = 8
N_EPOCH_VOLS    = {n_epoch_vols}

# Validation (v3: every epoch in late phase per brain doc)
VAL_EVERY       = 5
VAL_EVERY_LATE  = 2     # Brain doc: validate every 2 epochs late-stage (every 1 burns 1.2h)
VAL_N           = 3    # 3 vols — more stable threshold/checkpoint selection; VAL_MAX_CROP=256 keeps each vol ~1-2 min
VAL_OVERLAP     = 0.25

# Paths
ROOT_CANDS = [
    "/kaggle/input/competitions/vesuvius-challenge-surface-detection",
    "/kaggle/input/vesuvius-challenge-surface-detection",
]
ROOT_DIR = next((p for p in ROOT_CANDS if os.path.exists(p)), None)
if ROOT_DIR is None:
    print("[WARN] Competition data not found. Using placeholder.")
    ROOT_DIR = "/kaggle/input/competitions/vesuvius-challenge-surface-detection"
TRAIN_IMG_DIR = os.path.join(ROOT_DIR, "train_images")
TRAIN_LBL_DIR = os.path.join(ROOT_DIR, "train_labels")
CKPT_DIR = "/kaggle/working/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
# Shared coord cache: persists across epochs + workers (persistent_workers=False re-forks
# workers each epoch, killing in-memory LRU cache — disk cache survives re-forks).
# 50s coord compute → 0.5s disk load after first encounter. Huge epoch-over-epoch speedup.
COORD_CACHE_DIR = "/kaggle/working/coord_cache"
os.makedirs(COORD_CACHE_DIR, exist_ok=True)

# Seed everything
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"[CFG] {{MODEL_NAME}} ({{MODEL_ROLE}}): seed={{SEED}}, fold={{FOLD_IDX}}, patch={{PATCH_SIZE}}")
print(f"[CFG] Loss: CE={{W_CE}}, Dice={{W_DICE}}, MSR={{W_MSR}}, BND={{W_BND}}, TOPO={{W_TOPO}}")
print(f"[CFG] v3 Loss: SurfDist={{W_SURFDIST}}, SDF={{W_SDF}}, clDice={{W_CLDICE}}, CompReg={{W_COMPREG}}, TopoGuide={{W_TOPOGUIDE}}")
print(f"[CFG] Training: LR={{INITIAL_LR}}, clip={{GRAD_CLIP}}, warmup={{WARMUP_EPOCHS}}, accum={{GRAD_ACCUM}}, EMA={{EMA_DECAY}}")
print(f"[CFG] Phases: early<{{PHASE_EARLY_END}}, mid<{{PHASE_MID_END}}, late>=, SWA@{{SWA_START_FRAC}}")
print(f"[CFG] Sampling: bnd={{P_BOUNDARY}}, ring={{P_RING}}, fg={{P_FG}}, bg={{P_BG}}, hard={{P_HARD}}, bridge={{P_BRIDGE_RISK}}, thin={{P_THIN_RISK}}, curv={{P_CURVATURE}}")
'''


CELL_DATA_DISCOVERY = r'''
# ============================================================
# Data Discovery + Train/Val Split (GroupKFold by scroll_id)
# ============================================================
import csv as _csv_mod

# Load scroll_id from train.csv — gives honest val/LB correlation
# The test set contains only scroll 26002, so holding that scroll out for
# validation means val metrics directly predict LB generalization.
_scroll_map = {}   # volume_id -> scroll_id
_train_csv_path = os.path.join(ROOT_DIR, "train.csv")
if os.path.exists(_train_csv_path):
    with open(_train_csv_path) as _f:
        for _row in _csv_mod.DictReader(_f):
            _scroll_map[str(_row["id"])] = str(_row["scroll_id"])
    print(f"[DATA] Loaded scroll metadata: {len(_scroll_map)} entries from train.csv")
else:
    print("[WARN] train.csv not found — scroll_id grouping unavailable; using random fold split")

# Discover labeled volumes from train_images/ (excludes deprecated_train_images/)
vol_infos = []
if os.path.isdir(TRAIN_IMG_DIR):
    for f in sorted(os.listdir(TRAIN_IMG_DIR)):
        if f.endswith(".tif"):
            vid = f.replace(".tif", "")
            img_p = os.path.join(TRAIN_IMG_DIR, f)
            lbl_p = os.path.join(TRAIN_LBL_DIR, f)
            if os.path.exists(lbl_p):
                scroll_id = _scroll_map.get(vid, "unknown")
                vol_infos.append({"id": vid, "image": img_p, "label": lbl_p,
                                  "scroll_id": scroll_id})

print(f"[DATA] Found {len(vol_infos)} labeled volumes (train_images/, no deprecated)")

# Print scroll distribution
_scroll_dist = {}
for _v in vol_infos:
    _sid = _v["scroll_id"]
    _scroll_dist[_sid] = _scroll_dist.get(_sid, 0) + 1
print(f"[DATA] Scroll distribution: " + ", ".join(f"scroll_{k}={v}" for k,v in sorted(_scroll_dist.items())))

# GroupKFold by scroll_id:
# Hold out scroll 26002 — this is the ONLY test scroll (test.csv: id=1407735, scroll=26002).
# Val metrics on scroll 26002 training volumes = most honest LB proxy available.
# Train on the remaining 5 scrolls (34117, 35360, 26010, 44430, 53997): 698 volumes.
TEST_SCROLL_ID = "26002"
_has_test_scroll_in_train = any(v["scroll_id"] == TEST_SCROLL_ID for v in vol_infos)

if _has_test_scroll_in_train:
    val_vols   = [v for v in vol_infos if v["scroll_id"] == TEST_SCROLL_ID]
    train_vols = [v for v in vol_infos if v["scroll_id"] != TEST_SCROLL_ID]
    _n_train_scrolls = len({v["scroll_id"] for v in train_vols})
    print(f"[DATA] GroupKFold: val=scroll_{TEST_SCROLL_ID} ({len(val_vols)} vols, same as test scroll), "
          f"train={len(train_vols)} vols across {_n_train_scrolls} scrolls")
else:
    # Fallback: random fold split (test scroll absent from train data)
    _rng = random.Random(42)
    _idx = list(range(len(vol_infos)))
    _rng.shuffle(_idx)
    _fold_sz = max(1, len(_idx) // NUM_FOLDS)
    _val_start = FOLD_IDX * _fold_sz
    _val_end = min(_val_start + _fold_sz, len(_idx))
    _val_set = set(_idx[_val_start:_val_end])
    train_vols = [v for i, v in enumerate(vol_infos) if i not in _val_set]
    val_vols   = [v for i, v in enumerate(vol_infos) if i in _val_set]
    print(f"[DATA] Fallback fold {FOLD_IDX}: {len(train_vols)} train, {len(val_vols)} val "
          f"(scroll {TEST_SCROLL_ID} not in train_images/)")

# Shuffle val_vols once with a fixed seed so VAL_N samples consistently
# draw the same diverse set (not always the first N alphabetically).
# Using SEED ensures reproducibility across restarts.
_val_order_rng = random.Random(SEED)
_val_order_rng.shuffle(val_vols)

_show_val = val_vols[:min(VAL_N, 6)]
for _v in _show_val:
    print(f"  Val: {_v['id']} (scroll {_v['scroll_id']})")
if len(val_vols) > 6:
    print(f"  ... and {len(val_vols) - 6} more val volumes (calibration uses up to 12)")
'''


CELL_ARCHITECTURE = r'''
# ============================================================
# ResidualEncoderUNet v3 — with SDF + Topology Guidance aux heads
# Features: [32, 64, 128, 256, 320, 320], Blocks: [1,3,4,6,6,6]
# InstanceNorm3d + LeakyReLU, ~96M params (+<0.1M for aux heads)
# ============================================================

class ConvBlock3D(nn.Module):
    """Conv3d -> InstanceNorm3d -> LeakyReLU"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        if isinstance(stride, int):
            stride = [stride] * 3
        padding = [k // 2 for k in kernel_size]
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=padding, bias=bias)
        self.norm = nn.InstanceNorm3d(out_ch, eps=1e-5, affine=True)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResBlock3D(nn.Module):
    """Residual: conv->norm->act->conv->norm + skip -> act"""
    def __init__(self, ch, kernel_size=3, bias=True):
        super().__init__()
        p = kernel_size // 2
        self.conv1 = nn.Conv3d(ch, ch, kernel_size, padding=p, bias=bias)
        self.norm1 = nn.InstanceNorm3d(ch, eps=1e-5, affine=True)
        self.conv2 = nn.Conv3d(ch, ch, kernel_size, padding=p, bias=bias)
        self.norm2 = nn.InstanceNorm3d(ch, eps=1e-5, affine=True)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        r = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + r)


class ResidualEncoderUNet(nn.Module):
    """
    nnUNet v2 ResidualEncoderUNet for 3D segmentation.
    v3: Added aux_heads for SDF prediction and topology guidance.
    - sdf_head: Conv3d(features[0], 1) -> clipped SDF in [-5, +5]
    - topo_head: Conv3d(features[0], 1) -> distance-to-medial in [0, 1]
    Both operate on the finest decoder feature map.
    """
    def __init__(self, in_channels=1, num_classes=2,
                 features=(32, 64, 128, 256, 320, 320),
                 blocks=(1, 3, 4, 6, 6, 6),
                 strides=((1,1,1),(2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2)),
                 dec_convs=(1, 1, 1, 1, 1),
                 deep_supervision=True, use_grad_ckpt=True,
                 aux_heads=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self._ckpt = use_grad_ckpt
        self.aux_heads = aux_heads
        n = len(features)

        # Encoder
        self.enc = nn.ModuleList()
        for s in range(n):
            in_c = in_channels if s == 0 else features[s - 1]
            out_c = features[s]
            st = list(strides[s]) if isinstance(strides[s], (list, tuple)) else [strides[s]] * 3
            layers = [ConvBlock3D(in_c, out_c, 3, stride=st)]
            for _ in range(blocks[s] - 1):
                layers.append(ResBlock3D(out_c, 3))
            self.enc.append(nn.Sequential(*layers))

        # Decoder
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.seg = nn.ModuleList()
        for i in range(n - 1):
            s = n - 1 - i
            enc_ch = features[s]
            skip_ch = features[s - 1]
            out_ch = features[s - 1]
            st = list(strides[s]) if isinstance(strides[s], (list, tuple)) else [strides[s]] * 3
            self.up.append(
                nn.ConvTranspose3d(enc_ch, enc_ch, kernel_size=st, stride=st, bias=True)
            )
            n_convs = dec_convs[i] if i < len(dec_convs) else 1
            dec_layers = []
            for c in range(n_convs):
                ic = (enc_ch + skip_ch) if c == 0 else out_ch
                dec_layers.append(ConvBlock3D(ic, out_ch, 3))
            self.dec.append(nn.Sequential(*dec_layers))
            self.seg.append(nn.Conv3d(out_ch, num_classes, 1))

        # v3: Auxiliary heads (from finest decoder features)
        if aux_heads:
            self.sdf_head = nn.Conv3d(features[0], 1, 3, padding=1)
            self.topo_head = nn.Conv3d(features[0], 1, 3, padding=1)

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.enc):
            if self._ckpt and self.training and i > 0:
                x = grad_ckpt_fn(enc, x, use_reentrant=False)
            else:
                x = enc(x)
            skips.append(x)

        outputs = []
        x = skips[-1]
        finest_feat = None
        for i, (up_m, dec_m, seg_m) in enumerate(zip(self.up, self.dec, self.seg)):
            skip = skips[-(i + 2)]
            x = up_m(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            if self._ckpt and self.training:
                x = grad_ckpt_fn(dec_m, x, use_reentrant=False)
            else:
                x = dec_m(x)
            outputs.append(seg_m(x))
            # Save finest resolution features for aux heads
            if i == len(self.up) - 1:
                finest_feat = x

        outputs = outputs[::-1]  # [finest, ..., coarsest]

        if self.deep_supervision and self.training:
            if self.aux_heads and finest_feat is not None:
                sdf_pred = self.sdf_head(finest_feat).squeeze(1)   # (B, D, H, W)
                topo_pred = torch.sigmoid(self.topo_head(finest_feat).squeeze(1))  # (B, D, H, W), [0,1]
                return outputs, sdf_pred, topo_pred
            return outputs, None, None
        # Inference: only return seg output (backward compat)
        return outputs[0]


# Build model
model = ResidualEncoderUNet(
    in_channels=IN_CHANNELS, num_classes=NUM_CLASSES,
    features=FEATURES, blocks=BLOCKS, strides=STRIDES,
    dec_convs=DEC_CONVS, deep_supervision=True, use_grad_ckpt=GRAD_CKPT,
    aux_heads=True
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters()) / 1e6
n_aux = sum(p.numel() for n, p in model.named_parameters() if 'sdf_head' in n or 'topo_head' in n) / 1e6
print(f"[MODEL] ResidualEncoderUNet v3: {n_params:.1f}M params (aux: {n_aux:.3f}M)")

# ── torch.compile: DISABLED for T4 ───────────────────────────────────────────
# T4 (Turing, 40 SMs) triggers "Not enough SMs for max_autotune_gemm" warning —
# compile falls back to a suboptimal kernel, so the 10-15% gain claim doesn't hold.
# Worse: JIT compilation of the first batch allocates large temporary tensors that
# OOM the 15.6GB VRAM (confirmed: 3× OOM-DEGRADE on first batch, patches shrank to 80³).
# cudnn.benchmark=True (set above) provides kernel selection without the OOM risk.
print("[COMPILE] torch.compile disabled — T4 SMs insufficient; eager + cudnn.benchmark active")

# Model hash for reproducibility
_h = hashlib.sha1()
for k, v in sorted(model.state_dict().items()):
    _h.update(k.encode())
    _h.update(str(v.shape).encode())
MODEL_HASH = _h.hexdigest()[:12]
print(f"[MODEL] Architecture hash: {MODEL_HASH}")

# ── GPU Setup: Single-GPU only (DataParallel permanently disabled) ─────────────
# ROOT CAUSE AUDIT (confirmed across 2 crashes):
#   model.forward() returns (List[Tensor], Tensor, Tensor) during training.
#   List[Tensor] has elements at DIFFERENT spatial resolutions (deep supervision).
#   nn.DataParallel.gather() tries to torch.cat() outputs from both GPUs along
#   the batch dimension — but a Python list of varying-shape tensors cannot be
#   cat'd. Result: illegal memory access in F.conv3d on the gather kernel.
#
#   Crash site (both Version 1 @ 250s and Version 2 @ 312s):
#     File "ipykernel.py", line 123, in forward → outputs.append(seg_m(x))
#     torch.AcceleratorError: CUDA error: an illegal memory access was encountered
#
#   Disabling grad_ckpt (Version 2 fix) did NOT resolve it — DP gather is the cause.
#   Only safe fix: never wrap this model in DataParallel. Single T4 = 4.14 steps/s,
#   reliable. 8.5h budget → ~126,000 gradient steps → well-trained model.
_N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
_model_for_fwd = model          # ALWAYS single-GPU: DP incompatible with deep supervision
DATALOADER_BS = BATCH_SIZE
if _N_GPUS > 1:
    print(f"[GPU] {_N_GPUS}× T4 detected — using GPU:0 only (DataParallel disabled: "
          f"deep-supervision list output incompatible with DP gather)")
else:
    print(f"[GPU] Single-GPU mode | BS={DATALOADER_BS} | grad_ckpt={model._ckpt}")
'''


CELL_LOSS = r'''
# ============================================================
# Loss Functions v3: 10-component metric-aligned loss system
# Original: CE + BoundaryWeightedCE + Dice + MSR + TopoBridge
# New: SurfaceDistance + SDF + clDice + CompReg + TopoGuidance
# Phase-aware scheduling via global weights set by training loop
# ============================================================

# Phase-modulated weight globals (set by training loop's set_epoch)
_LOSS_PHASE = "early"  # "early", "mid", "late"
_TOPO_RAMP = 0.0       # 0->1 ramp over first 15% of training
_DS_SCALE = 1.0         # Deep supervision weight multiplier

# Phase weight multipliers: (early, mid, late) for each loss
# Brain doc: 6 stable losses with clear phase gates, not loss soup
# Early = learn regions; Mid = learn connectivity; Late = surface+topo polish
_PHASE_MULTS = {
    "ce":        (1.0, 1.0, 0.8),    # Core: always on
    "dice":      (1.0, 1.0, 0.8),    # Core: always on
    "sdf":       (0.5, 1.0, 0.8),    # Aux head: mild early, full mid, fade late
    # FIX: pre-warm surfdist/msr at 0.15 in early so mid-phase spike is 6x not 60x.
    # Observed: mid-phase loss jumped 1.07→2.55 when both hit at full weight simultaneously.
    "surfdist":  (0.15, 1.0, 1.5),   # SurfaceDice alignment: gentle early warm-up, emphatic late
    "msr":       (0.15, 1.0, 1.3),   # Anti-split: gentle early warm-up, balanced late
    "topo":      (0.0,  1.0, 1.3),   # Anti-merge (bridge): off early, strong late
    # Brain doc additions:
    "gapneg":    (0.0,  1.0, 1.3),   # Gap-negative: off early, strong mid+late (anti-merge weapon)
    "tversky":   (0.0,  1.0, 1.0),   # Tversky: off early, active mid+late (specialist control)
    # clDice: Shit et al. NeurIPS 2021 — soft skeleton overlap, directly targets TopoScore.
    # Ramp in mid (0.8x) to stabilise before full late push (1.3x matches topo).
    "cldice":    (0.0,  0.8, 1.3),
    # CompReg: active mid but emphasised in late where component count errors are scored.
    "compreg":   (0.0,  0.5, 1.2),
    "bnd":       (0.3,  0.3, 0.2),
    "topoguide": (0.5,  1.0, 0.5),
}

def _phase_mult(loss_name):
    """Get phase-dependent multiplier for a loss component."""
    mults = _PHASE_MULTS.get(loss_name, (1.0, 1.0, 1.0))
    if _LOSS_PHASE == "early":
        return mults[0]
    elif _LOSS_PHASE == "mid":
        return mults[1]
    return mults[2]


def compute_boundary_weight_map(target, ignore_label=IGNORE_LABEL, dilate=2, weight=3.0):
    """Per-voxel weight map that upweights boundary regions."""
    B = target.shape[0]
    wmap = torch.ones_like(target, dtype=torch.float32)
    for b in range(B):
        tgt_np = target[b].cpu().numpy()
        fg = (tgt_np == 1).astype(np.uint8)
        if fg.sum() == 0:
            continue
        dilated = ndi.binary_dilation(fg, iterations=dilate).astype(np.uint8)
        eroded = ndi.binary_erosion(fg, iterations=dilate).astype(np.uint8)
        boundary = ((dilated - eroded) > 0).astype(np.float32)
        wmap[b] += torch.from_numpy(boundary * (weight - 1.0)).to(target.device)
    wmap[target == ignore_label] = 0.0
    return wmap


def soft_dice_loss(logits, target, ignore_label=IGNORE_LABEL, eps=1e-5):
    """Per-sample soft Dice loss, ignoring IGNORE_LABEL voxels."""
    C = logits.shape[1]
    logits = logits.float().clamp(-50, 50)
    probs = torch.softmax(logits, dim=1)
    valid = (target != ignore_label)
    tgt = target.clone()
    tgt[~valid] = 0
    onehot = F.one_hot(tgt.long(), num_classes=C).permute(0, 4, 1, 2, 3).float()
    mask = valid.unsqueeze(1).float()
    probs = probs * mask
    onehot = onehot * mask
    dims = (0, 2, 3, 4)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def boundary_weighted_ce(logits, target, boundary_wmap, ignore_label=IGNORE_LABEL):
    """CE loss with per-voxel boundary weighting."""
    with torch.amp.autocast("cuda", enabled=False):
        logits = logits.float().clamp(-20, 20)
        ce_per_voxel = F.cross_entropy(logits, target, ignore_index=ignore_label, reduction='none')
        weighted = ce_per_voxel * boundary_wmap.float()
        valid = (target != ignore_label).float()
        n_valid = valid.sum() + 1e-8
    return weighted.sum() / n_valid


def medial_surface_recall_loss(fg_probs, medial_mask, valid_mask):
    """Penalize missing medial surface voxels in prediction."""
    with torch.amp.autocast("cuda", enabled=False):
        fg_probs = fg_probs.float().clamp(1e-4, 1 - 1e-4)
        med = medial_mask.float() * valid_mask.float()
        n_med = med.sum() + 1e-8
        recall = (fg_probs * med).sum() / n_med
    return 1.0 - recall


def multiscale_topo_bridge(logits, target, kernels=(3, 5, 7), ignore_label=IGNORE_LABEL):
    """v3: Multi-scale bridge penalty at multiple morphological scales.
    Catches thin AND medium bridges. Replaces single-kernel topo_bridge_penalty."""
    with torch.amp.autocast("cuda", enabled=False):
        logits = logits.float().clamp(-20, 20)
        fg_prob = torch.softmax(logits, dim=1)[:, 1]
        p5 = fg_prob.unsqueeze(1)
        valid = (target != ignore_label).float()

        total_penalty = torch.tensor(0.0, device=logits.device)
        for kernel in kernels:
            pad = kernel // 2
            eroded = -F.max_pool3d(-p5, kernel, stride=1, padding=pad)
            opened = F.max_pool3d(eroded, kernel, stride=1, padding=pad)
            opened = opened.squeeze(1)
            thin_mask = (fg_prob - opened).clamp(min=0).detach()
            penalty = (fg_prob * thin_mask * valid).sum() / (valid.sum() + 1e-8)
            total_penalty = total_penalty + penalty

    return total_penalty / len(kernels)


def surface_distance_loss(logits, target, tau=2.0, ignore_label=IGNORE_LABEL):
    """v3: Penalize predicted surface voxels far from GT surface.
    Directly targets SurfaceDice@tau metric. Differentiable via soft boundary."""
    with torch.amp.autocast("cuda", enabled=False):
        logits = logits.float().clamp(-20, 20)
        probs = torch.softmax(logits, dim=1)[:, 1]
        B = target.shape[0]
        total_loss = torch.tensor(0.0, device=logits.device)
        n_valid = 0

        for b in range(B):
            tgt_np = target[b].cpu().numpy()
            valid = (tgt_np != ignore_label)
            fg_gt = (tgt_np == 1)

            if fg_gt.sum() < 10 or (~fg_gt & valid).sum() < 10:
                continue

            # GT boundary via erosion XOR
            eroded = binary_erosion(fg_gt, iterations=1)
            gt_boundary = fg_gt ^ eroded

            if gt_boundary.sum() == 0:
                continue

            # Distance transform from GT boundary
            dt_gt = distance_transform_edt(~gt_boundary).astype(np.float32)
            dt_gt = np.clip(dt_gt, 0, tau * 3)
            dt_gt_t = torch.from_numpy(dt_gt).to(logits.device)

            # Predicted boundary via gradient magnitude of prob map
            p = probs[b]
            # Sobel-like gradient in 3D (sum of squared finite differences)
            gx = (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) ** 2
            gy = (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) ** 2
            gz = (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) ** 2
            grad_mag = torch.sqrt(gx + gy + gz + 1e-8)

            # Pad back to original size
            grad_full = torch.zeros_like(p)
            grad_full[1:-1, 1:-1, 1:-1] = grad_mag

            # Soft boundary = gradient magnitude (high at predicted edges)
            valid_t = torch.from_numpy(valid.astype(np.float32)).to(logits.device)

            # Loss = mean distance of predicted surface points to GT surface
            weighted_dist = (grad_full * dt_gt_t * valid_t).sum()
            norm = grad_full.sum() + 1e-8
            total_loss = total_loss + weighted_dist / norm
            n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=logits.device)
    return total_loss / n_valid


def sdf_regression_loss(sdf_pred, sdf_gt, target, ignore_label=IGNORE_LABEL):
    """v3: Huber loss on clipped signed distance field (aux head).
    sdf_pred: (B, D, H, W) from aux head
    sdf_gt: (B, D, H, W) precomputed GT SDF clipped to [-5, 5]"""
    if sdf_pred is None or sdf_gt is None:
        return torch.tensor(0.0, device=target.device)
    with torch.amp.autocast("cuda", enabled=False):
        sdf_pred = sdf_pred.float()
        sdf_gt_t = sdf_gt.float().to(sdf_pred.device)
        valid = (target != ignore_label).float()
        # Clamp prediction to match GT range
        sdf_pred_c = sdf_pred.clamp(-5, 5)
        # Huber (smooth L1) loss
        loss = F.smooth_l1_loss(sdf_pred_c * valid, sdf_gt_t * valid, reduction='sum')
        n_valid = valid.sum() + 1e-8
    return loss / n_valid


def cldice_loss(logits, target, n_iters=5, ignore_label=IGNORE_LABEL):
    """v3: Centerline Dice — enforces skeleton overlap via soft skeletonization.
    Uses iterative min-pool erosion as differentiable thinning approximation."""
    with torch.amp.autocast("cuda", enabled=False):
        logits = logits.float().clamp(-50, 50)
        probs = torch.softmax(logits, dim=1)[:, 1]
        valid = (target != ignore_label)
        tgt = (target == 1).float()

        # Mask invalid regions
        probs = probs * valid.float()
        tgt = tgt * valid.float()

        def soft_skeletonize(x, iters):
            """Iterative erosion-based soft skeletonization."""
            skel = torch.zeros_like(x)
            residual = x.unsqueeze(1)  # (B, 1, D, H, W)
            for _ in range(iters):
                # Min-pool = morphological erosion
                eroded = -F.max_pool3d(-residual, 3, stride=1, padding=1)
                # Skeleton layer = what erosion removed
                layer = (residual - eroded).squeeze(1)
                skel = skel + layer.clamp(min=0)
                residual = eroded
            # Add remaining core
            skel = skel + residual.squeeze(1)
            return skel

        skel_pred = soft_skeletonize(probs, n_iters)
        skel_gt = soft_skeletonize(tgt, n_iters)

        # cl_precision: how much of predicted skeleton falls on GT
        cl_prec_num = (skel_pred * tgt).sum()
        cl_prec_den = skel_pred.sum() + 1e-8

        # cl_recall: how much of GT skeleton falls on prediction
        cl_rec_num = (probs * skel_gt).sum()
        cl_rec_den = skel_gt.sum() + 1e-8

        cl_prec = cl_prec_num / cl_prec_den
        cl_rec = cl_rec_num / cl_rec_den

        cldice = 2.0 * cl_prec * cl_rec / (cl_prec + cl_rec + 1e-8)
    return 1.0 - cldice


def component_reg_loss(logits, target, ignore_label=IGNORE_LABEL):
    """v3: Soft penalty when predicted components >> GT components.
    Targets VOI_score directly. Only computed every few steps (expensive)."""
    with torch.amp.autocast("cuda", enabled=False):
        logits = logits.float().clamp(-20, 20)
        probs = torch.softmax(logits, dim=1)[:, 1]
        struct26 = generate_binary_structure(3, 3)

        B = target.shape[0]
        total_penalty = torch.tensor(0.0, device=logits.device)

        for b in range(B):
            tgt_np = target[b].cpu().numpy()
            valid = (tgt_np != ignore_label)
            gt_fg = (tgt_np == 1)

            pred_binary = (probs[b].detach().cpu().numpy() > 0.5)
            pred_binary = pred_binary & valid

            if gt_fg.sum() < 100:
                continue

            try:
                _, n_pred = cc_label(pred_binary, structure=struct26)
                _, n_gt = cc_label(gt_fg, structure=struct26)
            except Exception:
                continue

            # Penalize if predicted components are >2x GT components
            ratio = n_pred / max(n_gt, 1)
            if ratio > 2.0:
                penalty_val = min(1.0, (ratio - 2.0) * 0.25)
                # Weight by mean FG prob (gradient flows through probs)
                selected = probs[b][torch.from_numpy(pred_binary).to(probs.device)]
                if selected.numel() == 0:
                    continue
                mean_prob = selected.mean()
                total_penalty = total_penalty + penalty_val * mean_prob

    return total_penalty / max(B, 1)


def topo_guidance_loss(topo_pred, topo_gt, target, ignore_label=IGNORE_LABEL):
    """v3: MSE on distance-to-medial prediction (aux head).
    Forces internal representation of sheet structure."""
    if topo_pred is None or topo_gt is None:
        return torch.tensor(0.0, device=target.device)
    with torch.amp.autocast("cuda", enabled=False):
        topo_pred = topo_pred.float()
        topo_gt_t = topo_gt.float().to(topo_pred.device)
        # Only compute at valid FG voxels
        fg_valid = ((target == 1) & (target != ignore_label)).float()
        loss = ((topo_pred - topo_gt_t) ** 2 * fg_valid).sum()
        n_valid = fg_valid.sum() + 1e-8
    return loss / n_valid


def gap_negative_loss(logits, target, gap_mask, ignore_label=IGNORE_LABEL):
    """Brain doc (7): Anti-bridge gap-negative loss.
    Penalizes FG probability in separation zones between GT surfaces.
    L_gap = mean(N(x) * p(x)) — directly reduces bridges -> VOI_merge + Topo(k1).
    gap_mask: precomputed in dataset — regions between nearby GT surfaces."""
    if gap_mask is None:
        return torch.tensor(0.0, device=logits.device)
    with torch.amp.autocast("cuda", enabled=False):
        logits = logits.float().clamp(-20, 20)
        fg_prob = torch.softmax(logits, dim=1)[:, 1]
        gap_m = gap_mask.float().to(fg_prob.device)
        valid = (target != ignore_label).float()
        gap_valid = gap_m * valid
        n_gap = gap_valid.sum() + 1e-8
        # Penalize FG predictions in gap regions
        penalty = (fg_prob * gap_valid).sum() / n_gap
    return penalty


def tversky_loss(logits, target, alpha=0.3, beta=0.7, gamma=1.0, ignore_label=IGNORE_LABEL):
    """Brain doc (3): Tversky/Focal-Tversky for controlling splits vs merges.
    alpha=weight on FP, beta=weight on FN.
    High beta -> penalize FN more -> anti-split (keeps more FG).
    High alpha -> penalize FP more -> anti-merge (suppresses bridges).
    Use gamma>1 for focal effect (focus on hard examples)."""
    with torch.amp.autocast("cuda", enabled=False):
        logits = logits.float().clamp(-50, 50)
        probs = torch.softmax(logits, dim=1)[:, 1]
        valid = (target != ignore_label)
        tgt = (target == 1).float()
        v = valid.float()
        p = probs * v
        g = tgt * v
        tp = (p * g).sum()
        fp = (p * (1 - g)).sum()
        fn = ((1 - p) * g).sum()
        ti = (tp + 1e-5) / (tp + alpha * fp + beta * fn + 1e-5)
        if gamma != 1.0:
            return (1 - ti).pow(gamma)
        return 1 - ti


def compute_loss(outputs, sdf_pred, topo_pred, target, medial, sdf_gt, topo_gt,
                 gap_mask=None, step_i=0):
    """
    v3: Deep supervision loss with phase-aware scheduling.
    6 core + 2 specialist + 4 optional = comprehensive but phase-gated.
    Returns (total_loss, components_dict).
    """
    total = torch.tensor(0.0, device=target.device)
    comps = {"ce": 0.0, "dice": 0.0, "msr": 0.0, "bnd": 0.0, "topo": 0.0,
             "surfdist": 0.0, "sdf": 0.0, "cldice": 0.0, "compreg": 0.0, "topoguide": 0.0,
             "gapneg": 0.0, "tversky": 0.0}

    # Precompute boundary weight map at full resolution
    bnd_wmap_full = compute_boundary_weight_map(target) if W_BND > 0 else None

    for i, (logits, w) in enumerate(zip(outputs, DS_WEIGHTS)):
        if i > 0:
            w = w * _DS_SCALE
        if w < 1e-6:
            continue
        # Downsample target to match logits resolution
        if logits.shape[2:] != target.shape[1:]:
            tgt = F.interpolate(
                target.float().unsqueeze(1),
                size=logits.shape[2:], mode="nearest"
            ).long().squeeze(1)
        else:
            tgt = target

        # CE with label smoothing
        logits_c = logits.float().clamp(-50, 50)
        ce_mult = _phase_mult("ce")
        ce = F.cross_entropy(logits_c, tgt, ignore_index=IGNORE_LABEL,
                             label_smoothing=LABEL_SMOOTH)
        ce = torch.nan_to_num(ce, nan=0.0, posinf=10.0, neginf=0.0)

        # Dice
        dice_mult = _phase_mult("dice")
        dc = soft_dice_loss(logits, tgt, IGNORE_LABEL)
        dc = torch.nan_to_num(dc, nan=0.0, posinf=10.0, neginf=0.0)

        total = total + w * (W_CE * ce_mult * ce + W_DICE * dice_mult * dc)
        if not torch.isfinite(total):
            return total, comps
        comps["ce"] += w * W_CE * ce_mult * ce.item()
        comps["dice"] += w * W_DICE * dice_mult * dc.item()

        # Boundary-weighted CE at full resolution only
        if W_BND > 0 and i == 0 and bnd_wmap_full is not None:
            bnd_mult = _phase_mult("bnd")
            bce = boundary_weighted_ce(logits, tgt, bnd_wmap_full)
            bce = torch.nan_to_num(bce, nan=0.0, posinf=10.0, neginf=0.0)
            total = total + w * W_BND * bnd_mult * bce
            comps["bnd"] += w * W_BND * bnd_mult * bce.item()

    # === Full-resolution-only losses (no deep supervision) ===
    if len(outputs) == 0:
        return total, comps

    full_logits = outputs[0]
    ramp = _TOPO_RAMP

    # MSR (medial surface recall)
    msr_mult = _phase_mult("msr")
    if W_MSR > 0 and medial.sum() > 0 and ramp > 0 and msr_mult > 0:
        fg_probs = torch.softmax(full_logits.float(), dim=1)[:, 1]
        valid = (target != IGNORE_LABEL)
        msr = medial_surface_recall_loss(fg_probs, medial, valid)
        msr = torch.nan_to_num(msr, nan=0.0, posinf=10.0, neginf=0.0)
        msr_w = W_MSR * ramp * msr_mult
        total = total + msr_w * msr
        comps["msr"] += msr_w * msr.item()

    # Multi-scale topology bridge penalty
    topo_mult = _phase_mult("topo")
    if W_TOPO > 0 and ramp > 0 and topo_mult > 0:
        tp = multiscale_topo_bridge(full_logits, target)
        tp = torch.nan_to_num(tp, nan=0.0, posinf=10.0, neginf=0.0)
        topo_w = W_TOPO * ramp * topo_mult
        total = total + topo_w * tp
        comps["topo"] += topo_w * tp.item()

    # Surface distance loss (SurfaceDice@tau alignment)
    sd_mult = _phase_mult("surfdist")
    if W_SURFDIST > 0 and ramp > 0 and sd_mult > 0:
        sdl = surface_distance_loss(full_logits, target)
        sdl = torch.nan_to_num(sdl, nan=0.0, posinf=10.0, neginf=0.0)
        sd_w = W_SURFDIST * ramp * sd_mult
        total = total + sd_w * sdl
        comps["surfdist"] += sd_w * sdl.item()

    # SDF regression (aux head)
    sdf_mult = _phase_mult("sdf")
    if W_SDF > 0 and sdf_mult > 0:
        sdf_l = sdf_regression_loss(sdf_pred, sdf_gt, target)
        sdf_l = torch.nan_to_num(sdf_l, nan=0.0, posinf=10.0, neginf=0.0)
        sdf_w = W_SDF * sdf_mult
        total = total + sdf_w * sdf_l
        comps["sdf"] += sdf_w * sdf_l.item()

    # clDice: skeleton connectivity. Expensive (5 min-pool iters per call).
    # Computed every K steps to avoid being a compute-tax primary objective.
    # Blueprint: w=0.4 is large; reduce frequency to prevent gradient conflict with CE+Dice.
    # K=5 in late (was 3), K=8 in mid (was 6) — same direction, less dominance per step.
    cl_mult = _phase_mult("cldice")
    _cldice_k = 5 if _LOSS_PHASE == "late" else 8
    if W_CLDICE > 0 and ramp > 0 and cl_mult > 0 and step_i % _cldice_k == 0:
        cld = cldice_loss(full_logits, target)
        cld = torch.nan_to_num(cld, nan=0.0, posinf=10.0, neginf=0.0)
        cl_w = W_CLDICE * ramp * cl_mult
        total = total + cl_w * cld
        comps["cldice"] += cl_w * cld.item()

    # Component regularization: CPU-heavy SciPy CC labeling.
    # Rate-limited: late-only, 1/32 steps, b=0 only — prevents GPU/CPU sync stalls.
    cr_mult = _phase_mult("compreg")
    if (W_COMPREG > 0 and ramp > 0 and cr_mult > 0 and
            _LOSS_PHASE == "late" and step_i % 32 == 0):
        crl = component_reg_loss(full_logits[0:1], target[0:1])   # b=0 only
        crl = torch.nan_to_num(crl, nan=0.0, posinf=10.0, neginf=0.0)
        cr_w = W_COMPREG * ramp * cr_mult
        total = total + cr_w * crl
        comps["compreg"] += cr_w * crl.item()

    # Topology guidance (aux head)
    tg_mult = _phase_mult("topoguide")
    if W_TOPOGUIDE > 0 and tg_mult > 0:
        tgl = topo_guidance_loss(topo_pred, topo_gt, target)
        tgl = torch.nan_to_num(tgl, nan=0.0, posinf=10.0, neginf=0.0)
        tg_w = W_TOPOGUIDE * tg_mult
        total = total + tg_w * tgl
        comps["topoguide"] += tg_w * tgl.item()

    # Brain doc (7): Gap-negative loss — THE anti-bridge weapon
    gn_mult = _phase_mult("gapneg")
    if W_GAPNEG > 0 and ramp > 0 and gn_mult > 0 and gap_mask is not None:
        gnl = gap_negative_loss(full_logits, target, gap_mask)
        gnl = torch.nan_to_num(gnl, nan=0.0, posinf=10.0, neginf=0.0)
        gn_w = W_GAPNEG * ramp * gn_mult
        total = total + gn_w * gnl
        comps["gapneg"] += gn_w * gnl.item()

    # Brain doc (3): Tversky loss — specialist FN/FP control
    tv_mult = _phase_mult("tversky")
    if W_TVERSKY > 0 and tv_mult > 0:
        tvl = tversky_loss(full_logits, target,
                           alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA, gamma=TVERSKY_GAMMA)
        tvl = torch.nan_to_num(tvl, nan=0.0, posinf=10.0, neginf=0.0)
        tv_w = W_TVERSKY * tv_mult
        total = total + tv_w * tvl
        comps["tversky"] += tv_w * tvl.item()

    return total, comps

_active = [n for n, w in [("CE",W_CE),("Dice",W_DICE),("SDF",W_SDF),("SurfDist",W_SURFDIST),
           ("MSR",W_MSR),("Topo",W_TOPO),("GapNeg",W_GAPNEG),("Tversky",W_TVERSKY),
           ("BndCE",W_BND),("clDice",W_CLDICE),("CompReg",W_COMPREG),("TopoGuide",W_TOPOGUIDE)] if w > 0]
print(f"[LOSS] v3 active: {'+'.join(_active)} ({len(_active)} losses)")
print(f"[LOSS] Phase scheduling: early<{PHASE_EARLY_END}, mid<{PHASE_MID_END}, late>=, label_smooth={LABEL_SMOOTH:.3f}")
'''


CELL_DATASET = r'''
# ============================================================
# Dataset + Augmentation v3
# Fixes: hard mining, bridge-risk sampling, SDF/topo GT precompute,
#        boundary-fraction rejection
# ============================================================

def normalize_volume(vol, robust=True):
    """Brain doc: robust z-score (median/MAD) with intensity clipping.
    Prevents outlier voxels from distorting normalization."""
    v = vol.astype(np.float32)
    # Clip at 0.5-99.5 percentile to stabilize tails
    lo, hi = np.percentile(v, [0.5, 99.5])
    v = np.clip(v, lo, hi)
    if robust:
        mu = np.median(v)
        mad = np.median(np.abs(v - mu))
        sigma = mad * 1.4826  # MAD to std conversion factor
    else:
        mu = v.mean()
        sigma = v.std()
    return (v - mu) / max(sigma, 1e-8)


def compute_medial_surface(mask_3d):
    """Medial surface: 3D DT local maxima + 2D cross-axis skeleton aggregation.
    Cross-axis union catches medial voxels missed by 3D DT at curved sheet bends
    and at the surface-normal orientation boundaries. Directly improves MSR recall."""
    if mask_3d.sum() == 0:
        return np.zeros_like(mask_3d, dtype=np.float32)
    fg = mask_3d.astype(bool)
    # 3D component: local maxima of the interior DT (0.001 tolerance for flat-sheet ties)
    dt = distance_transform_edt(fg)
    lm = maximum_filter(dt, size=3)
    medial_3d = (dt > 0) & (dt >= lm - 0.001)
    # 2D cross-axis aggregation: compute medial axis per slice along each axis.
    # For a papyrus sheet running at an angle, 2D DT in some slice orientations
    # gives a denser, more stable medial axis than 3D DT alone.
    medial_2d = np.zeros_like(fg, dtype=bool)
    for axis in range(3):
        vol_ax = np.moveaxis(fg, axis, 0)
        result_ax = np.zeros(vol_ax.shape, dtype=bool)
        for i in range(vol_ax.shape[0]):
            sl = vol_ax[i]
            if sl.sum() < 2:
                continue
            dt2 = distance_transform_edt(sl)
            lm2 = maximum_filter(dt2, size=3)
            result_ax[i] = (dt2 > 0) & (dt2 >= lm2 - 0.001)
        medial_2d |= np.moveaxis(result_ax, 0, axis)
    return (medial_3d | medial_2d).astype(np.float32)


def compute_sdf_gt(mask_3d, clip_val=5.0):
    """v3: Compute clipped signed distance field from binary mask.
    Negative = inside, positive = outside. Shares DT with medial."""
    fg = mask_3d.astype(bool)
    if fg.sum() == 0:
        return np.full(mask_3d.shape, clip_val, dtype=np.float32)
    if (~fg).sum() == 0:
        return np.full(mask_3d.shape, -clip_val, dtype=np.float32)
    dt_inside = distance_transform_edt(fg)
    dt_outside = distance_transform_edt(~fg)
    sdf = dt_outside - dt_inside  # positive outside, negative inside
    return np.clip(sdf, -clip_val, clip_val).astype(np.float32)


def compute_topo_gt(mask_3d):
    """v3: Compute normalized distance-to-medial for topology guidance.
    Output in [0, 1] at FG voxels, 0 at BG."""
    fg = mask_3d.astype(bool)
    if fg.sum() == 0:
        return np.zeros(mask_3d.shape, dtype=np.float32)
    dt = distance_transform_edt(fg)
    max_dt = dt.max()
    if max_dt < 1e-8:
        return np.zeros(mask_3d.shape, dtype=np.float32)
    # Normalized distance: 0 at boundary, 1 at medial axis
    topo = (dt / max_dt).astype(np.float32)
    topo[~fg] = 0.0
    return topo


_ELASTIC_P   = 0.15
_MEDIAL_P    = 1.0
_LOSS_PHASE  = "early"
_OOM_SHRINK  = 0
_OOM_THIS_EPOCH = 0
TOPO_RAMP_FRAC = 0.20   # completes 5% into mid phase — smooth gradient when phase mult jumps


def elastic_deform_3d(img, lbl, med, sdf=None, topo=None, alpha=6.0, grid_step=32):
    """Elastic deformation with support for SDF and topo GT maps."""
    if random.random() > _ELASTIC_P:
        return img, lbl, med, sdf, topo
    shape = img.shape
    cshape = tuple(max(2, s // grid_step + 1) for s in shape)
    rng_state = np.random.RandomState()
    dz = rng_state.randn(*cshape).astype(np.float32) * alpha
    dy = rng_state.randn(*cshape).astype(np.float32) * alpha
    dx = rng_state.randn(*cshape).astype(np.float32) * alpha
    zoom_f = tuple(s / c for s, c in zip(shape, cshape))
    dz = ndi.zoom(dz, zoom_f, order=3)[:shape[0], :shape[1], :shape[2]]
    dy = ndi.zoom(dy, zoom_f, order=3)[:shape[0], :shape[1], :shape[2]]
    dx = ndi.zoom(dx, zoom_f, order=3)[:shape[0], :shape[1], :shape[2]]
    z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                           np.arange(shape[2]), indexing='ij')
    coords = [z + dz, y + dy, x + dx]
    img = ndi.map_coordinates(img, coords, order=1, mode='reflect').astype(np.float32)
    lbl = ndi.map_coordinates(lbl, coords, order=0, mode='constant',
                              cval=IGNORE_LABEL).astype(np.uint8)
    med = ndi.map_coordinates(med, coords, order=0, mode='constant', cval=0).astype(np.float32)
    if sdf is not None:
        sdf = ndi.map_coordinates(sdf, coords, order=1, mode='constant', cval=5.0).astype(np.float32)
    if topo is not None:
        topo = ndi.map_coordinates(topo, coords, order=1, mode='constant', cval=0.0).astype(np.float32)
    return img, lbl, med, sdf, topo


def augment_patch(img, lbl, med, sdf=None, topo=None):
    """Numpy augmentation for a 3D patch with SDF/topo support."""
    img, lbl, med, sdf, topo = elastic_deform_3d(img, lbl, med, sdf, topo)

    for ax in range(3):
        if random.random() < AUG_FLIP_PROB:
            img = np.flip(img, axis=ax).copy()
            lbl = np.flip(lbl, axis=ax).copy()
            med = np.flip(med, axis=ax).copy()
            if sdf is not None:
                sdf = np.flip(sdf, axis=ax).copy()
            if topo is not None:
                topo = np.flip(topo, axis=ax).copy()

    if random.random() < AUG_ROT90_PROB:
        k = random.choice([1, 2, 3])
        img = np.rot90(img, k, axes=(1, 2)).copy()
        lbl = np.rot90(lbl, k, axes=(1, 2)).copy()
        med = np.rot90(med, k, axes=(1, 2)).copy()
        if sdf is not None:
            sdf = np.rot90(sdf, k, axes=(1, 2)).copy()
        if topo is not None:
            topo = np.rot90(topo, k, axes=(1, 2)).copy()

    # Intensity augmentations (image only)
    if random.random() < AUG_BRIGHT_PROB:
        img = img + np.random.uniform(-AUG_BRIGHT_RANGE, AUG_BRIGHT_RANGE)
    if random.random() < AUG_CONTRAST_PROB:
        img = img * np.random.uniform(0.93, 1.07)
    if random.random() < AUG_GAMMA_PROB:
        mn = img.min()
        rng = img.max() - mn + 1e-8
        img_01 = (img - mn) / rng
        gamma = np.exp(np.random.uniform(-0.2, 0.3))
        img = np.power(np.clip(img_01, 1e-8, None), gamma) * rng + mn
    if random.random() < AUG_NOISE_PROB:
        img = img + np.random.normal(0, AUG_NOISE_STD, img.shape).astype(np.float32)

    # Random cuboid cutout
    if random.random() < 0.12:
        cs = [random.randint(s // 8, s // 4) for s in img.shape]
        cz = random.randint(0, img.shape[0] - cs[0])
        cy = random.randint(0, img.shape[1] - cs[1])
        cx = random.randint(0, img.shape[2] - cs[2])
        img[cz:cz+cs[0], cy:cy+cs[1], cx:cx+cs[2]] = 0.0

    return img, lbl, med, sdf, topo


class _LRU:
    """Simple LRU cache."""
    def __init__(self, cap=2):
        self.cap = cap
        self.od = OrderedDict()
    def get(self, key):
        if key in self.od:
            self.od.move_to_end(key)
            return self.od[key]
        return None
    def put(self, key, val):
        self.od[key] = val
        self.od.move_to_end(key)
        while len(self.od) > self.cap:
            self.od.popitem(last=False)


class VesuviusPatchDataset(Dataset):
    """
    v3: Fixed hard mining, bridge-risk sampling, SDF/topo GT precompute.
    Returns (img, lbl, medial, sdf_gt, topo_gt, vol_idx, center_coords).
    """
    SWAP_KEEP_FRAC = 0.62
    SWAP_INTERVAL_SEC = 200   # was 300: rotate volumes every 3.3 min instead of 5 min
    # → ~50% more distinct volumes seen per 8.5h run; each swap replaces 38% of set

    def __init__(self, vol_infos, patch_size, num_iters, augment=True):
        self.vols = list(vol_infos)
        self.ps = tuple(patch_size)
        self.num_iters = num_iters
        self.augment = augment
        max_k = min(N_EPOCH_VOLS * 2, len(self.vols))
        self.vol_cache = _LRU(max(max_k + 2, 10))
        self.coord_cache = _LRU(max(max_k + 4, 12))
        self.hard_coords = []
        self._epoch = 0
        self._vol_queue = []
        self._queue_seed_ctr = 0
        self._current_k = min(N_EPOCH_VOLS, len(self.vols))
        self._active_set = list(range(self._current_k))
        self._swaps_this_epoch = 0
        self._last_swap_time = time.time()

    def __len__(self):
        return self.num_iters

    def _refill_queue(self):
        rng = random.Random(SEED + self._queue_seed_ctr)
        self._queue_seed_ctr += 1
        q = list(range(len(self.vols)))
        rng.shuffle(q)
        self._vol_queue = q

    def _next_active_set(self):
        k = self._current_k
        n_keep = max(1, int(k * self.SWAP_KEEP_FRAC))
        n_new = k - n_keep
        if len(self._active_set) >= n_keep:
            # Deterministic per epoch×swap: global random.sample would diverge across workers
            _swap_rng = random.Random((SEED * 99991 + self._epoch * 1009 + self._swaps_this_epoch) & 0x7FFFFFFF)
            kept = _swap_rng.sample(self._active_set, n_keep)
        else:
            kept = list(self._active_set)
            n_new = k - len(kept)
        if len(self._vol_queue) < n_new:
            self._refill_queue()
        new_vols = [self._vol_queue.pop() for _ in range(n_new)]
        self._active_set = kept + new_vols
        self._last_swap_time = time.time()

    def _prewarm(self):
        for vi in self._active_set:
            info = self.vols[vi]
            img, lbl = self._load(info)
            self._get_coords(info, lbl)

    def set_epoch(self, epoch, elapsed_hours=0.0):
        """v3: Three-phase control with bridge-risk and SDF/topo scheduling."""
        global _ELASTIC_P, _MEDIAL_P, _LOSS_PHASE, _TOPO_RAMP, _DS_SCALE
        global _OOM_SHRINK
        self._epoch = epoch
        self._swaps_this_epoch = 0
        time_frac = min(1.0, elapsed_hours / MAX_TRAIN_HOURS)
        prev_phase = _LOSS_PHASE

        # Topo/MSR ramp-in
        if time_frac < TOPO_RAMP_FRAC:
            _TOPO_RAMP = min(1.0, time_frac / max(TOPO_RAMP_FRAC, 1e-8))
        else:
            _TOPO_RAMP = 1.0

        # v3: Three-phase curriculum
        if time_frac < PHASE_EARLY_END:
            self.ps = tuple(PATCH_SIZE_EARLY)
            _LOSS_PHASE = "early"
        elif time_frac < PHASE_MID_END:
            # Mid phase: always 128^3.
            # 160^3 every-4th-epoch strategy caused a repeating OOM-DEGRADE cycle:
            #   160^3 OOMs → OOM_SHRINK=3 → base 128 reduced to 80^3 → wastes 3/4 epochs.
            #   Net effect: 75% of mid-phase epochs run at 80^3 (too small for topology).
            # Fix: cap at 128^3 throughout mid phase. Stable, no OOM, full patch coverage.
            self.ps = tuple(min(p, 128) for p in PATCH_SIZE)
            _LOSS_PHASE = "mid"
        else:
            # Late phase: cap at 128³.
            # 160³ caused immediate OOM-DEGRADE at late-phase entry (K=8 → more RAM pressure).
            # 128³ confirmed stable through ep29 (Model B ran 128³ in late without GPU OOM).
            # DataLoader worker SIGKILL at 5.5h was from num_workers=4 (now fixed to 2),
            # not from 128³ patch size — 128³ late-phase is safe.
            self.ps = tuple(min(p, 128) for p in PATCH_SIZE)
            _LOSS_PHASE = "late"

        # OOM degradation
        if _OOM_SHRINK > 0:
            shrink = _OOM_SHRINK * 16
            self.ps = tuple(max(64, p - shrink) for p in self.ps)
            if _OOM_SHRINK >= 2:
                _ELASTIC_P = 0.0

        # Volume curriculum (conservative K to avoid I/O bottleneck)
        if time_frac < 0.25:
            self._current_k = 4
        elif time_frac < 0.55:
            self._current_k = min(6, N_EPOCH_VOLS)
        else:
            self._current_k = min(N_EPOCH_VOLS, len(self.vols))
        self._current_k = min(self._current_k, len(self.vols))

        # Phase-specific schedules
        if _LOSS_PHASE == "early":
            _ELASTIC_P = 0.15
            _MEDIAL_P = 0.5
            _DS_SCALE = 1.0
        elif _LOSS_PHASE == "mid":
            _ELASTIC_P = 0.10
            _MEDIAL_P = 0.8
            _DS_SCALE = 0.8
        else:  # late
            _ELASTIC_P = 0.03
            _MEDIAL_P = 1.0
            _DS_SCALE = 0.5

        if _LOSS_PHASE != prev_phase:
            pD, pH, pW = self.ps
            print(f"\n  >>> PHASE TRANSITION: {prev_phase} -> {_LOSS_PHASE} "
                  f"(t={time_frac:.1%}, patch={pD}x{pH}x{pW}, "
                  f"elastic={_ELASTIC_P:.2f})", flush=True)

        self._next_active_set()
        self._prewarm()

    def add_hard_coord(self, vol_idx, z, y, x):
        """v3 FIX: Called with ACTUAL volume index and patch center coords.
        Playbook: per-volume cap prevents overfitting to one bad volume."""
        self.hard_coords.append((vol_idx, int(z), int(y), int(x)))
        if len(self.hard_coords) > HARD_CACHE_SIZE:
            self.hard_coords = self.hard_coords[-HARD_CACHE_SIZE:]
        # Playbook: per-volume cap — max 30% of buffer from any one volume
        max_per_vol = max(10, HARD_CACHE_SIZE // 3)
        from collections import Counter
        vol_counts = Counter(c[0] for c in self.hard_coords)
        for vid, cnt in vol_counts.items():
            if cnt > max_per_vol:
                new_coords = []
                kept = 0
                for c in reversed(self.hard_coords):
                    if c[0] == vid:
                        if kept < max_per_vol:
                            new_coords.append(c)
                            kept += 1
                    else:
                        new_coords.append(c)
                self.hard_coords = list(reversed(new_coords))
                break

    def _compute_vol_tags(self, lbl):
        """Playbook: Tag volumes for stratification (packed/sparse/curved/noisy)."""
        fg = (lbl == 1)
        fg_frac = fg.sum() / max(fg.size, 1)
        tags = []
        if fg_frac > 0.35:
            tags.append("packed")
        elif fg_frac < 0.10:
            tags.append("sparse")
        else:
            tags.append("moderate")
        if fg.any():
            dt = distance_transform_edt(fg)
            mean_thick = dt[fg].mean() * 2
            if mean_thick < 4.0:
                tags.append("thin")
            elif mean_thick > 12.0:
                tags.append("thick")
        struct26 = generate_binary_structure(3, 3)
        _, n_cc = cc_label(fg, structure=struct26)
        if n_cc > 15:
            tags.append("fragmented")
        elif n_cc <= 3:
            tags.append("consolidated")
        return tags

    def _load(self, info):
        vid = info["id"]
        cached = self.vol_cache.get(vid)
        if cached is not None:
            return cached
        img = read_tif(info["image"])
        lbl = read_tif(info["label"])
        img = normalize_volume(img)
        lbl = lbl.astype(np.uint8)
        lbl[lbl == 2] = IGNORE_LABEL
        tags = self._compute_vol_tags(lbl)
        print(f"[DATA] Volume {vid} tags: {tags}")
        self.vol_cache.put(vid, (img, lbl))
        return img, lbl

    def _get_coords(self, info, lbl):
        vid = info["id"]
        # 1. In-memory LRU (hot path — same worker, same epoch)
        cached = self.coord_cache.get(vid)
        if cached is not None:
            return cached

        # 2. Disk cache — shared across all workers and epochs.
        # persistent_workers=False re-forks workers every epoch, killing the in-memory cache.
        # Without disk cache: 12 active volumes × 50s coord compute = 600s wasted per epoch.
        # With disk cache: 0.5s disk load after first encounter = 100× faster.
        _disk_path = os.path.join(COORD_CACHE_DIR, f"{vid}.npz")
        if os.path.exists(_disk_path):
            try:
                _d = np.load(_disk_path, allow_pickle=False)
                result = (_d["bnd"], _d["ring"], _d["fg"], _d["bg"],
                          _d["bridge"], _d["thin"], _d["curv"])
                self.coord_cache.put(vid, result)
                return result
            except Exception:
                pass  # corrupt cache — fall through to recompute

        fg = (lbl == 1)
        bg = (lbl == 0)

        # Boundary coords
        if fg.any() and bg.any():
            dilated = ndi.binary_dilation(fg, iterations=BOUNDARY_DILATE)
            boundary = dilated & bg
            eroded = ndi.binary_erosion(fg, iterations=BOUNDARY_DILATE)
            boundary = boundary | (fg & ~eroded)
            bnd_coords = np.argwhere(boundary)
        else:
            bnd_coords = np.empty((0, 3), dtype=np.int64)

        # Ring negatives
        if fg.any():
            ring_mask = ndi.binary_dilation(fg, iterations=RING_WIDTH) & ~fg & (lbl != IGNORE_LABEL)
            ring_coords = np.argwhere(ring_mask)
        else:
            ring_coords = np.empty((0, 3), dtype=np.int64)

        fg_coords = np.argwhere(fg)
        bg_coords = np.argwhere(bg)

        # v3: Bridge-risk coords (regions where two FG surfaces are close)
        bridge_risk_coords = np.empty((0, 3), dtype=np.int64)
        if fg.any() and P_BRIDGE_RISK > 0:
            try:
                struct26 = generate_binary_structure(3, 3)
                dilated_fg = ndi.binary_dilation(fg, iterations=5)
                cc_dil, n_cc = cc_label(dilated_fg, structure=struct26)
                # Find overlap regions between different dilated components
                if n_cc > 1:
                    # Regions where dilated FG from different components overlap
                    overlap = np.zeros_like(fg, dtype=bool)
                    for ci in range(1, min(n_cc + 1, 20)):  # cap to avoid slowness
                        comp_i = (cc_dil == ci)
                        other = dilated_fg & ~comp_i
                        overlap |= (comp_i & other)
                    if overlap.any():
                        bridge_risk_coords = np.argwhere(overlap)
            except Exception:
                pass

        # v3: Thin-region (split-risk) coords — where sheet thickness is small
        # Brain doc: sample where T_G(x) small but FG true (anti-split)
        thin_risk_coords = np.empty((0, 3), dtype=np.int64)
        if fg.any() and P_THIN_RISK > 0:
            try:
                dt_fg = distance_transform_edt(fg)
                # Thin FG voxels: inside FG but close to boundary (thickness < 3 voxels)
                thin_mask = fg & (dt_fg > 0) & (dt_fg <= 3.0)
                if thin_mask.any():
                    thin_risk_coords = np.argwhere(thin_mask)
            except Exception:
                pass

        # Playbook: Curvature sampler — high-curvature sheet bends (splits/holes born here)
        curvature_coords = np.empty((0, 3), dtype=np.int64)
        if fg.any() and P_CURVATURE > 0:
            try:
                if 'dt_fg' not in locals():
                    dt_fg = distance_transform_edt(fg)
                # Laplacian magnitude as curvature proxy
                laplacian = ndi.laplace(dt_fg.astype(np.float32))
                abs_lap = np.abs(laplacian)
                # High curvature: top 10% of Laplacian magnitude inside FG
                fg_lap = abs_lap[fg]
                if len(fg_lap) > 100:
                    thresh = np.percentile(fg_lap[fg_lap > 0], 90) if (fg_lap > 0).any() else 1.0
                    curv_mask = fg & (abs_lap >= thresh)
                    if curv_mask.any():
                        curvature_coords = np.argwhere(curv_mask)
            except Exception:
                pass

        # Cap coord arrays for memory
        cap = 100000
        all_coord_arrays = {
            'bnd_coords': bnd_coords, 'ring_coords': ring_coords,
            'fg_coords': fg_coords, 'bg_coords': bg_coords,
            'bridge_risk_coords': bridge_risk_coords, 'thin_risk_coords': thin_risk_coords,
            'curvature_coords': curvature_coords,
        }
        for arr_name, arr in all_coord_arrays.items():
            if len(arr) > cap:
                idx = np.random.choice(len(arr), cap, replace=False)
                all_coord_arrays[arr_name] = arr[idx]
        bnd_coords = all_coord_arrays['bnd_coords']
        ring_coords = all_coord_arrays['ring_coords']
        fg_coords = all_coord_arrays['fg_coords']
        bg_coords = all_coord_arrays['bg_coords']
        bridge_risk_coords = all_coord_arrays['bridge_risk_coords']
        thin_risk_coords = all_coord_arrays['thin_risk_coords']
        curvature_coords = all_coord_arrays['curvature_coords']

        result = (bnd_coords, ring_coords, fg_coords, bg_coords, bridge_risk_coords, thin_risk_coords, curvature_coords)
        self.coord_cache.put(vid, result)

        # Save to disk cache (int32 + compression: ~1-2 MB per volume vs 17 MB uncompressed).
        # Multiple workers may write simultaneously — last writer wins, all results are valid.
        try:
            _tmp = _disk_path + f".tmp{os.getpid()}"
            np.savez_compressed(_tmp,
                bnd=bnd_coords.astype(np.int32), ring=ring_coords.astype(np.int32),
                fg=fg_coords.astype(np.int32),   bg=bg_coords.astype(np.int32),
                bridge=bridge_risk_coords.astype(np.int32),
                thin=thin_risk_coords.astype(np.int32),
                curv=curvature_coords.astype(np.int32))
            os.replace(_tmp, _disk_path)   # atomic rename — no partial reads by other workers
        except Exception:
            pass  # non-fatal: training continues, just no disk cache for this volume

        return result

    def _sample_center(self, coords, shape):
        pD, pH, pW = self.ps
        D, H, W = shape
        if len(coords) == 0:
            z = random.randint(0, max(0, D - pD))
            y = random.randint(0, max(0, H - pH))
            x = random.randint(0, max(0, W - pW))
            return z, y, x
        idx = random.randint(0, len(coords) - 1)
        cz, cy, cx = coords[idx]
        z = max(0, min(cz - pD // 2, D - pD))
        y = max(0, min(cy - pH // 2, H - pH))
        x = max(0, min(cx - pW // 2, W - pW))
        return int(z), int(y), int(x)

    def _extract_patch(self, img, lbl, coords, shape):
        pD, pH, pW = self.ps

        for attempt in range(MAX_REJECT_ATTEMPTS):
            if shape[0] < pD or shape[1] < pH or shape[2] < pW:
                pad_d = max(0, pD - shape[0])
                pad_h = max(0, pH - shape[1])
                pad_w = max(0, pW - shape[2])
                img_p = np.pad(img, ((0,pad_d),(0,pad_h),(0,pad_w)), mode='constant')
                lbl_p = np.pad(lbl, ((0,pad_d),(0,pad_h),(0,pad_w)),
                               mode='constant', constant_values=IGNORE_LABEL)
                ip = img_p[:pD, :pH, :pW].copy()
                lp = lbl_p[:pD, :pH, :pW].copy()
                return ip, lp, (0, 0, 0)

            z, y, x = self._sample_center(coords, shape)
            ip = img[z:z+pD, y:y+pH, x:x+pW].copy()
            lp = lbl[z:z+pD, y:y+pH, x:x+pW].copy()

            if ip.shape != (pD, pH, pW):
                continue

            # Reject if too much IGNORE
            ign_frac = (lp == IGNORE_LABEL).sum() / lp.size
            if ign_frac > IGNORE_REJECT:
                continue

            # v3: Reject patches with no boundary content (boring)
            fg_frac = (lp == 1).sum() / lp.size
            bg_frac = (lp == 0).sum() / lp.size
            has_boundary = fg_frac > 0.01 and bg_frac > 0.01
            if not has_boundary and attempt < MAX_REJECT_ATTEMPTS - 2:
                continue

            return ip, lp, (z, y, x)

        # Fallback
        z = random.randint(0, max(0, shape[0] - pD))
        y = random.randint(0, max(0, shape[1] - pH))
        x = random.randint(0, max(0, shape[2] - pW))
        ip = img[z:z+pD, y:y+pH, x:x+pW].copy()
        lp = lbl[z:z+pD, y:y+pH, x:x+pW].copy()
        return ip, lp, (z, y, x)

    def _make_aux_targets(self, lp, med):
        """v3: Compute SDF, topo GT, and gap-negative mask from label patch.
        Phase-aware: skip topo and gap in early phase (phase mult=0.0 => results unused)."""
        fg_mask = (lp == 1)
        sdf = compute_sdf_gt(fg_mask)
        if _LOSS_PHASE == "early":
            # topo mult=0.0, gapneg mult=0.0 in early — skip expensive CPU ops
            topo = np.zeros(fg_mask.shape, dtype=np.float32)
            gap_mask = np.zeros(fg_mask.shape, dtype=np.float32)
        else:
            topo = compute_topo_gt(fg_mask)
            # Brain doc (7): Gap-negative mask — separation zones between GT surfaces
            gap_mask = self._compute_gap_mask(fg_mask, lp)
        return sdf, topo, gap_mask

    def _compute_gap_mask(self, fg_mask, lp):
        """Brain doc: construct gap-negative mask N(x) where GT indicates
        'between wraps' / separation zones. N=1 in narrow gaps between FG surfaces."""
        gap = np.zeros_like(fg_mask, dtype=np.float32)
        if fg_mask.sum() < 100 or (~fg_mask & (lp != IGNORE_LABEL)).sum() < 100:
            return gap
        try:
            struct26 = generate_binary_structure(3, 3)
            # Dilate FG surfaces and find gap regions
            dilated = ndi.binary_dilation(fg_mask, iterations=3)
            bg_near_fg = dilated & ~fg_mask & (lp != IGNORE_LABEL)
            if not bg_near_fg.any():
                return gap
            # Gap = background voxels near FG on multiple sides
            # Compute distance from each BG voxel to FG
            dt_from_fg = distance_transform_edt(~fg_mask)
            # Gap voxels: close to FG (within 5 voxels) but not FG
            close_mask = (dt_from_fg > 0) & (dt_from_fg <= 5.0) & (lp != IGNORE_LABEL) & ~fg_mask
            gap = close_mask.astype(np.float32)
        except Exception:
            pass
        return gap

    def __getitem__(self, _idx):
        pD, pH, pW = self.ps

        # Per-sample deterministic RNG: prevents worker duplication and cross-epoch repetition.
        # All sampling decisions (region type, coord selection, medial surface toggle) use
        # this local RNG so workers never produce identical patch streams.
        _wi = torch.utils.data.get_worker_info()
        _wid = _wi.id if _wi is not None else 0
        _rng = random.Random(hash((SEED, int(self._epoch), int(_idx), int(_wid))) & 0x7FFFFFFF)

        # Hard mining path (v3 FIX: uses actual coords now)
        if self.hard_coords and _rng.random() < P_HARD:
            vi, hz, hy, hx = _rng.choice(self.hard_coords)
            vi = vi % len(self.vols)
            info = self.vols[vi]
            img, lbl = self._load(info)
            shape = img.shape
            z = max(0, min(hz - pD // 2, max(0, shape[0] - pD)))
            y = max(0, min(hy - pH // 2, max(0, shape[1] - pH)))
            x = max(0, min(hx - pW // 2, max(0, shape[2] - pW)))
            ip = img[z:z+pD, y:y+pH, x:x+pW].copy()
            lp = lbl[z:z+pD, y:y+pH, x:x+pW].copy()
            if ip.shape == (pD, pH, pW):
                fg_mask = (lp == 1)
                med = compute_medial_surface(fg_mask) if _rng.random() < _MEDIAL_P else np.zeros_like(fg_mask, dtype=np.float32)
                sdf, topo, gap = self._make_aux_targets(lp, med)
                if self.augment:
                    ip, lp, med, sdf, topo = augment_patch(ip, lp, med, sdf, topo)
                return (torch.from_numpy(ip.astype(np.float32)).unsqueeze(0),
                        torch.from_numpy(lp.astype(np.int64)),
                        torch.from_numpy(med.astype(np.float32)),
                        torch.from_numpy(sdf.astype(np.float32)),
                        torch.from_numpy(topo.astype(np.float32)),
                        torch.from_numpy(gap.astype(np.float32)),
                        vi, (z + pD // 2, y + pH // 2, x + pW // 2))

        # Time-based mid-epoch rotation
        if _idx > 0 and (time.time() - self._last_swap_time) > self.SWAP_INTERVAL_SEC:
            self._next_active_set()
            self._prewarm()
            self._swaps_this_epoch += 1

        # Volume cycling within active window
        vol_idx = self._active_set[_idx % len(self._active_set)]
        info = self.vols[vol_idx]
        img, lbl = self._load(info)
        bnd, ring, fg, bg, bridge_risk, thin_risk, curvature = self._get_coords(info, lbl)

        # v3: Region sampling with bridge-risk + thin-risk + curvature (brain doc mixture model)
        r = _rng.random()
        cumul = 0.0
        if r < (cumul := P_BOUNDARY) and len(bnd) > 0:
            coords = bnd
        elif r < (cumul := cumul + P_RING) and len(ring) > 0:
            coords = ring
        elif r < (cumul := cumul + P_BRIDGE_RISK) and len(bridge_risk) > 0:
            coords = bridge_risk
        elif r < (cumul := cumul + P_THIN_RISK) and len(thin_risk) > 0:
            coords = thin_risk
        elif r < (cumul := cumul + P_CURVATURE) and len(curvature) > 0:
            coords = curvature
        elif r < (cumul := cumul + P_FG) and len(fg) > 0:
            coords = fg
        elif len(bg) > 0:
            coords = bg
        else:
            coords = fg if len(fg) > 0 else bnd

        ip, lp, center = self._extract_patch(img, lbl, coords, img.shape)

        fg_mask = (lp == 1)
        if _rng.random() < _MEDIAL_P:
            med = compute_medial_surface(fg_mask)
        else:
            med = np.zeros_like(fg_mask, dtype=np.float32)

        sdf, topo, gap = self._make_aux_targets(lp, med)

        if self.augment:
            ip, lp, med, sdf, topo = augment_patch(ip, lp, med, sdf, topo)

        img_t = torch.from_numpy(ip.astype(np.float32)).unsqueeze(0)
        lbl_t = torch.from_numpy(lp.astype(np.int64))
        med_t = torch.from_numpy(med.astype(np.float32))
        sdf_t = torch.from_numpy(sdf.astype(np.float32))
        topo_t = torch.from_numpy(topo.astype(np.float32))
        gap_t = torch.from_numpy(gap.astype(np.float32))

        # Convert origin to center for hard mining
        center_coord = (center[0] + pD // 2, center[1] + pH // 2, center[2] + pW // 2)
        return img_t, lbl_t, med_t, sdf_t, topo_t, gap_t, vol_idx, center_coord

print("[DATA] v3 Dataset ready (bridge-risk, thin-risk, SDF/topo GT, fixed hard mining)")
'''


CELL_TRAINING = r'''
# ============================================================
# Training Loop v3:
# - 3-phase curriculum (early/mid/late)
# - LR restart at phase transitions
# - SWA in final 10%
# - Mini-leaderboard validation with metric proxies
# - Fixed hard mining (actual coords)
# ============================================================

def cosine_warmup_lr(epoch, max_epoch, initial_lr, warmup=WARMUP_EPOCHS, floor=LR_FLOOR):
    """Cosine annealing with linear warmup, TIME-BASED."""
    time_frac = min(1.0, elapsed_h() / MAX_TRAIN_HOURS)
    warmup_frac = warmup / max(max_epoch, 1)
    if time_frac < warmup_frac:
        return max(floor, initial_lr * time_frac / max(warmup_frac, 1e-8))
    progress = (time_frac - warmup_frac) / max(1e-8, 1.0 - warmup_frac)
    return max(floor, initial_lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


def phase_aware_lr(epoch, initial_lr, time_frac):
    """v3: LR with restart at phase transitions.
    Brain doc: NEVER allow LR to collapse before high-res training.
    Uses cosine-with-floor so LR stays >= LR_FLOOR (3e-4) at all times.
    Late phase gets generous restart (30% of initial) and decays to floor only."""
    warmup_frac = WARMUP_EPOCHS / max(EPOCHS_BUDGET, 1)

    if time_frac < PHASE_EARLY_END:
        # Early phase: standard cosine from initial_lr -> LR_FLOOR
        if time_frac < warmup_frac:
            return max(LR_FLOOR, initial_lr * time_frac / max(warmup_frac, 1e-8))
        progress = (time_frac - warmup_frac) / max(1e-8, PHASE_EARLY_END - warmup_frac)
        return LR_FLOOR + (initial_lr - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    elif time_frac < PHASE_MID_END:
        # Mid phase: restart at LR_RESTART_FRAC_MID * initial_lr -> LR_FLOOR
        restart_lr = initial_lr * LR_RESTART_FRAC_MID
        phase_progress = (time_frac - PHASE_EARLY_END) / max(1e-8, PHASE_MID_END - PHASE_EARLY_END)
        # Mini warmup for first 5% of mid phase
        if phase_progress < 0.05:
            return max(LR_FLOOR, restart_lr * phase_progress / 0.05)
        adj_progress = (phase_progress - 0.05) / 0.95
        return LR_FLOOR + (restart_lr - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * min(adj_progress, 1.0)))

    else:
        # Late phase: restart at LR_RESTART_FRAC_LATE * initial_lr -> LR_FLOOR
        # Brain doc: this is THE stage that matters. Keep LR alive.
        restart_lr = initial_lr * LR_RESTART_FRAC_LATE
        phase_progress = (time_frac - PHASE_MID_END) / max(1e-8, 1.0 - PHASE_MID_END)
        # Mini warmup
        if phase_progress < 0.05:
            return max(LR_FLOOR, restart_lr * phase_progress / 0.05)
        # If in SWA zone, use SWA_LR (but still >= LR_FLOOR)
        if time_frac >= SWA_START_FRAC:
            return max(LR_FLOOR, SWA_LR)
        # Cosine from restart_lr to LR_FLOOR over the non-SWA portion of late phase
        swa_phase_start = (SWA_START_FRAC - PHASE_MID_END) / max(1e-8, 1.0 - PHASE_MID_END)
        adj_progress = (phase_progress - 0.05) / max(1e-8, swa_phase_start - 0.05)
        return LR_FLOOR + (restart_lr - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * min(adj_progress, 1.0)))


class EMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for k, v in model.state_dict().items():
            self.shadow[k] = v.float().clone().detach()

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1 - self.decay)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def apply(self, model):
        model.load_state_dict(self.shadow)

    def save(self, path):
        torch.save(self.shadow, path)


# ---- v3: Metric proxy functions for mini-leaderboard validation ----

def surface_dice_proxy(pred, gt, tau=2.0):
    """Approximate SurfaceDice@tau using distance transforms."""
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    if pred_bool.sum() == 0 or gt_bool.sum() == 0:
        return 0.0

    # Boundaries via erosion XOR
    pred_surface = pred_bool ^ binary_erosion(pred_bool, iterations=1)
    gt_surface = gt_bool ^ binary_erosion(gt_bool, iterations=1)

    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return 0.0

    dt_gt = distance_transform_edt(~gt_surface)
    dt_pred = distance_transform_edt(~pred_surface)

    precision = (dt_gt[pred_surface] <= tau).mean()
    recall = (dt_pred[gt_surface] <= tau).mean()
    return float(2 * precision * recall / (precision + recall + 1e-8))


def voi_proxy(pred, gt):
    """Approximate VOI score from connected component statistics."""
    struct26 = generate_binary_structure(3, 3)
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    if pred_bool.sum() == 0 or gt_bool.sum() == 0:
        return 0.0

    cc_pred, n_pred = cc_label(pred_bool, structure=struct26)
    cc_gt, n_gt = cc_label(gt_bool, structure=struct26)

    if n_pred == 0 or n_gt == 0:
        return 0.0

    # Compute conditional entropies H(gt|pred) and H(pred|gt)
    total = max(pred_bool.sum() + gt_bool.sum(), 1)

    # For each pred component, find which gt components it overlaps
    h_gt_pred = 0.0
    for i in range(1, n_pred + 1):
        mask_i = (cc_pred == i)
        size_i = mask_i.sum()
        if size_i == 0:
            continue
        gt_in_i = cc_gt[mask_i]
        gt_in_i = gt_in_i[gt_in_i > 0]
        if len(gt_in_i) == 0:
            continue
        counts = np.bincount(gt_in_i)
        probs = counts[counts > 0] / len(gt_in_i)
        h_gt_pred -= (size_i / total) * np.sum(probs * np.log2(probs + 1e-10))

    h_pred_gt = 0.0
    for j in range(1, n_gt + 1):
        mask_j = (cc_gt == j)
        size_j = mask_j.sum()
        if size_j == 0:
            continue
        pred_in_j = cc_pred[mask_j]
        pred_in_j = pred_in_j[pred_in_j > 0]
        if len(pred_in_j) == 0:
            continue
        counts = np.bincount(pred_in_j)
        probs = counts[counts > 0] / len(pred_in_j)
        h_pred_gt -= (size_j / total) * np.sum(probs * np.log2(probs + 1e-10))

    voi = h_gt_pred + h_pred_gt
    # Normalize to [0, 1] score (lower VOI is better)
    max_voi = np.log2(max(n_pred, 1)) + np.log2(max(n_gt, 1))
    if max_voi < 1e-8:
        return 1.0
    voi_score = max(0.0, 1.0 - voi / max(max_voi, 1e-8))
    return float(voi_score)


def topo_proxy(pred, gt):
    """Approximate TopoScore from component and hole counting."""
    struct26 = generate_binary_structure(3, 3)
    struct2d = generate_binary_structure(2, 1)

    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    # Component count similarity
    _, n_pred = cc_label(pred_bool, structure=struct26)
    _, n_gt = cc_label(gt_bool, structure=struct26)
    comp_score = max(0, 1.0 - abs(n_pred - n_gt) / max(n_gt, 3))

    # Hole counting (per-slice)
    n_holes_pred = 0
    n_holes_gt = 0
    for z in range(min(pred.shape[0], 200)):  # cap for speed
        bg_p = ~pred_bool[z]
        lbl_p, n_p = cc_label(bg_p, structure=struct2d)
        border = set()
        border.update(lbl_p[0, :].tolist()); border.update(lbl_p[-1, :].tolist())
        border.update(lbl_p[:, 0].tolist()); border.update(lbl_p[:, -1].tolist())
        n_holes_pred += sum(1 for k in range(1, n_p + 1) if k not in border)

        bg_g = ~gt_bool[z]
        lbl_g, n_g = cc_label(bg_g, structure=struct2d)
        border_g = set()
        border_g.update(lbl_g[0, :].tolist()); border_g.update(lbl_g[-1, :].tolist())
        border_g.update(lbl_g[:, 0].tolist()); border_g.update(lbl_g[:, -1].tolist())
        n_holes_gt += sum(1 for k in range(1, n_g + 1) if k not in border_g)

    hole_score = max(0, 1.0 - abs(n_holes_pred - n_holes_gt) / max(n_holes_gt, 10))

    # Playbook: k=1 proxy (cyclomatic number) — cycle/handle detection
    cycle_score = 1.0
    try:
        from skimage.morphology import skeletonize_3d
        # Subsample for speed if volume is large
        step = max(1, min(pred_bool.shape) // 128)
        pb_sub = pred_bool[::step, ::step, ::step] if step > 1 else pred_bool
        gb_sub = gt_bool[::step, ::step, ::step] if step > 1 else gt_bool

        def _cyclomatic(mask_3d):
            skel = skeletonize_3d(mask_3d.astype(np.uint8)) > 0
            if not skel.any():
                return 0
            n_v = int(skel.sum())
            # Count 26-connected edges via shift overlap
            n_e = 0
            for dz in range(-1, 2):
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dz == 0 and dy == 0 and dx == 0:
                            continue
                        shifted = ndi.shift(skel.astype(np.uint8), (dz, dy, dx), order=0)
                        n_e += int(np.sum(skel & (shifted > 0)))
            n_e //= 2
            _, n_cc_s = cc_label(skel, structure=struct26)
            return max(0, n_e - n_v + n_cc_s)

        beta_1_pred = _cyclomatic(pb_sub)
        beta_1_gt = _cyclomatic(gb_sub)
        cycle_score = max(0, 1.0 - abs(beta_1_pred - beta_1_gt) / max(beta_1_gt, 5))
    except ImportError:
        pass  # skimage not available on this kernel
    except Exception:
        pass

    return 0.35 * comp_score + 0.35 * hole_score + 0.30 * cycle_score


@torch.no_grad()
def mini_leaderboard_val(model_eval, n_vols=2):
    """v3: Full sliding-window val with metric-aligned scoring.
    Returns (composite_score, breakdown_dict)."""
    if len(val_vols) == 0:
        return 0.0, {}
    model_eval.eval()
    was_ds = model_eval.deep_supervision
    model_eval.deep_supervision = False

    # Free training-loop GPU memory before inference.
    # CRITICAL: cudnn.benchmark=True triggers algorithm search on first new shape.
    # Training uses 128³ patches; validation with 192³ would trigger benchmark for
    # a new shape, trying FFT/Winograd algorithms needing 30+ GB → OOM.
    # Fix: always use a safe 128³ cap for validation (same shape cuDNN already cached).
    torch.cuda.empty_cache()
    _VAL_PATCH = (min(PATCH_SIZE[0], 128), min(PATCH_SIZE[1], 128), min(PATCH_SIZE[2], 128))

    struct26 = generate_binary_structure(3, 3)
    amp_on = (DEVICE.type == "cuda")
    n_vols = min(n_vols, len(val_vols))

    all_scores = []
    for vi in range(n_vols):
        info = val_vols[vi]
        img = normalize_volume(read_tif(info["image"]))
        lbl = read_tif(info["label"]).astype(np.uint8)
        lbl[lbl == 2] = IGNORE_LABEL

        pD, pH, pW = _VAL_PATCH
        D, H, W = img.shape

        # Cap validation volume size — fold 2 volumes can be 2000^3, causing 7h validation.
        # Centre-crop to VAL_MAX_CROP so inference stays ~1-2 min regardless of fold.
        # 512→256: 8x fewer patches (250 patches @ 3.5s → ~27 patches @ 3.5s = ~1.5 min vs 15 min)
        VAL_MAX_CROP = 256
        if D > VAL_MAX_CROP or H > VAL_MAX_CROP or W > VAL_MAX_CROP:
            z0c = max(0, (D - VAL_MAX_CROP) // 2)
            y0c = max(0, (H - VAL_MAX_CROP) // 2)
            x0c = max(0, (W - VAL_MAX_CROP) // 2)
            img = img[z0c:z0c+VAL_MAX_CROP, y0c:y0c+VAL_MAX_CROP, x0c:x0c+VAL_MAX_CROP]
            lbl = lbl[z0c:z0c+VAL_MAX_CROP, y0c:y0c+VAL_MAX_CROP, x0c:x0c+VAL_MAX_CROP]
            D, H, W = img.shape

        roi = (min(pD, D), min(pH, H), min(pW, W))

        # Sliding window inference
        rD, rH, rW = roi
        overlap = VAL_OVERLAP
        sD = max(1, int(rD * (1 - overlap)))
        sH = max(1, int(rH * (1 - overlap)))
        sW = max(1, int(rW * (1 - overlap)))

        acc = np.zeros((D, H, W), dtype=np.float32)
        wacc = np.zeros((D, H, W), dtype=np.float32)

        def _g1d(n):
            if n <= 1: return np.ones(n, dtype=np.float32)
            x = np.linspace(-1, 1, n, dtype=np.float32)
            return np.exp(-2 * x * x)
        w3d = _g1d(rD)[:, None, None] * _g1d(rH)[None, :, None] * _g1d(rW)[None, None, :]
        w3d /= w3d.max() + 1e-8

        z_starts = sorted(set(list(range(0, max(1, D-rD+1), sD)) + [max(0, D-rD)]))
        y_starts = sorted(set(list(range(0, max(1, H-rH+1), sH)) + [max(0, H-rH)]))
        x_starts = sorted(set(list(range(0, max(1, W-rW+1), sW)) + [max(0, W-rW)]))

        for z0 in z_starts:
            for y0 in y_starts:
                for x0 in x_starts:
                    patch = img[z0:z0+rD, y0:y0+rH, x0:x0+rW]
                    if patch.shape != (rD, rH, rW):
                        continue
                    t = torch.from_numpy(patch[None, None].astype(np.float32)).to(DEVICE)
                    with torch.amp.autocast("cuda", enabled=amp_on):
                        logits = model_eval(t)
                        if isinstance(logits, (list, tuple)):
                            logits = logits[0] if not isinstance(logits[0], (list, tuple)) else logits[0][0]
                    probs = torch.softmax(logits[0].float(), dim=0)
                    fg_p = probs[1].cpu().numpy()
                    acc[z0:z0+rD, y0:y0+rH, x0:x0+rW] += fg_p * w3d
                    wacc[z0:z0+rD, y0:y0+rH, x0:x0+rW] += w3d

        wacc = np.maximum(wacc, 1e-8)
        fg_prob = acc / wacc

        valid = (lbl != IGNORE_LABEL)
        gt = (lbl == 1) & valid

        # Skip all-ignore or no-FG volumes (would give misleading scores)
        if valid.sum() == 0 or gt.sum() == 0:
            print(f"    Skipping {info['id']}: all-ignore or no FG in GT")
            continue

        # Sweep 3 thresholds — use best composite for checkpoint selection.
        # Fixed 0.5 threshold can mis-rank checkpoints if model calibration is offset.
        _best_composite = 0.0
        _best_breakdown = {}
        for _t in [0.40, 0.50, 0.60]:
            _pred_t = (fg_prob > _t).astype(np.uint8) & valid
            _sd_t = surface_dice_proxy(_pred_t, gt)
            _vo_t = voi_proxy(_pred_t, gt)
            _tp_t = topo_proxy(_pred_t, gt)
            _c = 0.30 * _tp_t + 0.35 * _sd_t + 0.35 * _vo_t
            if _c > _best_composite:
                _best_composite = _c
                _best_breakdown = {"surfdice": _sd_t, "voi": _vo_t, "topo": _tp_t,
                                   "composite": _c, "best_thresh": _t}
        sd = _best_breakdown.get("surfdice", 0.0)
        vo = _best_breakdown.get("voi", 0.0)
        tp = _best_breakdown.get("topo", 0.0)
        composite = _best_composite
        pred = (fg_prob > _best_breakdown.get("best_thresh", 0.50)).astype(np.uint8) & valid

        # Playbook: Threshold stability width — P1 systems have wide plateaus
        stability_width = 0.0
        try:
            # 5 thresholds (was 11) — stability width is a diagnostic, not a gating metric.
            # Reduces per-validation time from ~60s to ~30s on T4.
            thresholds = np.linspace(0.35, 0.65, 5)
            scores_at_t = []
            for t_test in thresholds:
                pred_t = (fg_prob > t_test).astype(np.uint8) & valid
                # Use only surfdice for stability (fastest proxy, most stable signal)
                sd_t = surface_dice_proxy(pred_t, gt)
                scores_at_t.append(sd_t)
            scores_arr = np.array(scores_at_t)
            max_score = scores_arr.max()
            if max_score > 0.01:
                stability_width = float(np.mean(scores_arr >= 0.95 * max_score))
        except Exception:
            pass

        all_scores.append({"surfdice": sd, "voi": vo, "topo": tp, "composite": composite, "stability_width": stability_width})

    model_eval.deep_supervision = was_ds
    model_eval.train()

    if not all_scores:
        return 0.0, {}

    avg_composite = float(np.mean([s["composite"] for s in all_scores]))
    avg_breakdown = {k: float(np.mean([s[k] for s in all_scores])) for k in all_scores[0]}
    return avg_composite, avg_breakdown


def sanity_probe(dataset, n_samples=5):
    """Pre-training sanity check."""
    print(f"\n{'='*60}")
    print(f"SANITY PROBE: Checking {n_samples} samples")
    print(f"{'='*60}")

    fg_fracs = []
    ign_fracs = []
    img_ranges = []

    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]
        img, lbl, med = sample[0], sample[1], sample[2]
        sdf_gt, topo_gt, gap_gt = sample[3], sample[4], sample[5]
        fg_frac = (lbl == 1).sum().item() / max(lbl.numel(), 1)
        ign_frac = (lbl == IGNORE_LABEL).sum().item() / max(lbl.numel(), 1)
        fg_fracs.append(fg_frac)
        ign_fracs.append(ign_frac)
        img_ranges.append((img.min().item(), img.max().item(), img.std().item()))

        print(f"  [{i}] img: shape={tuple(img.shape)}, "
              f"range=[{img.min():.3f}, {img.max():.3f}], std={img.std():.3f}")
        print(f"       lbl: FG={fg_frac*100:.1f}%, IGN={ign_frac*100:.1f}%, "
              f"medial={med.sum():.0f} vox, sdf_range=[{sdf_gt.min():.1f},{sdf_gt.max():.1f}]")

    assert any(f > 0 for f in fg_fracs), "FAIL: All samples have 0% FG!"
    n_usable = sum(1 for f in ign_fracs if f < 0.99)
    if n_usable < len(ign_fracs):
        print(f"  WARNING: {len(ign_fracs) - n_usable}/{len(ign_fracs)} samples have >99% IGNORE (normal for some volumes)")
    assert n_usable >= 1, "FAIL: ALL samples are >99% IGNORE — check data paths!"
    assert sum(1 for s in img_ranges if s[2] > 0.01) >= 1, "FAIL: All image stds too low!"

    print(f"\n  FG fraction: mean={np.mean(fg_fracs)*100:.1f}%, "
          f"range=[{np.min(fg_fracs)*100:.1f}%, {np.max(fg_fracs)*100:.1f}%]")
    print(f"  All sanity checks passed!")
    print(f"{'='*60}\n")


def train_model():
    global _OOM_SHRINK, _OOM_THIS_EPOCH, _LOSS_PHASE, _TOPO_RAMP, _DS_SCALE
    # Ensure globals exist if CELL_ARCHITECTURE ran in a prior session
    global _N_GPUS, _model_for_fwd, DATALOADER_BS
    if '_model_for_fwd' not in globals():
        _model_for_fwd = model   # safety: always single-GPU
    if 'DATALOADER_BS' not in globals():
        DATALOADER_BS = BATCH_SIZE
    if '_N_GPUS' not in globals():
        _N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1
    # Single-GPU always: DataParallel permanently disabled (illegal memory access)
    _effective_grad_accum = GRAD_ACCUM
    best_loss = float("inf")
    best_val_score = -1.0
    # Upgrade 6: Track individual metric bests separately
    best_surfdice_score = -1.0
    best_voi_score = -1.0
    best_topo_score = -1.0
    best_score_path = os.path.join(CKPT_DIR, f"{MODEL_NAME}_best_score.pt")
    best_loss_path = os.path.join(CKPT_DIR, f"{MODEL_NAME}_best_loss.pt")
    best_surfdice_path = os.path.join(CKPT_DIR, f"{MODEL_NAME}_best_surfdice.pt")
    best_voi_path = os.path.join(CKPT_DIR, f"{MODEL_NAME}_best_voi.pt")
    best_topo_path = os.path.join(CKPT_DIR, f"{MODEL_NAME}_best_topo.pt")
    meta_path = os.path.join(CKPT_DIR, f"{MODEL_NAME}_meta.json")
    # Blueprint: 4 late snapshots spaced apart — point-in-time diversity for logit ensemble
    _late_snap_saved = set()  # track which fractions have been saved

    optimizer = torch.optim.SGD(
        model.parameters(), lr=INITIAL_LR,
        momentum=MOMENTUM, nesterov=True, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    ema = EMA(model, decay=EMA_DECAY)

    # v3: SWA model (initialized later when SWA starts)
    swa_model = None
    swa_n = 0

    dataset = VesuviusPatchDataset(
        train_vols, patch_size=PATCH_SIZE,
        num_iters=ITERS_PER_EPOCH, augment=True
    )
    # num_workers=1: 2 workers OOM'd at late phase (t=57%) when K jumped 6→16.
    # CoW fork: each worker copies parent process RAM (14.5GB at ep15) + loads 16 large vols.
    # 2 × 14.5GB + volume cache exceeded 33.7GB → Linux OOM killer SIGKILL'd worker pid.
    # 1 worker halves CoW amplification; K=8 (n_epoch_vols=8) halves volume pool size.
    # pin_memory=True: measured H2D pinned=12.4 GB/s vs pageable=6.2 GB/s (2× faster transfer).
    _n_workers = 1 if torch.cuda.is_available() else 0  # was 2; OOM: K=16 late phase + 2 workers × CoW of 14.5GB parent = SIGKILL
    dataloader = DataLoader(
        dataset, batch_size=DATALOADER_BS, shuffle=False,
        num_workers=_n_workers, pin_memory=True, drop_last=True,
        persistent_workers=False, worker_init_fn=worker_init_fn,
    )

    sanity_probe(dataset, n_samples=2)   # was 5 — 5×50s=250s wasted; 2 is enough to verify data pipeline

    # Manifest
    manifest = {
        "version": "v3.0",
        "model": MODEL_NAME,
        "model_role": MODEL_ROLE,
        "seed": SEED, "fold": FOLD_IDX, "num_folds": NUM_FOLDS,
        "patch_size": list(PATCH_SIZE), "patch_size_early": list(PATCH_SIZE_EARLY),
        "features": list(FEATURES), "blocks": list(BLOCKS),
        "initial_lr": INITIAL_LR, "grad_clip": GRAD_CLIP,
        "grad_accum": GRAD_ACCUM, "ema_decay": EMA_DECAY,
        "label_smooth": LABEL_SMOOTH,
        "warmup_epochs": WARMUP_EPOCHS, "lr_floor": LR_FLOOR,
        "epochs_budget": EPOCHS_BUDGET, "max_train_hours": MAX_TRAIN_HOURS,
        "w_ce": W_CE, "w_dice": W_DICE, "w_msr": W_MSR, "w_bnd": W_BND, "w_topo": W_TOPO,
        "w_surfdist": W_SURFDIST, "w_sdf": W_SDF, "w_cldice": W_CLDICE,
        "w_compreg": W_COMPREG, "w_topoguide": W_TOPOGUIDE,
        "ring_width": RING_WIDTH, "ignore_reject": IGNORE_REJECT,
        "p_boundary": P_BOUNDARY, "p_ring": P_RING, "p_fg": P_FG,
        "p_bg": P_BG, "p_hard": P_HARD, "p_bridge_risk": P_BRIDGE_RISK, "p_thin_risk": P_THIN_RISK, "p_curvature": P_CURVATURE,
        "phase_early_end": PHASE_EARLY_END, "phase_mid_end": PHASE_MID_END,
        "swa_start_frac": SWA_START_FRAC,
        "normalization": "zscore",
        "label_mapping": {"bg": 0, "fg": 1, "ignore": 255, "remap_2_to_ignore": True},
        "postproc_defaults": {
            "dust_min_3d": 192, "hole_max_2d": 32,
            "open_r_xy": 1, "open_r_3d": 1, "bridge_kill": True,
        },
        "n_train": len(train_vols), "n_val": len(val_vols),
        "arch_hash": MODEL_HASH,
        "pytorch_version": torch.__version__, "device": str(DEVICE),
    }
    manifest_path = os.path.join(CKPT_DIR, f"{MODEL_NAME}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[MANIFEST] Saved: {manifest_path}")

    print(f"\n{'='*60}")
    print(f"Training {MODEL_NAME} v3: up to {EPOCHS_BUDGET} epochs, {ITERS_PER_EPOCH} iters/epoch")
    print(f"3-phase schedule: early<{PHASE_EARLY_END}, mid<{PHASE_MID_END}, late+SWA@{SWA_START_FRAC}")
    print(f"{'='*60}")

    meta = {"model": MODEL_NAME, "seed": SEED, "fold": FOLD_IDX,
            "arch_hash": MODEL_HASH, "patch_size": list(PATCH_SIZE)}
    epoch = 0
    amp_enabled = (DEVICE.type == "cuda")
    loss_history = []
    prev_phase = "early"
    val_scores_history = []  # Playbook: Track scores for tail monitoring
    tail_patience = 0
    TAIL_PATIENCE_MAX = 5  # Stop if tail degrades 5x (VAL_EVERY_LATE=2 → 10 epochs grace; 3 caused premature stops with 2-vol val variance)

    for epoch in range(EPOCHS_BUDGET):
        if not budget_ok(MAX_TRAIN_HOURS):
            print(f"\n[TIME] Budget exhausted at epoch {epoch}. Stopping.")
            break

        time_frac = min(1.0, elapsed_h() / MAX_TRAIN_HOURS)

        # v3: Phase-aware LR
        lr = phase_aware_lr(epoch, INITIAL_LR, time_frac)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Phase control
        prev_k = dataset._current_k
        prev_ps = dataset.ps
        prev_phase = _LOSS_PHASE
        dataset.set_epoch(epoch, elapsed_h())
        cur_k = dataset._current_k
        cur_ps = dataset.ps

        # Record phase transitions
        if _LOSS_PHASE != prev_phase:
            if "phase_changes" not in meta:
                meta["phase_changes"] = []
            meta["phase_changes"].append({
                "epoch": epoch, "from": prev_phase, "to": _LOSS_PHASE,
                "patch_size": list(cur_ps), "elapsed_h": round(elapsed_h(), 2),
                "lr_at_transition": lr,
            })

        # Record K changes
        if cur_k != prev_k:
            if "k_changes" not in meta:
                meta["k_changes"] = []
            meta["k_changes"].append({
                "epoch": epoch, "k": cur_k, "elapsed_h": round(elapsed_h(), 2)
            })

        if epoch == 0 or cur_k != prev_k or cur_ps != prev_ps or _LOSS_PHASE != prev_phase:
            pD, pH, pW = cur_ps
            oom_tag = f" OOM_LVL={_OOM_SHRINK}" if _OOM_SHRINK > 0 else ""
            print(f"  [PHASE={_LOSS_PHASE}] patch={pD}x{pH}x{pW}, K={cur_k}, "
                  f"lr={lr:.6f}, clip={GRAD_CLIP}{oom_tag}", flush=True)

        # v3: SWA initialization
        # Note: model.state_dict() is always the BASE model (not DP wrapper) — DP stores model in .module
        if time_frac >= SWA_START_FRAC and swa_model is None:
            _base_model = model.module if hasattr(model, 'module') else model
            swa_model = copy.deepcopy({k: v.float().clone() for k, v in _base_model.state_dict().items()})
            swa_n = 1
            print(f"  [SWA] Initialized at epoch {epoch} (time_frac={time_frac:.2f})")

        _model_for_fwd.train()
        ep_loss = 0.0
        ep_comps = {"ce": 0.0, "dice": 0.0, "msr": 0.0, "bnd": 0.0, "topo": 0.0,
                     "surfdist": 0.0, "sdf": 0.0, "cldice": 0.0, "compreg": 0.0, "topoguide": 0.0,
                     "gapneg": 0.0, "tversky": 0.0}
        _ep_fg_fracs = []   # blueprint gate: verify healthy FG mix each epoch
        n_fwd = 0
        n_opt_steps = 0
        nan_count = 0
        _skip_batches = 0   # non-OOM RuntimeError count per epoch
        t_ep = time.time()

        optimizer.zero_grad(set_to_none=True)
        step_i = -1

        for step_i, batch in enumerate(dataloader):
            img_b, lbl_b, med_b, sdf_gt_b, topo_gt_b, gap_b, vol_indices, centers = batch
            img_b = img_b.to(DEVICE, non_blocking=True)
            lbl_b = lbl_b.to(DEVICE, non_blocking=True)
            med_b = med_b.to(DEVICE, non_blocking=True)
            sdf_gt_b = sdf_gt_b.to(DEVICE, non_blocking=True)
            topo_gt_b = topo_gt_b.to(DEVICE, non_blocking=True)
            gap_b = gap_b.to(DEVICE, non_blocking=True)
            _ep_fg_fracs.append(float((lbl_b == 1).float().mean().item()))

            try:
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    result = _model_for_fwd(img_b)
                    if isinstance(result, tuple) and len(result) == 3:
                        outputs, sdf_pred, topo_pred = result
                    else:
                        outputs = result if isinstance(result, list) else [result]
                        sdf_pred, topo_pred = None, None
                    if not isinstance(outputs, (list, tuple)):
                        outputs = [outputs]
                    loss, loss_comps = compute_loss(
                        outputs, sdf_pred, topo_pred, lbl_b, med_b,
                        sdf_gt_b, topo_gt_b, gap_mask=gap_b, step_i=step_i
                    )
                    loss_scaled = loss / _effective_grad_accum

                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    nan_count += 1
                    if nan_count % 3 == 0:
                        lr = max(lr * 0.5, LR_FLOOR)
                        for pg in optimizer.param_groups:
                            pg["lr"] = lr
                        print(f"  [NaN] {nan_count} NaN losses, LR backed off to {lr:.2e}")
                    continue

                scaler.scale(loss_scaled).backward()

                _has_nan_grad = False
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        _has_nan_grad = True
                        break
                if _has_nan_grad:
                    optimizer.zero_grad(set_to_none=True)
                    nan_count += 1
                    continue

                ep_loss += loss.item()
                n_fwd += 1
                for k in loss_comps:
                    ep_comps[k] += loss_comps[k]

                if (step_i + 1) % _effective_grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(model)
                    n_opt_steps += 1

                # v3 FIX: Hard mining with ACTUAL volume index and center coords
                if n_fwd > 10 and loss.item() > ep_loss / n_fwd * 1.5:
                    for bi in range(img_b.shape[0]):
                        actual_vi = int(vol_indices[bi].item()) if torch.is_tensor(vol_indices[bi]) else int(vol_indices[bi])
                        cz, cy, cx = int(centers[0][bi]), int(centers[1][bi]), int(centers[2][bi])
                        dataset.add_hard_coord(actual_vi, cz, cy, cx)

                if (step_i + 1) % 50 == 0:
                    avg_so_far = ep_loss / max(n_fwd, 1)
                    el_ep = time.time() - t_ep
                    eta_ep = el_ep / (step_i + 1) * (ITERS_PER_EPOCH - step_i - 1)
                    print(f"    [{step_i+1}/{ITERS_PER_EPOCH}] loss={avg_so_far:.4f} | "
                          f"{el_ep:.0f}s elapsed | ETA {eta_ep:.0f}s", flush=True)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    _OOM_THIS_EPOCH += 1
                    gc.collect()
                    torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    if _OOM_THIS_EPOCH >= 3 and _OOM_SHRINK < 3:
                        _OOM_SHRINK += 1
                        print(f"  [OOM-DEGRADE] Level {_OOM_SHRINK}")
                    continue
                # Non-OOM RuntimeError (CUDA assertion, autocast issue, etc.):
                # log + skip this batch instead of crashing the entire training run.
                _skip_batches += 1
                _err_str = str(e)[:200].replace('\n', ' ')
                print(f"  [SKIP] RuntimeError step {step_i} (#{_skip_batches}): {_err_str}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if _skip_batches >= 15:
                    print(f"  [ABORT-EPOCH] {_skip_batches} consecutive batch errors — aborting epoch early")
                    break
                continue

        # Handle remaining accumulated gradients
        if step_i >= 0 and (step_i + 1) % _effective_grad_accum != 0 and n_fwd > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            n_opt_steps += 1

        avg_loss = ep_loss / max(n_fwd, 1)
        dt = time.time() - t_ep
        remain_h = MAX_TRAIN_HOURS - elapsed_h()

        # OOM recovery: one clean epoch → fully reset (not -=1 which left model at 80³ for 3 epochs)
        if _OOM_THIS_EPOCH == 0 and _OOM_SHRINK > 0:
            _OOM_SHRINK = 0
        _OOM_THIS_EPOCH = 0

        # Catastrophic NaN cascade recovery: if >= 20 NaN losses in one epoch,
        # the model weights may have diverged. Reload best checkpoint + cut LR.
        if nan_count >= 20 and os.path.exists(best_score_path):
            print(f"  [NaN-CASCADE] {nan_count} NaN losses in epoch {epoch} — reloading best ckpt")
            try:
                _nc_sd = torch.load(best_score_path, map_location=DEVICE, weights_only=True)
                _base_nc = model.module if hasattr(model, 'module') else model
                _base_nc.load_state_dict(_nc_sd, strict=False)
                ema.shadow = {k: v.clone() for k, v in _base_nc.state_dict().items()}
                del _nc_sd
                lr = max(lr * 0.25, LR_FLOOR)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                print(f"  [NaN-CASCADE] Recovery done — LR reset to {lr:.2e}")
            except Exception as _nce:
                print(f"  [NaN-CASCADE] Recovery failed: {_nce}")

        # v3: SWA weight averaging (always use base model, not DP wrapper)
        if swa_model is not None and n_opt_steps > 0:
            swa_n += 1
            _base_m = model.module if hasattr(model, 'module') else model
            with torch.no_grad():
                for k, v in _base_m.state_dict().items():
                    if k in swa_model:
                        swa_model[k].mul_((swa_n - 1) / swa_n).add_(v.float(), alpha=1.0 / swa_n)

        # Loss logging
        comp_str = " ".join(f"{k}={v/max(n_fwd,1):.4f}" for k, v in ep_comps.items() if v > 0)
        loss_history.append({"epoch": epoch, "loss": avg_loss, "phase": _LOSS_PHASE,
                            "patch_size": list(dataset.ps), "lr": lr,
                            **{k: v/max(n_fwd,1) for k, v in ep_comps.items()}})

        if avg_loss < best_loss:
            best_loss = avg_loss
            ema.save(best_loss_path)
            meta["best_loss_epoch"] = epoch
            meta["best_loss"] = best_loss

        # v3: Validation frequency depends on phase
        val_every = VAL_EVERY_LATE if _LOSS_PHASE == "late" else VAL_EVERY
        val_str = ""
        # No epoch-0 validation: model has no useful weights yet and GPU memory is
        # packed with optimizer states. First val fires at epoch (val_every - 1).
        _do_val = ((epoch + 1) % val_every == 0) and budget_ok(MAX_TRAIN_HOURS - 0.3)
        if _do_val:
            saved_state = {k: v.clone() for k, v in model.state_dict().items()}
            ema.apply(model)
            score, breakdown = mini_leaderboard_val(model)
            model.load_state_dict(saved_state)
            model.train()   # mini_leaderboard_val leaves model in eval() — restore

            val_str = f" | score={score:.4f}"
            if breakdown:
                val_str += f" (sd={breakdown.get('surfdice',0):.3f} voi={breakdown.get('voi',0):.3f} tp={breakdown.get('topo',0):.3f} stab={breakdown.get('stability_width',0):.2f})"
            if score > best_val_score:
                best_val_score = score
                ema.save(best_score_path)
                meta["best_val_epoch"] = epoch
                meta["best_val_score"] = best_val_score
                meta["best_val_breakdown"] = breakdown
                val_str += " *best*"

            # Upgrade 6: Save per-metric best checkpoints
            if breakdown:
                _sd = breakdown.get("surfdice", 0.0)
                _vo = breakdown.get("voi", 0.0)
                _tp = breakdown.get("topo", 0.0)
                if _sd > best_surfdice_score:
                    best_surfdice_score = _sd
                    ema.save(best_surfdice_path)
                    meta["best_surfdice_epoch"] = epoch
                    meta["best_surfdice"] = best_surfdice_score
                if _vo > best_voi_score:
                    best_voi_score = _vo
                    ema.save(best_voi_path)
                    meta["best_voi_epoch"] = epoch
                    meta["best_voi"] = best_voi_score
                if _tp > best_topo_score:
                    best_topo_score = _tp
                    ema.save(best_topo_path)
                    meta["best_topo_epoch"] = epoch
                    meta["best_topo"] = best_topo_score

            # Playbook: Tail-aware early stopping — stop when score declining in late phase
            if breakdown:
                val_scores_history.append({"epoch": epoch, "score": score})
                if len(val_scores_history) >= 3 and _LOSS_PHASE == "late":
                    if score < best_val_score * 0.95:
                        tail_patience += 1
                        if tail_patience >= TAIL_PATIENCE_MAX:
                            print(f"\n  [TAIL-STOP] Score declining in late phase ({tail_patience}x). "
                                  f"Current={score:.4f}, Best={best_val_score:.4f}. Stopping early.")
                            break
                    else:
                        tail_patience = 0

        if epoch % 5 == 0:
            log_resources(f"ep{epoch}")

        _sps = n_opt_steps / max(dt, 1e-8)  # effective optimizer steps/sec
        _gpu_tag = f" | GPU0={_sps:.2f}sps"
        # Blueprint gate: verify batch FG% is healthy (not 80%+ empty background)
        _avg_fg_pct = float(np.mean(_ep_fg_fracs) * 100) if _ep_fg_fracs else 0.0
        _fg_tag = f" | fg={_avg_fg_pct:.1f}%"
        if _avg_fg_pct < 3.0:
            _fg_tag += " [WARN:LOW-FG — sampler may be pulling empty patches]"
        elif _avg_fg_pct > 70.0:
            _fg_tag += " [WARN:HIGH-FG — check label balance]"
        print(f"  Ep {epoch:3d}/{EPOCHS_BUDGET} | loss={avg_loss:.4f} | {comp_str} | "
              f"lr={lr:.6f} | phase={_LOSS_PHASE} | {dt:.0f}s{_gpu_tag}{_fg_tag} | rem={remain_h:.2f}h{val_str}")

        meta["last_epoch"] = epoch
        meta["last_loss"] = avg_loss
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Periodic safety checkpoint every 5 epochs — ensures a recent recovery point always exists.
        # If training crashes after epoch N, the last periodic save is at most 5 epochs behind.
        if epoch % 5 == 4 and n_fwd > 0:   # epochs 4, 9, 14, 19, ...
            _periodic_path = os.path.join(CKPT_DIR, f"{MODEL_NAME}_periodic.pt")
            try:
                ema.save(_periodic_path)
                print(f"  [PERIODIC-CKPT] ep{epoch}: saved recovery checkpoint")
            except Exception as _pce:
                print(f"  [PERIODIC-CKPT] Save failed: {_pce}")

        # Proactive GC every 10 epochs to slow CoW-induced RAM creep (~0.6GB/epoch without it).
        # Workers fork parent RAM at epoch start; releasing Python objects reduces fork footprint.
        if epoch % 10 == 9:
            gc.collect()

        # Blueprint: 4 late snapshots at fixed time fractions (point-in-time ensemble diversity).
        # Each captures a different optimizer state; logit-averaging them is like a mini-ensemble.
        if n_fwd > 0:
            for _sf in [0.60, 0.70, 0.80, 0.90]:
                if time_frac >= _sf and _sf not in _late_snap_saved:
                    _snap_path = os.path.join(CKPT_DIR,
                                              f"{MODEL_NAME}_late{int(_sf*100)}.pt")
                    try:
                        ema.save(_snap_path)
                        _late_snap_saved.add(_sf)
                        print(f"  [SNAP] Late snapshot at {_sf:.0%}: {_snap_path}")
                    except Exception as _se:
                        print(f"  [SNAP] Save failed at {_sf:.0%}: {_se}")

    # Save final checkpoints
    if best_val_score < 0 and os.path.exists(best_loss_path):
        shutil.copy2(best_loss_path, best_score_path)
        meta["best_val_epoch"] = meta.get("best_loss_epoch", 0)
        meta["best_val_score"] = 0.0

    # v3: Save SWA checkpoint if available
    if swa_model is not None:
        swa_path = os.path.join(CKPT_DIR, f"{MODEL_NAME}_swa.pt")
        torch.save(swa_model, swa_path)
        print(f"[SWA] Saved: {swa_path} (averaged {swa_n} checkpoints)")
        # Use SWA if it scores better
        saved_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(swa_model)
        swa_score, swa_bd = mini_leaderboard_val(model)
        model.load_state_dict(saved_state)
        if swa_score > best_val_score:
            shutil.copy2(swa_path, best_score_path)
            meta["best_val_score"] = swa_score
            meta["best_val_breakdown"] = swa_bd
            meta["swa_used"] = True
            print(f"[SWA] SWA model is better: {swa_score:.4f} > {best_val_score:.4f}")
        else:
            meta["swa_used"] = False
            print(f"[SWA] EMA model is better: {best_val_score:.4f} >= {swa_score:.4f}")

    last_path = os.path.join(CKPT_DIR, f"{MODEL_NAME}_last.pt")
    ema.save(last_path)
    meta["total_epochs"] = epoch + 1
    meta["total_time_h"] = elapsed_h()

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    history_path = os.path.join(CKPT_DIR, f"{MODEL_NAME}_history.json")
    with open(history_path, "w") as f:
        json.dump(loss_history, f, indent=2)

    log_resources("final")
    print(f"\n[DONE] {MODEL_NAME}: {epoch+1} epochs in {elapsed_h():.2f}h")
    print(f"  Best val score: {best_val_score:.4f} (epoch {meta.get('best_val_epoch', '?')})")
    print(f"  Best loss: {best_loss:.5f} (epoch {meta.get('best_loss_epoch', '?')})")
    return best_score_path, meta

best_ckpt_path, train_meta = train_model()
'''


CELL_CALIBRATION = r'''
# ============================================================
# Threshold Calibration v3: Score-aligned metric
# Score = 0.30*TopoProxy + 0.35*SurfDiceProxy + 0.35*VOIProxy
# ============================================================

@torch.no_grad()
def sliding_window_inference(vol_f32, model_eval, roi, overlap=0.25):
    """Sliding window inference with Gaussian weighting. Returns softmax probs (C,D,H,W)."""
    D, H, W = vol_f32.shape
    rD, rH, rW = roi
    sD = max(1, int(rD * (1.0 - overlap)))
    sH = max(1, int(rH * (1.0 - overlap)))
    sW = max(1, int(rW * (1.0 - overlap)))

    def _gauss_1d(n):
        if n <= 1: return np.ones(n, dtype=np.float32)
        x = np.linspace(-1, 1, n, dtype=np.float32)
        return np.exp(-2 * x * x)
    w3d = (_gauss_1d(rD)[:, None, None] *
           _gauss_1d(rH)[None, :, None] *
           _gauss_1d(rW)[None, None, :])
    w3d /= w3d.max() + 1e-8

    acc = np.zeros((NUM_CLASSES, D, H, W), dtype=np.float32)
    wacc = np.zeros((D, H, W), dtype=np.float32)

    z_starts = sorted(set(list(range(0, max(1, D-rD+1), sD)) + [max(0, D-rD)]))
    y_starts = sorted(set(list(range(0, max(1, H-rH+1), sH)) + [max(0, H-rH)]))
    x_starts = sorted(set(list(range(0, max(1, W-rW+1), sW)) + [max(0, W-rW)]))

    amp_on = (DEVICE.type == "cuda")
    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                patch = vol_f32[z0:z0+rD, y0:y0+rH, x0:x0+rW]
                if patch.shape != (rD, rH, rW):
                    continue
                t = torch.from_numpy(patch[None, None]).to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=amp_on):
                    logits = model_eval(t)
                    if isinstance(logits, (list, tuple)):
                        logits = logits[0] if not isinstance(logits[0], (list, tuple)) else logits[0][0]
                probs = torch.softmax(logits[0].float(), dim=0).cpu().numpy()
                for c in range(NUM_CLASSES):
                    acc[c, z0:z0+rD, y0:y0+rH, x0:x0+rW] += probs[c] * w3d
                wacc[z0:z0+rD, y0:y0+rH, x0:x0+rW] += w3d

    wacc = np.maximum(wacc, 1e-8)
    for c in range(NUM_CLASSES):
        acc[c] /= wacc
    return acc


def calibrate_temperature(cal_probs, cal_labels):
    """Find temperature that minimizes NLL on calibration data."""
    best_t = 1.0
    best_nll = float('inf')
    for t_cand in np.arange(0.5, 3.05, 0.1):
        nll = 0.0
        n = 0
        for fp, lbl in zip(cal_probs, cal_labels):
            valid = (lbl != IGNORE_LABEL) & (lbl != 2)
            if valid.sum() == 0:
                continue
            fp_clip = np.clip(fp[valid], 1e-7, 1 - 1e-7)
            logit = np.log(fp_clip / (1 - fp_clip)) / t_cand
            p_scaled = 1.0 / (1.0 + np.exp(-logit))
            gt = (lbl[valid] == 1).astype(np.float32)
            p_safe = np.clip(p_scaled, 1e-7, 1 - 1e-7)
            nll += -(gt * np.log(p_safe) + (1 - gt) * np.log(1 - p_safe)).sum()
            n += valid.sum()
        if n > 0:
            avg_nll = nll / n
            if avg_nll < best_nll:
                best_nll = avg_nll
                best_t = float(t_cand)
    return best_t


def score_aligned_metrics(fg_prob, lbl, tl, th):
    """v3: Compute competition-aligned score for threshold calibration.
    Score = 0.30*TopoProxy + 0.35*SurfDiceProxy + 0.35*VOIProxy"""
    struct26 = generate_binary_structure(3, 3)
    gt = (lbl == 1)
    ignore = (lbl == 2) | (lbl == IGNORE_LABEL)

    # Hysteresis thresholding
    strong = fg_prob >= th
    weak = fg_prob >= tl
    cc_w, n_w = cc_label(weak, structure=struct26)
    if n_w == 0:
        pred = np.zeros_like(fg_prob, dtype=bool)
    else:
        strong_ids = np.unique(cc_w[strong])
        strong_ids = strong_ids[strong_ids != 0]
        pred = np.isin(cc_w, strong_ids)

    pred[ignore] = False
    gt_clean = gt.copy()
    gt_clean[ignore] = False

    # Compute proxies
    sd = surface_dice_proxy(pred.astype(np.uint8), gt_clean.astype(np.uint8))
    vo = voi_proxy(pred.astype(np.uint8), gt_clean.astype(np.uint8))
    tp = topo_proxy(pred.astype(np.uint8), gt_clean.astype(np.uint8))

    combined = 0.30 * tp + 0.35 * sd + 0.35 * vo

    # Component counts for logging
    cc_pred, n_pred = cc_label(pred, structure=struct26)
    cc_gt, n_gt = cc_label(gt_clean, structure=struct26)

    return {
        "surfdice": sd, "voi": vo, "topo": tp, "combined": combined,
        "n_pred_cc": int(n_pred), "n_gt_cc": int(n_gt),
    }


def calibrate_threshold():
    """v3: Grid search for best hysteresis thresholds using score-aligned metric."""
    model.eval()
    model.deep_supervision = False
    # Flush training state from GPU before calibration inference.
    # Calibration runs right after train_model() so optimizer+EMA still in VRAM.
    # cuDNN benchmark with 192³ patches requires 30+ GB workspace → OOM.
    # Cap to 128³ (same as validation, already benchmark-cached during training).
    torch.cuda.empty_cache()

    # Blueprint: threshold selection needs ≥6-12 val volumes for generalization.
    # CAL_N=12 (vs training VAL_N=3) runs once at end-of-training — time budget is acceptable.
    # val_vols is pre-shuffled so we get a diverse sample, not just the first 12 alphabetically.
    n_cal = min(12, len(val_vols))
    if n_cal == 0:
        print("[CAL] No val volumes, using defaults")
        return 0.34, 0.62, 0.85, 0.0

    best_score = -1
    best_tl, best_th = 0.34, 0.62

    _CAL_PATCH = (min(PATCH_SIZE[0], 128), min(PATCH_SIZE[1], 128), min(PATCH_SIZE[2], 128))
    # Cap volume size for calibration — same as validation to prevent multi-hour calibration
    # on large volumes (some competition volumes are 2000³+).
    CAL_MAX_CROP = 256
    cal_probs = []
    cal_labels = []
    for vi in range(n_cal):
        info = val_vols[vi]
        img = normalize_volume(read_tif(info["image"]))
        lbl = read_tif(info["label"]).astype(np.uint8)
        D, H, W = img.shape
        if D > CAL_MAX_CROP or H > CAL_MAX_CROP or W > CAL_MAX_CROP:
            z0c = max(0, (D - CAL_MAX_CROP) // 2)
            y0c = max(0, (H - CAL_MAX_CROP) // 2)
            x0c = max(0, (W - CAL_MAX_CROP) // 2)
            img = img[z0c:z0c+CAL_MAX_CROP, y0c:y0c+CAL_MAX_CROP, x0c:x0c+CAL_MAX_CROP]
            lbl = lbl[z0c:z0c+CAL_MAX_CROP, y0c:y0c+CAL_MAX_CROP, x0c:x0c+CAL_MAX_CROP]
        roi = tuple(min(p, s) for p, s in zip(_CAL_PATCH, img.shape))
        probs = sliding_window_inference(img, model, roi, overlap=VAL_OVERLAP)
        fg_prob = probs[1]
        cal_probs.append(fg_prob)
        cal_labels.append(lbl)
        print(f"  [CAL] Volume {info['id']}: prob range [{fg_prob.min():.3f}, {fg_prob.max():.3f}]")

    # Blueprint: variance-penalized threshold selection J = mean - 0.25*std
    # Wider range (tl ∈ [0.22, 0.64], th ∈ [0.44, 0.84]) — step=0.04 for speed,
    # Nelder-Mead refines below to sub-step precision.
    for th in np.arange(0.44, 0.86, 0.04):
        for tl in np.arange(max(0.22, float(th) - 0.36), float(th) - 0.05, 0.04):
            scores = []
            for fp, lbl in zip(cal_probs, cal_labels):
                m = score_aligned_metrics(fp, lbl, float(tl), float(th))
                scores.append(m["combined"])
            J = float(np.mean(scores)) - 0.25 * float(np.std(scores))
            if J > best_score:
                best_score = J
                best_tl, best_th = float(tl), float(th)

    print(f"[CAL] Grid best: tl={best_tl:.3f}, th={best_th:.3f}, J={best_score:.4f}")

    # Blueprint: refine with 2D Nelder-Mead simplex (cal_probs cached — fast)
    try:
        from scipy.optimize import minimize as _scipy_min
        def _cal_obj(params):
            tl_v, th_v = float(params[0]), float(params[1])
            if tl_v >= th_v or tl_v < 0.18 or th_v > 0.92 or th_v < 0.38:
                return 1.0  # infeasible sentinel
            _s = [score_aligned_metrics(fp, lbl, tl_v, th_v)["combined"]
                  for fp, lbl in zip(cal_probs, cal_labels)]
            return -(float(np.mean(_s)) - 0.25 * float(np.std(_s)))
        _res = _scipy_min(_cal_obj, x0=[best_tl, best_th], method='Nelder-Mead',
                          options={'xatol': 0.005, 'fatol': 1e-4, 'maxiter': 60, 'disp': False})
        _tl_r, _th_r = float(_res.x[0]), float(_res.x[1])
        if 0.18 <= _tl_r < _th_r <= 0.92:
            best_tl, best_th = _tl_r, _th_r
            best_score = -_res.fun
            print(f"[CAL] Nelder-Mead: tl={best_tl:.3f}, th={best_th:.3f}, J={best_score:.4f}")
    except Exception as _nm_e:
        print(f"[CAL] Nelder-Mead skipped: {_nm_e}")

    print(f"[CAL] Best thresholds: tl={best_tl:.3f}, th={best_th:.3f}, score={best_score:.4f}")

    temperature = calibrate_temperature(cal_probs, cal_labels)
    print(f"[CAL] Temperature: {temperature:.2f}")

    # Detailed metrics at best thresholds
    print(f"\n[VAL] Score-aligned validation metrics:")
    for fp, lbl, info in zip(cal_probs, cal_labels, val_vols[:n_cal]):
        metrics = score_aligned_metrics(fp, lbl, best_tl, best_th)
        print(f"  {info['id']}: surfdice={metrics['surfdice']:.4f}, voi={metrics['voi']:.3f}, "
              f"topo={metrics['topo']:.3f}, combined={metrics['combined']:.4f}")
        print(f"    Components: pred={metrics['n_pred_cc']}, gt={metrics['n_gt_cc']}")

    # Upgrade 1: Fit learned per-volume threshold predictor
    # Collect histogram features for each validation volume and per-volume best (tl, th)
    thresh_coefs = {}
    try:
        _feat_rows = []
        _tl_targets = []
        _th_targets = []

        for fp_vol, lbl_vol in zip(cal_probs, cal_labels):
            # Find best (tl, th) for this individual volume
            _best_v_score = -1.0
            _best_v_tl, _best_v_th = best_tl, best_th
            for _th_v in np.arange(0.44, 0.88, 0.04):   # widened to match outer Nelder-Mead range
                for _tl_v in np.arange(max(0.22, float(_th_v) - 0.36), float(_th_v) - 0.05, 0.04):
                    _m = score_aligned_metrics(fp_vol, lbl_vol, float(_tl_v), float(_th_v))
                    if _m["combined"] > _best_v_score:
                        _best_v_score = _m["combined"]
                        _best_v_tl = float(_tl_v)
                        _best_v_th = float(_th_v)
            _tl_targets.append(_best_v_tl)
            _th_targets.append(_best_v_th)

            # Extract features: [mean, std, p10, p25, p50, p75, p90, fg01, fg02, fg03, fg05]
            _p_flat = fp_vol.ravel()
            _pcts = np.percentile(_p_flat, [10, 25, 50, 75, 90])
            _feats = [
                float(_p_flat.mean()), float(_p_flat.std()),
                float(_pcts[0]), float(_pcts[1]), float(_pcts[2]),
                float(_pcts[3]), float(_pcts[4]),
                float((_p_flat > 0.1).mean()), float((_p_flat > 0.2).mean()),
                float((_p_flat > 0.3).mean()), float((_p_flat > 0.5).mean()),
            ]
            _feat_rows.append(_feats)

        if len(_feat_rows) >= 2:
            # Simple ridge regression (closed-form, no sklearn needed)
            _X = np.array(_feat_rows, dtype=np.float64)   # (N, 11)
            _X_centered = _X - _X.mean(axis=0)
            _lam = 1e-3  # ridge regularization
            _I = np.eye(_X_centered.shape[1])
            # Ridge: coefs = (X'X + λI)^-1 X'y
            _XtX = _X_centered.T @ _X_centered
            _XtX_reg = _XtX + _lam * _I
            _tl_arr = np.array(_tl_targets) - np.mean(_tl_targets)
            _th_arr = np.array(_th_targets) - np.mean(_th_targets)
            _tl_coefs = np.linalg.solve(_XtX_reg, _X_centered.T @ _tl_arr)
            _th_coefs = np.linalg.solve(_XtX_reg, _X_centered.T @ _th_arr)
            thresh_coefs = {
                "tl_coefs": _tl_coefs.tolist(),
                "th_coefs": _th_coefs.tolist(),
                "tl_intercept": float(np.mean(_tl_targets)),
                "th_intercept": float(np.mean(_th_targets)),
                "feat_mean": _X.mean(axis=0).tolist(),
            }
            print(f"[CAL] Learned threshold predictor fitted on {len(_feat_rows)} volumes")
            print(f"  tl mean={thresh_coefs['tl_intercept']:.3f}, "
                  f"th mean={thresh_coefs['th_intercept']:.3f}")
        else:
            print(f"[CAL] Not enough volumes ({len(_feat_rows)}) to fit threshold predictor")
    except Exception as _e:
        print(f"[CAL] Threshold predictor fitting failed: {_e}")
        thresh_coefs = {}

    model.deep_supervision = True
    return best_tl, best_th, temperature, best_score, thresh_coefs


# Load best checkpoint for calibration (Upgrade 6: try all metric checkpoints)
_ckpt_candidates = [
    (os.path.join(CKPT_DIR, f"{MODEL_NAME}_best_score.pt"), "best_score"),
    (os.path.join(CKPT_DIR, f"{MODEL_NAME}_best_surfdice.pt"), "best_surfdice"),
    (os.path.join(CKPT_DIR, f"{MODEL_NAME}_best_voi.pt"), "best_voi"),
    (os.path.join(CKPT_DIR, f"{MODEL_NAME}_best_topo.pt"), "best_topo"),
    (os.path.join(CKPT_DIR, f"{MODEL_NAME}_best_loss.pt"), "best_loss"),
]
score_path = os.path.join(CKPT_DIR, f"{MODEL_NAME}_best_score.pt")  # kept for compat
_has_checkpoint = False
for _ckpt_path, _ckpt_name in _ckpt_candidates:
    if os.path.exists(_ckpt_path):
        sd = torch.load(_ckpt_path, map_location=DEVICE)
        model.load_state_dict(sd)
        print(f"[CAL] Loaded {_ckpt_name} checkpoint: {_ckpt_path}")
        _has_checkpoint = True
        score_path = _ckpt_path  # use this for copy
        break
if not _has_checkpoint:
    print("[CAL] WARNING: No checkpoint found! Using current model weights.")

if _has_checkpoint and budget_ok(MAX_TRAIN_HOURS + 0.5):
    # Unwrap torch.compile before calibration.
    # torch.compile (default mode) tries to re-trace the model in eval mode.
    # Dynamo's fake-tensor dtype inference can misidentify float32 input as float64,
    # causing: RuntimeError('Input type (double) and bias type (c10::Half) should be same').
    # Using _orig_mod bypasses Dynamo entirely for calibration (eager mode, correct dtypes).
    model = getattr(model, '_orig_mod', model)
    model.eval()
    try:
        cal_tl, cal_th, cal_temp, cal_score, cal_thresh_coefs = calibrate_threshold()
    except Exception as _cal_e:
        print(f"[CAL] WARNING: calibration failed ({type(_cal_e).__name__}: {_cal_e})")
        print("[CAL] Falling back to defaults — meta.json will still be saved.")
        cal_tl, cal_th, cal_temp, cal_score, cal_thresh_coefs = 0.34, 0.62, 0.85, 0.0, {}
else:
    cal_tl, cal_th, cal_temp, cal_score, cal_thresh_coefs = 0.34, 0.62, 0.85, 0.0, {}
    if not _has_checkpoint:
        print("[CAL] Skipped (no checkpoint), using defaults")
    else:
        print("[CAL] Skipped (time budget), using defaults")

# Update meta
meta_path = os.path.join(CKPT_DIR, f"{MODEL_NAME}_meta.json")
if not os.path.exists(meta_path):
    meta = {"model_name": MODEL_NAME, "version": "v3.0"}
    print(f"[CAL] WARNING: No meta file found, creating minimal one")
else:
    with open(meta_path, "r") as f:
        meta = json.load(f)
meta["tl"] = cal_tl
meta["th"] = cal_th
meta["temperature"] = cal_temp
meta["combined_proxy"] = cal_score
meta["model_role"] = MODEL_ROLE
# Upgrade 1: Export learned threshold predictor coefficients
if cal_thresh_coefs:
    meta["thresh_coefs"] = cal_thresh_coefs

# Role-specific postproc defaults
if MODEL_ROLE == "topology":
    meta["dust_min"] = 256
    meta["inf_overlap"] = 0.5
    meta["bk_min_lobe"] = 15000
elif MODEL_ROLE == "surface":
    meta["dust_min"] = 128
    meta["inf_overlap"] = 0.5
    meta["bk_min_lobe"] = 10000
else:  # balanced
    meta["dust_min"] = 192
    meta["inf_overlap"] = 0.5
    meta["bk_min_lobe"] = 10000

# Copy calibrated checkpoint to primary path
primary_path = os.path.join(CKPT_DIR, f"{MODEL_NAME}_best.pt")
if os.path.exists(score_path):
    shutil.copy2(score_path, primary_path)
    print(f"[CAL] Copied calibrated checkpoint -> {primary_path}")

with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n[FINAL] {MODEL_NAME} artifacts in {CKPT_DIR}:")
for fn in sorted(os.listdir(CKPT_DIR)):
    if MODEL_NAME in fn:
        sz = os.path.getsize(os.path.join(CKPT_DIR, fn))
        print(f"  {fn}: {sz/1e6:.1f} MB")
'''


# ============================================================
# PREDICTION NOTEBOOK CELLS
# ============================================================

CELL_PRED_ENV = r'''
# ============================================================
# Environment + Imports (Prediction/Ensemble v3)
# ============================================================
import os, sys, time, gc, warnings, json, zipfile, hashlib, traceback
warnings.filterwarnings("ignore")

# max_split_size_mb:9000 — the cuDNN PRECOMP_GEMM workspace for Conv3d(96ch, 128³) is
# 10.12 GiB. Without this setting, PyTorch's caching allocator splits that block to service
# smaller requests (model weights 191 MB, encoder skip-connection tensors ~700 MB), leaving
# only 9.86 GiB of fragmented free blocks — not enough for the next model's workspace.
# By marking blocks >9 GB as unsplittable, the 10.12 GB block stays intact in the cache
# and is cleanly reused for all 9 models (zero cudaMalloc after the first model).
# Small allocations go to cudaMalloc from the ~4.4 GB of free physical VRAM instead.
# Do NOT use expandable_segments:True — it uses CUDA VMM which corrupts the T4 context.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:9000")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt_fn

import scipy.ndimage as ndi
from scipy.ndimage import (distance_transform_edt, generate_binary_structure,
                           label as cc_label, binary_erosion, binary_dilation,
                           binary_opening, gaussian_filter)

try:
    import tifffile
    _HAVE_TIFFFILE = True
except ImportError:
    _HAVE_TIFFFILE = False

def _read_tif_pil(path):
    from PIL import Image
    img = Image.open(path)
    frames = []
    try:
        while True:
            frames.append(np.array(img))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return np.stack(frames, axis=0)

def read_tif(path):
    # tifffile preferred; fall back to PIL for LZW-compressed TIFFs.
    # PIL handles LZW natively with no extra packages — works offline.
    if _HAVE_TIFFFILE:
        try:
            return tifffile.imread(path)
        except Exception:
            return _read_tif_pil(path)
    return _read_tif_pil(path)

def write_tif(path, data):
    if _HAVE_TIFFFILE:
        tifffile.imwrite(path, data)
    else:
        from PIL import Image as PILImage
        imgs = [PILImage.fromarray(data[z]) for z in range(data.shape[0])]
        imgs[0].save(path, save_all=True, append_images=imgs[1:])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[ENV] PyTorch {torch.__version__}, Device: {DEVICE}")
if DEVICE.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"[ENV] GPU: {props.name}, VRAM: {props.total_memory / 1e9:.1f} GB")
    # benchmark=False (heuristic, default): cuDNN picks algorithm via lookup table.
    # For Conv3d(96ch, 128³) on T4, heuristic selects PRECOMP_GEMM (10.12 GB workspace).
    # This is fine — max_split_size_mb:9000 keeps the 10.12 GB block intact in cache
    # so the workspace is reused cleanly across all 9 models with no re-allocation.
    # Using benchmark=True would pollute the cache with many small blocks from algo tests.
    torch.backends.cudnn.benchmark = False
    print("[ENV] cuDNN benchmark=False (heuristic algo selection; workspace reused via max_split_size_mb:9000)")

T_START = time.time()
def elapsed_h():
    return (time.time() - T_START) / 3600.0
def budget_ok(max_h=8.5):
    return elapsed_h() < max_h
'''


CELL_PRED_CONFIG = r'''
# ============================================================
# Configuration v3 + Explicit Model Paths
# ============================================================
NUM_CLASSES = 2
IGNORE_LABEL = 255
IN_CHANNELS = 1
FEATURES = [32, 64, 128, 256, 320, 320]
BLOCKS = [1, 3, 4, 6, 6, 6]
STRIDES = [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]]
DEC_CONVS = [1, 1, 1, 1, 1]

ROOT_CANDS = [
    "/kaggle/input/competitions/vesuvius-challenge-surface-detection",
    "/kaggle/input/vesuvius-challenge-surface-detection",
]
ROOT_DIR = next((p for p in ROOT_CANDS if os.path.exists(p)), None)
if ROOT_DIR is None:
    ROOT_DIR = "/kaggle/input/competitions/vesuvius-challenge-surface-detection"
TEST_DIR = os.path.join(ROOT_DIR, "test_images")

MODEL_DATASET_MAP = {
    # Super model X: single unified model; loads all available per-metric checkpoints
    # as separate ensemble variants (best_score/surfdice/voi/topo + swa = up to 5 variants).
    "model_x": {
        "dataset_slugs": ["vesuvius-v3-checkpoints", "vesuvius-model-x", "vesuvius-model-x-ckpt"],
        "ckpt_candidates": ["model_x_best_score.pt", "model_x_best_surfdice.pt",
                            "model_x_best_voi.pt", "model_x_best_topo.pt",
                            "model_x_best_loss.pt",
                            "model_x_late60.pt", "model_x_late70.pt",
                            "model_x_late80.pt", "model_x_late90.pt",
                            "model_x_swa.pt", "model_x_best.pt"],
        "meta": "model_x_meta.json",
    },
}

MODEL_PATHS = {}
META_PATHS = {}

# Multi-checkpoint ensemble: load ALL available per-metric checkpoints as separate ensemble members.
# 1 trained model → up to 4 variants. 3 models → up to 12 variants. Zero extra training cost.
# Variants have different weight optima (topo-specialist ckpt ≠ surfdice-specialist ckpt).
# Weight multipliers downweight metric-specific checkpoints vs. composite-optimal checkpoint.
CKPT_TYPE_WEIGHTS = {
    "best_score":    1.00,  # composite-optimal — full weight
    "best_surfdice": 0.70,  # surfdice-optimised — slight downweight (may sacrifice topo/voi)
    "best_topo":     0.70,  # topo-optimised — slight downweight (may sacrifice sd/voi)
    "best_voi":      0.60,  # voi-optimised — lowest (voi is already strong at epoch 4: 0.866)
    "best_loss":     0.60,  # best training loss — no val filtering, more conservative weight
    "late60":        0.65,  # late snapshot at 60% training time — early late-phase
    "late70":        0.65,  # late snapshot at 70% training time
    "late80":        0.65,  # late snapshot at 80% training time (SWA start)
    "late90":        0.65,  # late snapshot at 90% training time — near end
    "swa":           0.75,  # stochastic weight average — smooth ensemble of late-phase weights
    "best":          0.80,  # legacy fallback key
}
CKPT_MULTS = {}  # compound_key → weight multiplier

# Build dataset search roots dynamically.
# Kaggle mounts user datasets at /kaggle/input/datasets/{username}/ (not /kaggle/input/).
# Also check standard /kaggle/input/ for backwards compat and -ckpt slug fallbacks.
_DATASET_SEARCH_ROOTS = ["/kaggle/input"]
_ds_users_base = "/kaggle/input/datasets"
if os.path.exists(_ds_users_base):
    for _uname in sorted(os.listdir(_ds_users_base)):
        _upath = os.path.join(_ds_users_base, _uname)
        if os.path.isdir(_upath):
            _DATASET_SEARCH_ROOTS.append(_upath)
print(f"[CFG] Dataset search roots: {_DATASET_SEARCH_ROOTS}")

for mname, spec in MODEL_DATASET_MAP.items():
    found_any = False
    for _root in _DATASET_SEARCH_ROOTS:
        for slug in spec["dataset_slugs"]:
            slug_base = os.path.join(_root, slug)
            if not os.path.exists(slug_base):
                continue
            # Handle optional version subdirectory: slug/1/, slug/2/ (Kaggle versioned datasets)
            _ver_dirs = [""]
            for _e in sorted(os.listdir(slug_base), reverse=True):
                if os.path.isdir(os.path.join(slug_base, _e)) and _e.isdigit():
                    _ver_dirs.insert(0, _e)
            for _ver in _ver_dirs:
                ver_base = os.path.join(slug_base, _ver) if _ver else slug_base
                for sub in ["checkpoints", ""]:
                    base = os.path.join(ver_base, sub) if sub else ver_base
                    meta_path = os.path.join(base, spec["meta"])
                    for ckpt_name in spec.get("ckpt_candidates", [f"{mname}_best.pt"]):
                        ckpt = os.path.join(base, ckpt_name)
                        if os.path.exists(ckpt):
                            ckpt_stem = ckpt_name.replace(".pt", "")
                            ckpt_type = ckpt_stem[len(f"{mname}_"):] if ckpt_stem.startswith(f"{mname}_") else "best"
                            compound_key = f"{mname}_{ckpt_type}"
                            if compound_key not in MODEL_PATHS:
                                MODEL_PATHS[compound_key] = ckpt
                                CKPT_MULTS[compound_key] = CKPT_TYPE_WEIGHTS.get(ckpt_type, 0.80)
                                if os.path.exists(meta_path):
                                    META_PATHS[compound_key] = meta_path
                                found_any = True
    if not found_any:
        print(f"[WARN] {mname} not found in any of {spec['dataset_slugs']}")

assert len(MODEL_PATHS) > 0, "FATAL: No model checkpoints found!"

print(f"[CFG] Loaded {len(MODEL_PATHS)} checkpoint variant(s) for ensemble:")
for k, v in MODEL_PATHS.items():
    sz = os.path.getsize(v) / 1e6
    mult = CKPT_MULTS.get(k, 1.0)
    print(f"  {k}: {v} ({sz:.1f} MB, weight_mult={mult:.2f})")

DEFAULT_ROI = (128, 128, 128)  # trained at 128³ max; larger patches are OOD and OOM
INF_OVERLAP = 0.50   # 0.50→125 patches/vol (was 0.75→343 patches/vol which timed out on 120 vols)
USE_TTA = True
MAX_PRED_HOURS = 8.5
ENS_MODE = "logit_weighted"

# v3: Role weights for ensemble (balanced > topology > surface)
ROLE_WEIGHTS = {"generalist": 1.4, "anti_merge": 1.1, "surface": 1.3, "balanced": 1.0}
CONF_TEMPERATURE = 0.80

# v3: Post-processing (updated defaults)
PP_DUST_MIN_3D = 192
PP_HOLE_MAX_2D = 128   # was 32 — fill larger surface through-holes per Z-slice
PP_OPEN_R_XY = 1
PP_OPEN_R_3D = 1
PP_SMOOTH_SIGMA = 0.3
PP_BRIDGE_KILL = True
PP_BK_NECK_R = 1
PP_BK_MIN_LOBE = 10000
PP_BK_MIN_NECK_LEN = 2
PP_EMPTY_TL_DROP = 0.08
PP_EMPTY_TH_DROP = 0.10
PP_OVERFULL_FRAC = 0.45
PP_OVERFULL_TH_RAISE = 0.05
DEFAULT_TL = 0.34
DEFAULT_TH = 0.62
AUTO_DROP_THRESH = 0.15
# TTA only on difficult volumes (saves 7x overhead on easy ones). 0.35 = ~top 30% of volumes.
TTA_DIFFICULTY_THRESH = 0.35
# Upgrade 4: Pathology gate — bridge-kill only if CC count above this
PP_PATHOLOGY_CC_THRESH = 8

# Global inference patch cap — mutable list so OOM retry can reduce it without
# passing it through the thread closure boundary.  Starts at 128³ (max quality).
# On CUDA OOM the main loop reduces this and retries the same volume.
# Ladder: 128 → 96 → 64 (each step is ~56% memory reduction for cubic patches).
_INF_PATCH_CAP_GLOBAL = [128, 128, 128]

# §1 ID contract: test.csv is the authoritative source of test IDs.
# Old notebook design: never hard-code IDs, never use directory listing alone.
# The actual hidden test set has ~120 volumes; test.csv lists every one.
_test_csv_path = os.path.join(ROOT_DIR, "test.csv")
if os.path.exists(_test_csv_path):
    import pandas as _pd
    _test_df = _pd.read_csv(_test_csv_path)
    _csv_ids = [str(i) for i in _test_df["id"].tolist()]
    # Deduplicate while preserving CSV order
    _seen_ids = set()
    test_ids = [i for i in _csv_ids if i not in _seen_ids and not _seen_ids.add(i)]
    test_paths = [os.path.join(TEST_DIR, f"{i}.tif") for i in test_ids]
    _on_disk = [os.path.exists(p) for p in test_paths]
    _missing = [i for i, ok in zip(test_ids, _on_disk) if not ok]
    if _missing:
        print(f"[WARN] {len(_missing)}/{len(test_ids)} test IDs in CSV have no .tif on disk: {_missing[:5]}")
    print(f"[CFG] test.csv → {len(test_ids)} IDs, {sum(_on_disk)} found on disk")
else:
    # Fallback: directory scan (backward compat for test runs without test.csv)
    print("[WARN] test.csv not found — falling back to TEST_DIR directory scan")
    test_paths = sorted([os.path.join(TEST_DIR, f)
                         for f in os.listdir(TEST_DIR) if f.endswith(".tif")]) \
                 if os.path.isdir(TEST_DIR) else []
    test_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_paths]
print(f"[CFG] {len(test_ids)} test volumes, fusion={ENS_MODE}")
'''


CELL_PRED_ARCHITECTURE = r'''
# ============================================================
# Architecture (inference mode — no aux heads needed)
# ============================================================

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int): kernel_size = [kernel_size] * 3
        if isinstance(stride, int): stride = [stride] * 3
        padding = [k // 2 for k in kernel_size]
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm = nn.InstanceNorm3d(out_ch, eps=1e-5, affine=True)
        self.act = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResBlock3D(nn.Module):
    def __init__(self, ch, kernel_size=3, bias=True):
        super().__init__()
        p = kernel_size // 2
        self.conv1 = nn.Conv3d(ch, ch, kernel_size, padding=p, bias=bias)
        self.norm1 = nn.InstanceNorm3d(ch, eps=1e-5, affine=True)
        self.conv2 = nn.Conv3d(ch, ch, kernel_size, padding=p, bias=bias)
        self.norm2 = nn.InstanceNorm3d(ch, eps=1e-5, affine=True)
        self.act = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        r = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + r)

class ResidualEncoderUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2,
                 features=(32,64,128,256,320,320), blocks=(1,3,4,6,6,6),
                 strides=((1,1,1),(2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2)),
                 dec_convs=(1,1,1,1,1), deep_supervision=False, use_grad_ckpt=False,
                 aux_heads=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self._ckpt = use_grad_ckpt
        self.aux_heads = aux_heads
        n = len(features)
        self.enc = nn.ModuleList()
        for s in range(n):
            in_c = in_channels if s == 0 else features[s-1]
            out_c = features[s]
            st = list(strides[s]) if isinstance(strides[s], (list,tuple)) else [strides[s]]*3
            layers = [ConvBlock3D(in_c, out_c, 3, stride=st)]
            for _ in range(blocks[s]-1):
                layers.append(ResBlock3D(out_c, 3))
            self.enc.append(nn.Sequential(*layers))
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.seg = nn.ModuleList()
        for i in range(n-1):
            s = n-1-i
            enc_ch, skip_ch, out_ch = features[s], features[s-1], features[s-1]
            st = list(strides[s]) if isinstance(strides[s], (list,tuple)) else [strides[s]]*3
            self.up.append(nn.ConvTranspose3d(enc_ch, enc_ch, kernel_size=st, stride=st, bias=True))
            nc = dec_convs[i] if i < len(dec_convs) else 1
            dl = []
            for c in range(nc):
                dl.append(ConvBlock3D((enc_ch+skip_ch) if c==0 else out_ch, out_ch, 3))
            self.dec.append(nn.Sequential(*dl))
            self.seg.append(nn.Conv3d(out_ch, num_classes, 1))
        if aux_heads:
            self.sdf_head = nn.Conv3d(features[0], 1, 3, padding=1)
            self.topo_head = nn.Conv3d(features[0], 1, 3, padding=1)

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            skips.append(x)
        outputs = []
        x = skips[-1]
        for i, (u, d, s) in enumerate(zip(self.up, self.dec, self.seg)):
            skip = skips[-(i+2)]
            x = u(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = d(x)
            outputs.append(s(x))
        outputs = outputs[::-1]
        if self.deep_supervision and self.training:
            return outputs
        return outputs[0]
'''


CELL_PRED_LOAD = r'''
# ============================================================
# Load Models (v3: handles aux head keys gracefully)
# ============================================================
models = {}
metas = {}
model_devices = {}  # mname → device; models distributed round-robin across all GPUs

_n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
_gpu_idx = 0
if _n_gpus > 1:
    print(f"[LOAD] {_n_gpus} GPUs detected — distributing models round-robin")

for mname, ckpt_path in MODEL_PATHS.items():
    print(f"\n[LOAD] {mname} from {ckpt_path}...")
    m = ResidualEncoderUNet(
        in_channels=IN_CHANNELS, num_classes=NUM_CLASSES,
        features=FEATURES, blocks=BLOCKS, strides=STRIDES,
        dec_convs=DEC_CONVS, deep_supervision=False, use_grad_ckpt=False,
        aux_heads=False  # No aux heads needed for inference
    )

    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # Strip torch.compile prefix: checkpoints saved while compiled have "_orig_mod." on every key.
    # The inference model is not compiled at load time, so keys must be stripped first.
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
        print(f"[LOAD] Stripped _orig_mod. prefix (compiled checkpoint) for {mname}")

    # v3: Filter out aux head keys (sdf_head, topo_head) from checkpoint
    filtered_sd = {k: v for k, v in sd.items()
                   if not k.startswith("sdf_head.") and not k.startswith("topo_head.")}

    missing, unexpected = m.load_state_dict(filtered_sd, strict=False)
    if unexpected:
        # Filter out seg head mismatches from deep supervision
        critical_unexpected = [k for k in unexpected if ".seg." not in k]
        if critical_unexpected:
            raise RuntimeError(f"[FATAL] {mname}: {len(critical_unexpected)} UNEXPECTED keys!")
    if missing:
        non_seg_missing = [k for k in missing if ".seg." not in k]
        if non_seg_missing:
            raise RuntimeError(f"[FATAL] {mname}: {len(non_seg_missing)} critical missing keys!")

    # Round-robin GPU assignment: remembers which GPU each model will run on.
    # Models are kept in CPU RAM and moved to GPU one at a time during inference.
    # This keeps ~13.9 GB free per GPU → 128³ PRECOMP_GEMM workspace (10.1 GB) fits safely.
    # (All-GPU-resident approach left only 10.44 GB free → 34.17 GB workspace → OOM.)
    _dev = torch.device(f"cuda:{_gpu_idx % _n_gpus}") if torch.cuda.is_available() else torch.device("cpu")
    _gpu_idx += 1
    m = m.to('cpu').eval().half()   # store in CPU RAM; moved to GPU lazily per-inference
    models[mname] = m
    model_devices[mname] = _dev

    meta_path = META_PATHS.get(mname)
    if meta_path and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        meta["ckpt_weight_mult"] = CKPT_MULTS.get(mname, 1.0)  # inject per-variant weight multiplier
        metas[mname] = meta
        print(f"  Meta: fold={meta.get('fold')}, score={meta.get('best_val_score', meta.get('combined_proxy', '?'))}, "
              f"tl={meta.get('tl', '?')}, th={meta.get('th', '?')}, temp={meta.get('temperature', '?')}, "
              f"ckpt_mult={meta['ckpt_weight_mult']:.2f}")
    else:
        metas[mname] = {"model": mname, "ckpt_weight_mult": CKPT_MULTS.get(mname, 1.0)}

    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"  {mname}: {n_params:.1f}M params")

# Auto-drop weak models
active_models = {}
for mname, m in models.items():
    meta = metas.get(mname, {})
    proxy = meta.get("best_val_score", meta.get("combined_proxy", 0.5))
    if proxy < AUTO_DROP_THRESH and len(models) > 1:
        print(f"  [DROP] {mname}: proxy={proxy:.4f} < {AUTO_DROP_THRESH}")
    else:
        active_models[mname] = m

if len(active_models) == 0:
    active_models = dict(models)

models = active_models
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

print(f"\n[LOAD] {len(models)} model(s) active for inference")
'''


CELL_PRED_POSTPROCESS = r'''
# ============================================================
# Post-Processing v3: 26-CC everywhere, smart dust, multi-scale bridge killer
# ============================================================

def hysteresis_3d(prob, tl, th):
    """3D hysteresis thresholding with 26-connectivity."""
    struct26 = generate_binary_structure(3, 3)
    strong = prob >= th
    weak = prob >= tl
    cc, n = cc_label(weak, structure=struct26)
    if n == 0:
        return np.zeros_like(prob, dtype=np.uint8)
    strong_ids = np.unique(cc[strong])
    strong_ids = strong_ids[strong_ids != 0]
    return np.isin(cc, strong_ids).astype(np.uint8)


def smart_dust_removal(mask, probs=None, min_size=PP_DUST_MIN_3D, conf_threshold=0.7):
    """v3: Remove small components only if they don't contain high-confidence voxels."""
    if min_size <= 0:
        return mask
    struct26 = generate_binary_structure(3, 3)
    cc, n = cc_label(mask, structure=struct26)
    if n == 0:
        return mask
    sizes = np.bincount(cc.ravel())
    keep = np.zeros(n + 1, dtype=bool)
    for i in range(1, n + 1):
        if sizes[i] >= min_size:
            keep[i] = True
        elif probs is not None:
            # Keep small components with high-confidence voxels
            comp_mask = (cc == i)
            if probs[comp_mask].max() > conf_threshold:
                keep[i] = True
    return (keep[cc]).astype(np.uint8)


def fill_holes_2d(mask, max_area=PP_HOLE_MAX_2D):
    """v3: Cautious 2D hole filling — only fill if fully surrounded."""
    if max_area <= 0:
        return mask
    out = mask.copy()
    struct2d = generate_binary_structure(2, 1)
    for z in range(out.shape[0]):
        sl = out[z].astype(bool)
        bg = ~sl
        lbl, n = cc_label(bg, structure=struct2d)
        if n == 0:
            continue
        border = set()
        border.update(lbl[0, :].tolist())
        border.update(lbl[-1, :].tolist())
        border.update(lbl[:, 0].tolist())
        border.update(lbl[:, -1].tolist())
        for k in range(1, n + 1):
            if k in border:
                continue
            area = int((lbl == k).sum())
            if area <= max_area:
                sl[lbl == k] = True
        out[z] = sl.astype(np.uint8)
    return out


def xy_opening(mask, radius=PP_OPEN_R_XY):
    if radius <= 0:
        return mask
    struct2d = generate_binary_structure(2, 1)
    out = np.empty_like(mask)
    for z in range(mask.shape[0]):
        sl = mask[z].astype(bool)
        sl = binary_erosion(sl, structure=struct2d, iterations=radius)
        sl = binary_dilation(sl, structure=struct2d, iterations=radius)
        out[z] = sl.astype(np.uint8)
    return out


def opening_3d(mask, radius=PP_OPEN_R_3D):
    if radius <= 0:
        return mask
    struct26 = generate_binary_structure(3, 3)  # v3: 26-connectivity
    m = mask.astype(bool)
    m = binary_erosion(m, structure=struct26, iterations=radius)
    m = binary_dilation(m, structure=struct26, iterations=radius)
    return m.astype(np.uint8)


def thickness_bridge_killer(mask, probs=None, theta_neck=3.0, min_lobe_size=PP_BK_MIN_LOBE):
    """v3: Bridge cutting via local thickness + morphological skeleton.
    Brain doc algorithm L1:
    1. Compute local thickness (2 * distance_transform inside FG)
    2. Find thin neck voxels (thickness <= theta_neck)
    3. For each thin neck component, test if cutting improves topology
    4. Accept cut only if it splits into 2+ large components without explosion
    """
    struct26 = generate_binary_structure(3, 3)
    struct6 = generate_binary_structure(3, 1)
    mask_bool = mask.astype(bool)
    orig_sum = mask_bool.sum()
    if orig_sum == 0:
        return mask

    # Step 1: Local thickness map
    dt_fg = distance_transform_edt(mask_bool)
    thickness = 2.0 * dt_fg  # local thickness proxy

    # Step 2: Find thin neck candidates
    neck_mask = mask_bool & (thickness > 0) & (thickness <= theta_neck)
    if not neck_mask.any():
        return mask

    # Step 3: Evaluate each thin-neck connected component
    cc_neck, n_neck = cc_label(neck_mask, structure=struct26)
    if n_neck == 0:
        return mask

    # Pre-compute original component count
    _, n_orig = cc_label(mask_bool, structure=struct26)
    result = mask_bool.copy()
    cuts_made = 0

    for ni in range(1, min(n_neck + 1, 50)):  # cap iterations
        neck_comp = (cc_neck == ni)
        neck_size = neck_comp.sum()
        if neck_size < 3 or neck_size > 5000:  # too small or too big to be a neck
            continue

        # Temporarily remove this neck
        test_mask = result.copy()
        test_mask[neck_comp] = False

        # Check resulting topology
        cc_test, n_test = cc_label(test_mask, structure=struct26)
        sizes_test = np.bincount(cc_test.ravel())

        # Accept cut if:
        # - Creates exactly 2+ large components (separates real structures)
        # - Doesn't cause component explosion (n_test <= n_orig + 5)
        # - Doesn't remove too much volume
        n_large = sum(1 for s in sizes_test[1:] if s >= min_lobe_size)
        vol_kept = test_mask.sum() / max(orig_sum, 1)

        if n_large >= 2 and n_test <= n_orig + 5 and vol_kept >= 0.5:
            result = test_mask
            cuts_made += 1

    if cuts_made > 0:
        return result.astype(np.uint8)
    return mask


def multiscale_bridge_killer(mask, radii=(1, 2, 3), min_lobe_size=PP_BK_MIN_LOBE):
    """Fallback: Multi-scale erosion-based bridge killer."""
    struct6 = generate_binary_structure(3, 1)
    struct26 = generate_binary_structure(3, 3)

    best_result = mask.copy()
    orig_sum = mask.sum()
    if orig_sum == 0:
        return mask

    for radius in radii:
        eroded = binary_erosion(mask.astype(bool), structure=struct6, iterations=radius)
        if not eroded.any():
            continue

        cc_eroded, n_eroded = cc_label(eroded, structure=struct26)
        sizes = np.bincount(cc_eroded.ravel())

        result = np.zeros_like(mask, dtype=np.uint8)
        for i in range(1, n_eroded + 1):
            if sizes[i] < min_lobe_size:
                continue
            comp = (cc_eroded == i)
            grown = binary_dilation(comp, structure=struct6, iterations=radius)
            grown = grown & mask.astype(bool)
            result[grown] = 1

        if result.sum() >= orig_sum * 0.4:
            best_result = result
            break

    return best_result


def cavity_control_3d(mask, max_cavity_size=2000):  # was 500 — fill larger enclosed 3D cavities
    """Brain doc L3: Fill small enclosed cavities (k2 proxy).
    C = fill(P) - P. Fill only if |C_j| <= threshold."""
    try:
        struct26 = generate_binary_structure(3, 3)
        mask_bool = mask.astype(bool)
        # Fill ALL holes by flood-filling from borders
        from scipy.ndimage import binary_fill_holes
        filled = binary_fill_holes(mask_bool, structure=struct26)
        cavities = filled & ~mask_bool
        if not cavities.any():
            return mask
        # Only fill small cavities
        cc_cav, n_cav = cc_label(cavities, structure=struct26)
        result = mask.copy()
        filled_count = 0
        for i in range(1, n_cav + 1):
            comp = (cc_cav == i)
            if comp.sum() <= max_cavity_size:
                result[comp] = 1
                filled_count += 1
        return result
    except Exception:
        return mask


def _close_surface_gaps(mask, fused_prob, max_bridge_r=6, prob_thresh=0.25):
    """Aggressive surface gap-closure for 2-12 CC range.
    Progressively dilates the entire mask to find the minimum bridge radius that
    merges all CCs into one connected surface, then erodes back.
    Constrained: bridging voxels must have fused_prob > prob_thresh (surface-consistent).
    Called after Frangi+postprocess but before catastrophe_fallback."""
    struct26 = generate_binary_structure(3, 3)
    _, n_start = cc_label(mask.astype(bool), structure=struct26)
    if n_start <= 1:
        return mask   # already clean

    best_mask = mask.copy()
    best_n = n_start

    prob_ok = (fused_prob > prob_thresh)

    for r in range(1, max_bridge_r + 1):
        dilated = ndi.binary_dilation(mask.astype(bool), iterations=r)
        # Bridge only through high-probability voxels (surface-consistent)
        bridged = (dilated & prob_ok) | mask.astype(bool)
        eroded = ndi.binary_erosion(bridged, iterations=max(1, r - 1))
        # Keep original + bridged to avoid shrinking
        candidate = (eroded | mask.astype(bool)).astype(np.uint8)
        _, n_cand = cc_label(candidate.astype(bool), structure=struct26)
        if n_cand < best_n:
            best_n = n_cand
            best_mask = candidate
            if n_cand == 1:
                break  # fully connected, stop

    if best_n < n_start:
        print(f"    [CLOSE_GAPS] Surface gaps closed: {n_start} → {best_n} CCs")
    return best_mask


def anti_split_repair(mask, probs=None, max_gap=5, min_comp_size=500):
    """Brain doc L2: Careful fragment reconnection.
    Merge fragments only if: gap small AND thickness suggests same sheet.
    NEVER connect if it would create bridges across wraps."""
    struct26 = generate_binary_structure(3, 3)
    mask_bool = mask.astype(bool)
    cc, n = cc_label(mask_bool, structure=struct26)
    if n <= 1:
        return mask

    sizes = np.bincount(cc.ravel())
    # Find small fragments
    small_comps = [i for i in range(1, n + 1) if 10 < sizes[i] < min_comp_size]
    if not small_comps:
        return mask

    result = mask.copy()
    repaired = 0

    for ci in small_comps:
        comp = (cc == ci)
        # Find nearest large component via dilation
        dilated = ndi.binary_dilation(comp, iterations=max_gap)
        # Check which large components the dilation touches
        touched_labels = set(np.unique(cc[dilated & (cc != ci) & (cc > 0)])) - {0}
        large_touched = [l for l in touched_labels if sizes[l] >= min_comp_size]

        if len(large_touched) != 1:
            # Don't repair if touching 0 or 2+ large components (ambiguous/bridge risk)
            continue

        # Check gap region — only connect if probs are reasonably high in the gap
        gap_region = dilated & ~mask_bool
        if probs is not None and gap_region.any():
            gap_confidence = probs[gap_region].mean() if gap_region.any() else 0
            if gap_confidence < 0.3:
                continue  # Low confidence gap — don't connect

        # Connect via constrained dilation of the small component
        bridge = ndi.binary_dilation(comp, iterations=max_gap) & ~mask_bool
        # Only keep bridge voxels between the two components
        target_comp = (cc == large_touched[0])
        target_dilated = ndi.binary_dilation(target_comp, iterations=max_gap)
        bridge = bridge & target_dilated
        result[bridge] = 1
        repaired += 1

    return result


def postprocess(prob, tl, th, probs_for_dust=None, verbose=True, _idempotency_check=True,
                n_cc_pre=None):
    """v3 PODIUM: Full post-processing pipeline with brain doc algorithms.
    Upgrade 4: Pathology-gated bridge-kill and anti-split — only run aggressive steps
    when pathology indicators (CC count, fragmentation) actually require them."""
    tag = "    [PP]" if verbose else ""

    prob = gaussian_filter(prob, sigma=PP_SMOOTH_SIGMA).astype(np.float32)
    prob = np.clip(prob, 0, 1)

    mask = hysteresis_3d(prob, tl, th)
    if verbose:
        print(f"{tag} After hysteresis(tl={tl:.2f},th={th:.2f}): {mask.sum()} FG voxels")

    # Fallback for empty mask
    n_retries = 0
    retry_tl, retry_th = tl, th
    while mask.sum() == 0 and n_retries < 3:
        retry_tl = max(0.15, retry_tl - PP_EMPTY_TL_DROP)
        retry_th = max(0.30, retry_th - PP_EMPTY_TH_DROP)
        mask = hysteresis_3d(prob, retry_tl, retry_th)
        n_retries += 1

    # 3D opening
    if PP_OPEN_R_3D > 0:
        before = mask.sum()
        mask = opening_3d(mask, PP_OPEN_R_3D)
        if verbose and mask.sum() != before:
            print(f"{tag} 3D opening: {before} -> {mask.sum()}")

    # Upgrade 4: Pathology-gated bridge kill — only if CC count signals fragmentation
    struct26_pp = generate_binary_structure(3, 3)
    _, n_cc_now = cc_label(mask.astype(bool), structure=struct26_pp)
    # Use pre-computed CC count if caller provided it (from fused logit at th)
    _n_cc_check = n_cc_pre if n_cc_pre is not None else n_cc_now
    _run_bridge_kill = PP_BRIDGE_KILL and (_n_cc_check > PP_PATHOLOGY_CC_THRESH)
    if verbose:
        gate_tag = "ACTIVE" if _run_bridge_kill else f"SKIPPED (CC={_n_cc_check}<={PP_PATHOLOGY_CC_THRESH})"
        print(f"{tag} Bridge-kill gate: {gate_tag}")

    if _run_bridge_kill:
        before = mask.sum()
        try:
            mask = thickness_bridge_killer(mask, probs=probs_for_dust)
        except Exception:
            mask = multiscale_bridge_killer(mask)
        if verbose and mask.sum() != before:
            _, n_after = cc_label(mask.astype(bool), structure=struct26_pp)
            print(f"{tag} Bridge-kill: {before} -> {mask.sum()} ({n_after} components)")

    # v3: Smart dust removal (always run)
    before = mask.sum()
    mask = smart_dust_removal(mask, probs=probs_for_dust)
    if verbose and mask.sum() != before:
        print(f"{tag} Smart dust: {before} -> {mask.sum()}")

    # Brain doc L3: 3D cavity control
    before = mask.sum()
    mask = cavity_control_3d(mask)
    if verbose and mask.sum() != before:
        print(f"{tag} Cavity fill: {before} -> {mask.sum()}")

    mask = fill_holes_2d(mask)
    if PP_OPEN_R_XY > 0:
        mask = xy_opening(mask)

    # Upgrade 4: Pathology-gated anti-split — only if fragmented
    _, before_cc = cc_label(mask.astype(bool), structure=struct26_pp)
    _run_anti_split = (before_cc > 3) and (_n_cc_check >= 3)
    if _run_anti_split:
        before = mask.sum()
        mask = anti_split_repair(mask, probs=probs_for_dust)
        _, after_cc = cc_label(mask.astype(bool), structure=struct26_pp)
        if verbose and after_cc != before_cc:
            print(f"{tag} Anti-split: {before_cc} -> {after_cc} components")

    # Overfull guard
    fg_frac = mask.sum() / max(mask.size, 1)
    if fg_frac > PP_OVERFULL_FRAC:
        raised_th = th + PP_OVERFULL_TH_RAISE
        if verbose:
            print(f"{tag} Overfull ({fg_frac:.3f})! Re-thresholding with th={raised_th:.2f}")
        mask = hysteresis_3d(prob, tl, raised_th)
        mask = smart_dust_removal(mask)
        mask = fill_holes_2d(mask)

    # Playbook: Idempotency guard — postproc applied twice should give same result
    if _idempotency_check:
        mask2 = postprocess(prob, tl, th, probs_for_dust=probs_for_dust, verbose=False,
                            _idempotency_check=False, n_cc_pre=n_cc_pre)
        if not np.array_equal(mask, mask2):
            diff_vox = int(np.sum(mask != mask2))
            if verbose:
                print(f"{tag} WARNING: Postproc not idempotent! {diff_vox} voxels differ. Using stabilized result.")
            mask = mask2

    return mask


# Ensemble threshold computation
def _weighted_median(values, weights):
    if not values: return 0.5
    pairs = sorted(zip(values, weights))
    cumw = np.cumsum([w for _, w in pairs])
    total = cumw[-1]
    idx = np.searchsorted(cumw, total * 0.5)
    idx = min(idx, len(pairs) - 1)
    return pairs[idx][0]

_tls, _ths, _weights = [], [], []
for mn, meta in metas.items():
    if mn not in models: continue
    _tls.append(meta.get("tl", DEFAULT_TL))
    _ths.append(meta.get("th", DEFAULT_TH))
    proxy = meta.get("best_val_score", meta.get("combined_proxy", 0.5))
    _weights.append(max(float(proxy), 0.1))

if _weights:
    ENS_TL = _weighted_median(_tls, _weights)
    ENS_TH = _weighted_median(_ths, _weights)
else:
    ENS_TL, ENS_TH = DEFAULT_TL, DEFAULT_TH

PER_MODEL_TL = {mn: metas.get(mn, {}).get("tl", DEFAULT_TL) for mn in models}
PER_MODEL_TH = {mn: metas.get(mn, {}).get("th", DEFAULT_TH) for mn in models}

print(f"[PP] Ensemble thresholds: tl={ENS_TL:.3f}, th={ENS_TH:.3f}")
'''


CELL_PRED_INFERENCE = r'''
# ============================================================
# Ensemble Sliding-Window Inference v3
# Upgrade 3: Logit-space Gaussian blending (seam-safe)
# Upgrade 2: Trimmed-mean logit fusion + per-model temperature
# Upgrade 5: Hard-volume adaptive TTA
# Upgrade 1: Learned per-volume threshold predictor
# ============================================================

def normalize_volume(vol, robust=True):
    """Brain doc: robust z-score with intensity clipping."""
    v = vol.astype(np.float32)
    lo, hi = np.percentile(v, [0.5, 99.5])
    v = np.clip(v, lo, hi)
    if robust:
        mu = np.median(v)
        sigma = np.median(np.abs(v - mu)) * 1.4826
    else:
        mu = v.mean()
        sigma = v.std()
    return (v - mu) / max(sigma, 1e-8)


def _gauss_1d(n):
    if n <= 1: return np.ones(n, dtype=np.float32)
    x = np.linspace(-1, 1, n, dtype=np.float32)
    return np.exp(-2.0 * x * x)


def _gauss_3d(shape):
    w = (_gauss_1d(shape[0])[:, None, None] *
         _gauss_1d(shape[1])[None, :, None] *
         _gauss_1d(shape[2])[None, None, :])
    return w / (w.max() + 1e-8)


def sharpen_probs(prob, temperature):
    if temperature == 1.0 or temperature <= 0:
        return prob
    p_clip = np.clip(prob, 1e-6, 1 - 1e-6)
    logit = np.log(p_clip / (1 - p_clip)) / temperature
    return 1.0 / (1.0 + np.exp(-logit))


@torch.no_grad()
def infer_single_model(vol_f32, model_m, roi, overlap, use_tta=False):
    """Sliding window inference. Returns FG logit map (D,H,W).
    Upgrade 3: accumulate logits (not probs) with Gaussian weights — seam-artifact free.
    Caller applies temperature + sigmoid once after this function."""
    D, H, W = vol_f32.shape
    rD, rH, rW = roi
    sD = max(1, int(rD * (1.0 - overlap)))
    sH = max(1, int(rH * (1.0 - overlap)))
    sW = max(1, int(rW * (1.0 - overlap)))
    w3d = _gauss_3d((rD, rH, rW))

    acc = np.zeros((D, H, W), dtype=np.float32)
    wacc = np.zeros((D, H, W), dtype=np.float32)

    z_starts = sorted(set(list(range(0, max(1, D-rD+1), sD)) + [max(0, D-rD)]))
    y_starts = sorted(set(list(range(0, max(1, H-rH+1), sH)) + [max(0, H-rH)]))
    x_starts = sorted(set(list(range(0, max(1, W-rW+1), sW)) + [max(0, W-rW)]))

    _model_device = next(model_m.parameters()).device  # use model's assigned GPU, not global DEVICE
    amp_on = (_model_device.type == "cuda")
    is_half = next(model_m.parameters()).dtype == torch.float16

    def _run_patch_logit(patch_np):
        """Run patch through model. Returns FG log-odds: logit[1] - logit[0]."""
        t = torch.from_numpy(patch_np[None, None].copy())
        if is_half: t = t.half()
        t = t.to(_model_device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_on):
            logits = model_m(t)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
        lg = logits[0].float()   # shape (C, D, H, W)
        # log-odds FG vs BG — avoids softmax saturation
        logit_fg = (lg[1] - lg[0]).cpu().numpy()
        return logit_fg

    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                patch = vol_f32[z0:z0+rD, y0:y0+rH, x0:x0+rW]
                if patch.shape != (rD, rH, rW):
                    continue

                logit_fg = _run_patch_logit(patch)

                if use_tta:
                    n_tta = 1
                    # X, Y, Z axis flips — 3 axes × 2 orientations = 6 augmentations
                    # In-plane rotations removed (blueprint: avoid heavy TTA; 3 flips saves
                    # 3/7 ≈ 43% TTA compute vs original 7-way, preserving inference budget)
                    for ax in [2, 1, 0]:
                        flipped = np.flip(patch, axis=ax).copy()
                        lf = _run_patch_logit(flipped)
                        logit_fg = logit_fg + np.flip(lf, axis=ax)
                        n_tta += 1
                    logit_fg /= n_tta

                # Accumulate logits (Upgrade 3: logit-space blending, no seam artifacts)
                acc[z0:z0+rD, y0:y0+rH, x0:x0+rW] += logit_fg * w3d
                wacc[z0:z0+rD, y0:y0+rH, x0:x0+rW] += w3d

    wacc = np.maximum(wacc, 1e-8)
    return acc / wacc  # returns normalized logit map, NOT probability


def _difficulty_score(logit_map):
    """Upgrade 5: Volume difficulty for adaptive TTA.
    High score = complex topology, needs TTA. Low = easy, skip TTA."""
    prob = 1.0 / (1.0 + np.exp(-np.clip(logit_map, -50, 50)))
    fg_mask = (prob > 0.5)
    if not fg_mask.any():
        return 1.0  # empty = ambiguous = hard
    struct26 = generate_binary_structure(3, 3)
    _, n_cc = cc_label(fg_mask, structure=struct26)
    # Boundary voxels: proportion near decision boundary [0.3, 0.7]
    boundary_frac = float(np.mean((prob > 0.3) & (prob < 0.7)))
    # Fragmentation: CCs per unit FG volume
    fg_frac = float(fg_mask.mean())
    cc_norm = n_cc / max(fg_frac * 1000, 1)
    return float(np.clip(boundary_frac * 0.6 + min(cc_norm / 10.0, 0.4), 0.0, 1.0))


def _learned_threshold(prob, meta_coefs):
    """Upgrade 1: Use linear predictor coefficients from calibration meta.
    Features: [mean, std, p10, p25, p50, p75, p90, fg01, fg02, fg03, fg05].
    Returns (tl, th). Falls back to find_stable_threshold if no coefficients."""
    if not meta_coefs:
        return find_stable_threshold(prob)
    try:
        p_flat = prob.ravel()
        pcts = np.percentile(p_flat, [10, 25, 50, 75, 90])
        feats = np.array([
            float(p_flat.mean()),
            float(p_flat.std()),
            float(pcts[0]), float(pcts[1]), float(pcts[2]),
            float(pcts[3]), float(pcts[4]),
            float((p_flat > 0.1).mean()),
            float((p_flat > 0.2).mean()),
            float((p_flat > 0.3).mean()),
            float((p_flat > 0.5).mean()),
        ], dtype=np.float32)
        tl_coefs = np.array(meta_coefs.get("tl_coefs", []), dtype=np.float32)
        th_coefs = np.array(meta_coefs.get("th_coefs", []), dtype=np.float32)
        tl_intercept = float(meta_coefs.get("tl_intercept", 0.34))
        th_intercept = float(meta_coefs.get("th_intercept", 0.62))
        if len(tl_coefs) == len(feats) and len(th_coefs) == len(feats):
            tl = float(np.clip(np.dot(tl_coefs, feats) + tl_intercept, 0.18, 0.60))
            th = float(np.clip(np.dot(th_coefs, feats) + th_intercept, 0.38, 0.90))  # matches Nelder-Mead range
            if th > tl + 0.05:
                return tl, th
    except Exception:
        pass
    return find_stable_threshold(prob)


def risk_aware_fusion(per_model_probs, model_weights):
    """v3: Downweight models based on topology risk signals."""
    struct26 = generate_binary_structure(3, 3)
    names = list(per_model_probs.keys())
    adjusted = dict(model_weights)

    if len(names) < 2:
        return adjusted

    # Per-model topology risk assessment
    fg_fracs = {}
    comp_counts = {}
    for nm in names:
        binary = (per_model_probs[nm] > 0.5).astype(np.uint8)
        fg_fracs[nm] = float(binary.mean())
        if binary.sum() > 0:
            _, n_cc = cc_label(binary, structure=struct26)
            comp_counts[nm] = n_cc
        else:
            comp_counts[nm] = 0

    median_fg = float(np.median(list(fg_fracs.values())))
    median_cc = float(np.median(list(comp_counts.values()))) if comp_counts else 0

    for nm in names:
        # Near-empty mask — complete exclusion (0.15 weight still poisons trimmed mean)
        if fg_fracs[nm] < 1e-5 and median_fg > 1e-4:
            adjusted[nm] = 0.0
            print(f"    [RISK] {nm}: near-empty, EXCLUDED (weight -> 0)")
            continue

        # FG fraction outlier
        # Thresholds loosened (3.0→5.0, 0.33→0.20): the previous 3.0 threshold incorrectly
        # downweighted Model A (fg_ratio≈4.6, genuinely large surface) to 0.3× weight,
        # causing ensemble catastrophe. Plates can legitimately have 5–8× the median FG
        # when some models underpredict. Only penalise true outliers (>5× or <20% median).
        if median_fg > 1e-6:
            ratio = fg_fracs[nm] / max(median_fg, 1e-6)
            if ratio < 0.05 or ratio > 20.0:
                # Severely outlying (predicts essentially nothing or 20× too much) → exclude
                adjusted[nm] = 0.0
                print(f"    [RISK] {nm}: fg_ratio={ratio:.2f}, EXCLUDED (weight -> 0)")
                continue
            elif ratio > 5.0 or ratio < 0.20:
                adjusted[nm] = model_weights[nm] * 0.5
                print(f"    [RISK] {nm}: fg_ratio={ratio:.1f}, weight -> {adjusted[nm]:.3f}")

        # Fragmentation risk
        if median_cc > 0:
            cc_ratio = comp_counts[nm] / max(median_cc, 1)
            if cc_ratio > 5.0:
                adjusted[nm] = min(adjusted[nm], model_weights[nm] * 0.25)
                print(f"    [RISK] {nm}: {comp_counts[nm]} CCs (5x median), weight -> {adjusted[nm]:.3f}")
            elif cc_ratio > 3.0:
                adjusted[nm] = min(adjusted[nm], model_weights[nm] * 0.5)

    return adjusted


def trimmed_mean_logit_fusion(per_model_probs, trim_frac=0.15):
    """Upgrade 2: Trimmed-mean logit fusion. More robust than median for 3+ models.
    z_fuse(x) = trimmed_mean_m z_m(x) — drops top+bottom trim_frac of models per voxel.
    Falls back to median for n<=3 (trim would leave too few votes)."""
    names = list(per_model_probs.keys())
    logits_stack = []
    for nm in names:
        p = np.clip(per_model_probs[nm], 1e-6, 1 - 1e-6)
        logits_stack.append(np.log(p / (1 - p)))
    logits_arr = np.stack(logits_stack, axis=0)   # (N, D, H, W)
    n = len(names)
    if n <= 3:
        # Median for small ensembles (trimming would remove too many)
        return np.median(logits_arr, axis=0)
    # Trimmed mean: sort along model axis, drop top and bottom k models per voxel
    k = max(1, int(n * trim_frac))
    logits_sorted = np.sort(logits_arr, axis=0)
    trimmed = logits_sorted[k:n-k]   # shape (n-2k, D, H, W)
    return trimmed.mean(axis=0)


def find_stable_threshold(prob, tl_range=(0.25, 0.55), th_range=(0.40, 0.70), steps=6):
    """Brain doc: Threshold stability curve — find flat maxima region.
    Compute S(t) for grid of thresholds and pick the most stable."""
    best_score = -1.0
    best_tl, best_th = 0.40, 0.50
    best_stability = 0.0

    tl_grid = np.linspace(tl_range[0], tl_range[1], steps)
    th_grid = np.linspace(th_range[0], th_range[1], steps)
    scores = {}

    for tl in tl_grid:
        for th in th_grid:
            if th <= tl:
                continue
            mask = hysteresis_3d(prob, float(tl), float(th))
            fg_frac = mask.sum() / max(mask.size, 1)
            if fg_frac < 0.001 or fg_frac > 0.60:
                scores[(tl, th)] = 0.0
                continue
            struct26 = generate_binary_structure(3, 3)
            _, n_cc = cc_label(mask.astype(bool), structure=struct26)
            # Heuristic score: moderate FG fraction, few components
            s = max(0, 1.0 - abs(fg_frac - 0.15) * 3) * max(0, 1.0 - n_cc / 30.0)
            scores[(tl, th)] = s

    if not scores:
        return best_tl, best_th

    # Find flat maxima: best score with neighbors also high (stability)
    for (tl, th), s in scores.items():
        if s <= 0:
            continue
        # Check stability: how much does score change with small perturbation
        neighbors = []
        for dtl in [-0.05, 0, 0.05]:
            for dth in [-0.05, 0, 0.05]:
                key = (round(tl + dtl, 3), round(th + dth, 3))
                if key in scores:
                    neighbors.append(scores[key])
        if len(neighbors) >= 3:
            stability = min(neighbors) / max(s, 1e-8)
        else:
            stability = 0.5
        combined = s * 0.6 + stability * 0.4
        if combined > best_score:
            best_score = combined
            best_tl, best_th = float(tl), float(th)
            best_stability = stability

    return best_tl, best_th


def catastrophe_fallback(fused_mask, per_model_probs, model_roles):
    """Brain doc N: Catastrophe detection + fallback ladder.
    If fused result has explosion/bridge-risk, fall back to specialist output."""
    struct26 = generate_binary_structure(3, 3)
    _, n_cc_fused = cc_label(fused_mask.astype(bool), structure=struct26)

    # Surface detection: papyrus forms connected sheets. >25 CCs means the ensemble
    # fragmented the surface (gap artefacts from fusing misaligned model outputs).
    # Threshold raised from 20→25 because multi-scale Frangi reduces CCs; slight extra
    # tolerance avoids premature fallback to a single (potentially less accurate) model.
    fg_frac = fused_mask.sum() / max(fused_mask.size, 1)
    is_catastrophe = (n_cc_fused > 25) or (fg_frac < 0.005 and fg_frac > 0)

    if not is_catastrophe:
        return fused_mask

    print(f"    [CATASTROPHE] Detected: {n_cc_fused} components, fg_frac={fg_frac:.4f}")

    # Fallback ladder: "balanced" first so model_x variants are found before specialists
    fallback_order = ["balanced", "anti_merge", "generalist", "surface"]
    for role in fallback_order:
        for nm, r in model_roles.items():
            if r == role and nm in per_model_probs:
                prob = per_model_probs[nm]
                fb_mask = hysteresis_3d(prob, ENS_TL, ENS_TH)
                fb_mask = smart_dust_removal(fb_mask, probs=prob)
                _, n_cc_fb = cc_label(fb_mask.astype(bool), structure=struct26)
                fb_fg = fb_mask.sum() / max(fb_mask.size, 1)
                if n_cc_fb <= 15 and fb_fg > 0.01:
                    print(f"    [FALLBACK] Using {nm} ({role}): {n_cc_fb} components")
                    return fb_mask

    # Ultimate fallback: most conservative (highest threshold)
    print(f"    [FALLBACK] Using raised threshold as last resort")
    for nm in per_model_probs:
        prob = per_model_probs[nm]
        fb_mask = hysteresis_3d(prob, 0.45, 0.65)
        fb_mask = smart_dust_removal(fb_mask, probs=prob, min_size=512)
        _, n_cc = cc_label(fb_mask.astype(bool), structure=struct26)
        if n_cc <= 20:
            return fb_mask

    return fused_mask


def _frangi_surface_filter_single(prob_f64, sigma, alpha=0.5):
    """Single-scale 3D Frangi plate filter. Returns unnormalized surfaceness map.
    Uses analytical Cardano eigenvalue decomposition (fully vectorized)."""
    p = prob_f64
    Hzz = gaussian_filter(p, sigma, order=[2, 0, 0])
    Hyy = gaussian_filter(p, sigma, order=[0, 2, 0])
    Hxx = gaussian_filter(p, sigma, order=[0, 0, 2])
    Hzy = gaussian_filter(p, sigma, order=[1, 1, 0])
    Hzx = gaussian_filter(p, sigma, order=[1, 0, 1])
    Hyx = gaussian_filter(p, sigma, order=[0, 1, 1])
    q = (Hzz + Hyy + Hxx) / 3.0
    bz = Hzz - q;  by = Hyy - q;  bx = Hxx - q
    p2 = (bz**2 + by**2 + bx**2 + 2.0*(Hzy**2 + Hzx**2 + Hyx**2)) / 6.0
    p_sq = np.sqrt(np.maximum(p2, 0.0));  del p2
    inv_p = np.where(p_sq > 1e-12, 1.0 / p_sq, 0.0)
    cz = bz*inv_p;  cy = by*inv_p;  cx = bx*inv_p;  del bz, by, bx
    czy = Hzy*inv_p;  czx = Hzx*inv_p;  cyx = Hyx*inv_p
    del Hzz, Hyy, Hxx, Hzy, Hzx, Hyx, inv_p
    r = (cz*(cy*cx - cyx**2) - czy*(czy*cx - cyx*czx) + czx*(czy*cyx - cy*czx)) / 2.0
    del cz, cy, cx, czy, czx, cyx
    phi = np.arccos(np.clip(r, -1.0, 1.0)) / 3.0;  del r
    two_p = 2.0 * p_sq;  del p_sq
    # Unsorted eigenvalues from Cardano (ev_a=k0, ev_b=k1, ev_c=k2)
    ev_a = q + two_p * np.cos(phi)
    ev_b = q + two_p * np.cos(phi + 2.0*np.pi/3.0)
    ev_c = 3.0*q - ev_a - ev_b;  del q, two_p, phi
    # Sort by magnitude: |lam1| <= |lam2| <= |lam3|
    # For dark-on-bright sheet: lam3 is large-magnitude negative (normal to surface)
    # lam1, lam2 ~ 0  (tangent directions along the surface plane)
    abs_a, abs_b, abs_c = np.abs(ev_a), np.abs(ev_b), np.abs(ev_c)
    a_largest  = (abs_a >= abs_b) & (abs_a >= abs_c)
    b_largest  = (abs_b >  abs_a) & (abs_b >= abs_c)
    lam3 = np.where(a_largest, ev_a, np.where(b_largest, ev_b, ev_c))
    a_smallest = (abs_a <= abs_b) & (abs_a <= abs_c)
    b_smallest = (abs_b <  abs_a) & (abs_b <= abs_c)
    lam1 = np.where(a_smallest, ev_a, np.where(b_smallest, ev_b, ev_c))
    lam2 = ev_a + ev_b + ev_c - lam1 - lam3
    del ev_a, ev_b, ev_c, abs_a, abs_b, abs_c, a_largest, b_largest, a_smallest, b_smallest
    # Correct Frangi 1998 plate criterion: Ra = sqrt(lam1^2 + lam2^2) / |lam3|
    # Bug fix: original used sqrt(|lam1*lam2|)/|lam3| which fails to suppress tubes
    # (tube: lam1~0, lam2~lam3 → sqrt(|0*lam3|)/|lam3|=0, wrongly looks like plate)
    # Correct formula: sqrt(lam1^2+lam2^2)/|lam3| → tube gives |lam2|/|lam3|~1 (suppressed) ✓
    abs_l3 = np.abs(lam3)
    R_a = np.where(abs_l3 > 1e-12, np.sqrt(lam1**2 + lam2**2) / abs_l3, 1.0)
    S2 = lam1**2 + lam2**2 + lam3**2;  del lam1, lam2, abs_l3
    c2 = max(float(S2.max()) * 0.25, 1e-10)
    V = np.where(lam3 < 0,
        np.exp(-R_a**2 / (2.0*alpha**2)) * (1.0 - np.exp(-S2 / (2.0*c2))),
        0.0).astype(np.float32)
    del lam3, R_a, S2
    return V * float(sigma**2)   # Lindeberg scale-space normalization


def _frangi_surface_filter(prob, sigmas=(0.5, 1.0, 2.0), alpha=0.5):
    """Multi-scale 3D Frangi plate filter (host baseline technique, +0.019 LB).
    Identifies bright sheet-like structures via multi-scale Hessian eigenvalue analysis.
    Runs independently at each sigma and takes the maximum response (Lindeberg normalized).

    For papyrus surface (bright 2D sheet in 3D volume):
        lambda3 << 0  (large negative: strong curvature across the surface)
        lambda1, lambda2 ~ 0  (small: smooth variation along the surface plane)

    Returns surfaceness map normalized to [0, 1].
    Skips (returns zeros) if volume > 100M voxels to avoid CPU RAM OOM."""
    if prob.size > 100_000_000:
        print(f"    [FRANGI] Volume {prob.size/1e6:.0f}M voxels > 100M limit — skipping")
        return np.zeros_like(prob, dtype=np.float32)
    p_f64 = prob.astype(np.float64)
    best_V = np.zeros_like(prob, dtype=np.float32)
    for sigma in sigmas:
        try:
            V_s = _frangi_surface_filter_single(p_f64, sigma, alpha)
            best_V = np.maximum(best_V, V_s)
            del V_s
        except Exception as _se:
            print(f"    [FRANGI] sigma={sigma} failed: {_se}")
    del p_f64
    vmax = float(best_V.max())
    return (best_V / vmax) if vmax > 1e-10 else best_V


def logit_fuse_ensemble(per_model_probs, model_weights, model_roles=None, thresh_coefs=None):
    """v3 PODIUM: Trimmed-mean logit fusion + learned threshold + catastrophe fallback.
    Upgrade 2: trimmed-mean replaces median (more robust for large ensembles).
    Upgrade 1: learned threshold predictor replaces heuristic grid search."""
    names = list(per_model_probs.keys())
    n = len(names)

    if n == 1:
        fused_logit = np.log(np.clip(next(iter(per_model_probs.values())), 1e-6, 1-1e-6) /
                             np.clip(1 - next(iter(per_model_probs.values())), 1e-6, 1-1e-6))
        # Apply ensemble temperature to single model too
        if CONF_TEMPERATURE > 0 and CONF_TEMPERATURE != 1.0:
            fused_logit = fused_logit / CONF_TEMPERATURE
        prob = 1.0 / (1.0 + np.exp(-np.clip(fused_logit, -50, 50)))
        # Frangi surface filter (single-model path)
        # Blueprint: apply ONLY in uncertainty band [ENS_TL, ENS_TH) — not globally.
        try:
            _smap1 = _frangi_surface_filter(prob, sigmas=(0.5, 1.0, 2.0))
            if _smap1.max() > 1e-6:
                _prob1 = prob.copy()
                _s1_pos = _smap1 > 0.30;  _s1_neg = _smap1 < 0.05
                _bl1 = (_prob1 >= ENS_TL) & (_prob1 < ENS_TH)  # calibrated band, not hardcoded 0.44
                _prob1[_s1_pos & _bl1] = np.maximum(_prob1[_s1_pos & _bl1], ENS_TH + 0.01)
                _prob1[_s1_neg & (_prob1 >= ENS_TL) & (_prob1 < ENS_TH)] *= 0.6
                prob = np.clip(_prob1, 0.0, 1.0).astype(np.float32)
            del _smap1
        except Exception:
            pass
        return postprocess(prob, ENS_TL, ENS_TH, probs_for_dust=prob, verbose=True)

    # Step 1: Risk-aware weight adjustment
    vol_weights = risk_aware_fusion(per_model_probs, model_weights)

    # Step 2: Upgrade 2 — Trimmed-mean logit fusion on risk-filtered active models.
    # BUG FIX: vol_weights was computed above but trimmed_mean_logit_fusion previously
    # received ALL models (including weight=0 excluded ones), making risk_aware_fusion a
    # no-op for the multi-model case. Now filter to active models first, then fuse.
    active_probs = {nm: p for nm, p in per_model_probs.items()
                    if vol_weights.get(nm, 1.0) > 0.01}
    if len(active_probs) == 0:
        active_probs = per_model_probs  # safety: never exclude everything
        print("    [RISK] All models excluded — reverting to full ensemble")
    elif len(active_probs) < n:
        print(f"    [RISK] Active ensemble: {len(active_probs)}/{n} models (excluded {n - len(active_probs)})")

    n_active = len(active_probs)
    if n_active >= 2:
        result = trimmed_mean_logit_fusion(active_probs, trim_frac=0.15)
    else:
        # n_active == 1: weighted average (same as single-model but via vol_weights)
        w_total = sum(vol_weights[nm] for nm in active_probs)
        result = np.zeros_like(next(iter(active_probs.values())))
        for nm in active_probs:
            p = np.clip(active_probs[nm], 1e-6, 1 - 1e-6)
            result += np.log(p / (1 - p)) * (vol_weights[nm] / max(w_total, 1e-8))

    # Step 3: Ensemble temperature (applied after per-model temps already in probs)
    if CONF_TEMPERATURE != 1.0 and CONF_TEMPERATURE > 0:
        result = result / CONF_TEMPERATURE

    fused_prob = 1.0 / (1.0 + np.exp(-np.clip(result, -50, 50)))

    # Step 4: Upgrade 1 — Learned per-volume threshold predictor
    try:
        vol_tl, vol_th = _learned_threshold(fused_prob, thresh_coefs)
        print(f"    [THRESH] Predictor: tl={vol_tl:.3f}, th={vol_th:.3f}")
    except Exception:
        vol_tl, vol_th = ENS_TL, ENS_TH
        print(f"    [THRESH] Fallback: tl={vol_tl:.3f}, th={vol_th:.3f}")

    # Step 4.5: CC-minimizing threshold nudge.
    # If the initial threshold gives many fragments, try small nudges (±0.04, ±0.08)
    # to find a threshold that naturally produces fewer CCs without radically changing FG fraction.
    # Only activates when > 8 CCs detected — otherwise leave learned threshold untouched.
    try:
        _struct26_tn = generate_binary_structure(3, 3)
        _, _n_tn = cc_label((fused_prob > vol_th).astype(bool), structure=_struct26_tn)
        if _n_tn > 8:
            _fg_base_tn = float((fused_prob > vol_th).mean())
            _best_th_tn, _best_tl_tn, _best_n_tn = vol_th, vol_tl, _n_tn
            for _nudge in [-0.04, -0.08, 0.04, 0.08]:
                _th_c = float(np.clip(vol_th + _nudge, 0.28, 0.78))
                _tl_c = float(np.clip(vol_tl + _nudge, 0.14, _th_c - 0.06))
                _, _n_c = cc_label((fused_prob > _th_c).astype(bool), structure=_struct26_tn)
                _fg_c = float((fused_prob > _th_c).mean())
                # Accept only if: fewer CCs AND FG fraction doesn't change > 40%
                _fg_ok = (_fg_base_tn < 0.005) or (abs(_fg_c - _fg_base_tn) < 0.40 * _fg_base_tn)
                if _n_c < _best_n_tn and _fg_ok:
                    _best_n_tn, _best_th_tn, _best_tl_tn = _n_c, _th_c, _tl_c
            if _best_th_tn != vol_th:
                print(f"    [THRESH-NUDGE] CC-opt: {_n_tn}→{_best_n_tn} CCs, "
                      f"th={vol_th:.3f}→{_best_th_tn:.3f}")
                vol_th, vol_tl = _best_th_tn, _best_tl_tn
    except Exception:
        pass  # non-critical

    # Step 5: Uncertainty gating — suppress thin high-uncertainty connectors
    if n >= 2:
        prob_stack = np.stack([per_model_probs[nm] for nm in names], axis=0)
        uncertainty = np.var(prob_stack, axis=0)
        uncertain_mask = (uncertainty > 0.05) & (fused_prob > 0.3) & (fused_prob < 0.7)
        fused_prob[uncertain_mask] *= 0.7  # dampen uncertain connectors

    # Step 5.5: Frangi surface filter (host baseline technique, +~0.019 LB)
    # 3D Hessian eigenvalue analysis: bright plate-like (sheet) voxels get confirmed,
    # non-surface blobs get attenuated. Directly mirrors host postprocessing step.
    try:
        _t_fr = time.time()
        _smap = _frangi_surface_filter(fused_prob, sigmas=(0.5, 1.0, 2.0))
        if _smap.max() > 1e-6:
            _fp2 = fused_prob.copy()
            _surf_pos = _smap > 0.30   # strong plate-like surface response
            _surf_neg = _smap < 0.05   # definitely not surface
            # Blueprint: restrict Frangi boost/attenuation to uncertainty band [vol_tl, vol_th).
            # Outside this band the model is confident — Frangi would add noise, not signal.
            _borderline = (_fp2 >= vol_tl) & (_fp2 < vol_th)  # calibrated band, not hardcoded 0.44
            _fp2[_surf_pos & _borderline] = np.maximum(
                _fp2[_surf_pos & _borderline], vol_th + 0.01)
            # Attenuate: non-surface voxels inside uncertainty band only (conservative).
            _mid_blob = _surf_neg & (_fp2 >= vol_tl) & (_fp2 < vol_th)
            _fp2[_mid_blob] *= 0.6
            fused_prob = np.clip(_fp2, 0.0, 1.0).astype(np.float32)
            print(f"    [FRANGI] {time.time()-_t_fr:.1f}s | "
                  f"surface={float(_surf_pos.mean())*100:.1f}% | "
                  f"boost={int((_surf_pos & _borderline).sum())} | "
                  f"attenuate={int(_mid_blob.sum())}")
            del _fp2, _surf_pos, _surf_neg, _borderline, _mid_blob
        del _smap
    except Exception as _fe:
        print(f"    [FRANGI] Skipped ({type(_fe).__name__}: {_fe})")

    # Step 6: Postprocess with pathology gating
    struct26 = generate_binary_structure(3, 3)
    _, n_cc_pre = cc_label((fused_prob > vol_th).astype(bool), structure=struct26)
    mask = postprocess(fused_prob, vol_tl, vol_th, probs_for_dust=fused_prob,
                       verbose=True, n_cc_pre=n_cc_pre)

    # Step 6.5: Surface gap closure — connect nearby fragments before catastrophe check.
    # Runs only when 2-24 CCs exist (enough fragmentation to need help but not catastrophe).
    # Uses progressive morphological bridging constrained to high-probability regions.
    try:
        struct26_gc = generate_binary_structure(3, 3)
        _, n_before_gc = cc_label(mask.astype(bool), structure=struct26_gc)
        if 2 <= n_before_gc <= 24:
            mask = _close_surface_gaps(mask, fused_prob, max_bridge_r=5, prob_thresh=0.28)
    except Exception as _gce:
        pass  # non-critical, continue to catastrophe check

    # Step 7: Catastrophe fallback
    if model_roles is not None:
        mask = catastrophe_fallback(mask, per_model_probs, model_roles)

    return mask


def compute_auto_weights(model_names, metas_dict):
    """v3: Auto-weights from score-aligned proxy × role × ckpt_type multiplier."""
    raw = {}
    for mn in model_names:
        meta = metas_dict.get(mn, {})
        proxy = meta.get("best_val_score", meta.get("combined_proxy", 0.3))
        role = meta.get("model_role", "balanced")
        role_mult = ROLE_WEIGHTS.get(role, 1.0)
        ckpt_mult = float(meta.get("ckpt_weight_mult", 1.0))  # per-checkpoint type downweight
        raw[mn] = max(float(proxy) * role_mult * ckpt_mult, 0.05)

    vals = np.array([raw[mn] for mn in model_names])
    exp_vals = np.exp(vals - vals.max())
    weights = exp_vals / (exp_vals.sum() + 1e-8)
    weights = weights * len(model_names)
    weights = np.clip(weights, 0.1, None)
    return {mn: float(w) for mn, w in zip(model_names, weights)}


def ensemble_predict(vol_u8, use_tta):
    """Run all models with per-model ROI, then fuse.
    Upgrade 5: volume-difficulty-based adaptive TTA (not model-weakness-based).
    Upgrade 7: per-model temperature applied in logit space before sigmoid."""
    vol_f32 = normalize_volume(vol_u8)
    model_weights = compute_auto_weights(list(models.keys()), metas)

    per_model_logits = {}   # raw logit maps before temperature
    per_model_probs = {}    # temperature-scaled probs

    import threading, collections
    from concurrent.futures import ThreadPoolExecutor

    # GPU-grouped parallel inference: group models by their assigned device.
    # One thread per GPU runs its models serially → no concurrent allocation on the same device.
    # Both GPUs work simultaneously → ~2× wall-clock speedup vs pure serial.
    # Peak VRAM per GPU: model weights (955/764 MB) + one 128³ forward (~500 MB) << 14.56 GB.
    _gpu_groups = collections.defaultdict(list)
    for _mn in models:
        _gpu_groups[str(model_devices.get(_mn, DEVICE))].append(_mn)
    _n_gpu_threads = len(_gpu_groups)
    _logit_lock = threading.Lock()

    def _infer_gpu_group(mnames_for_gpu, use_tta_flag, tag):
        for _mn in mnames_for_gpu:
            _dev = model_devices.get(_mn, DEVICE)
            _m = models[_mn]
            # Move this model alone onto GPU; all others remain in CPU RAM.
            # With 1 model on GPU (191 MB), ~13.9 GB free → 128³ PRECOMP_GEMM workspace
            # (10.12 GB) fits with 3.8 GB to spare. 9-models-resident left only 10.44 GB → OOM.
            _m.to(_dev)
            _roi = tuple(metas.get(_mn, {}).get("patch_size", list(DEFAULT_ROI)))
            # Patch cap reads from _INF_PATCH_CAP_GLOBAL so OOM retry in main loop
            # can reduce it (128→96→64) and re-enter ensemble_predict at lower quality.
            _INF_PATCH_CAP = tuple(_INF_PATCH_CAP_GLOBAL)
            _roi = tuple(min(r, c) for r, c in zip(_roi, _INF_PATCH_CAP))
            _roi_c = tuple(min(r, s) for r, s in zip(_roi, vol_f32.shape))
            _ovlp = INF_OVERLAP  # always use config, not training meta (meta may store lower training overlap)
            _t0 = time.time()
            _lmap = infer_single_model(vol_f32, _m, _roi_c, _ovlp, use_tta_flag)
            _elapsed = time.time() - _t0
            _m.to('cpu')  # return to CPU RAM immediately
            # No synchronize: causes CUDA context poisoning via sticky VMM errors on T4
            # No empty_cache: causes allocator fragmentation; 10.12 GB workspace block stays
            # in PyTorch's caching allocator and is reused cleanly for the next model
            with _logit_lock:
                per_model_logits[_mn] = _lmap
                print(f"    {_mn} [{tag}]: {_elapsed:.1f}s roi={_roi_c} dev={_dev}")

    # Step 1: Scout pass — parallel across GPUs, serial within each GPU
    t0_all = time.time()
    _group_lists = list(_gpu_groups.values())
    with ThreadPoolExecutor(max_workers=_n_gpu_threads) as _pool:
        _futs = [_pool.submit(_infer_gpu_group, grp, False, "scout") for grp in _group_lists]
        for _f in _futs: _f.result()  # re-raise any exceptions

    # Step 2: TTA pass — always applied (TTA_DIFFICULTY_THRESH=0.0)
    did_tta = False
    if use_tta:
        w_total = sum(model_weights[mn] for mn in models)
        mean_logit = sum(per_model_logits[mn] * (model_weights[mn] / max(w_total, 1e-8))
                         for mn in models)
        difficulty = _difficulty_score(mean_logit)
        print(f"    [TTA] Volume difficulty: {difficulty:.3f} (thresh={TTA_DIFFICULTY_THRESH})")

        if difficulty > TTA_DIFFICULTY_THRESH and budget_ok(MAX_PRED_HOURS - 0.3):
            print(f"    [TTA] Applying full TTA — parallel across {_n_gpu_threads} GPU(s)")
            with ThreadPoolExecutor(max_workers=_n_gpu_threads) as _pool:
                _futs = [_pool.submit(_infer_gpu_group, grp, True, "+TTA") for grp in _group_lists]
                for _f in _futs: _f.result()
            did_tta = True
        else:
            print(f"    [TTA] Skipping (difficulty {difficulty:.3f} ≤ thresh {TTA_DIFFICULTY_THRESH})")

    # Step 3: Upgrade 7 — Apply per-model temperature in logit space, then sigmoid once
    for mname in models:
        logit_map = per_model_logits[mname]
        temp_cal = float(metas.get(mname, {}).get("temperature", 1.0))
        if temp_cal > 0 and temp_cal != 1.0:
            logit_map = logit_map / temp_cal   # scale logits (not probs)
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit_map, -50, 50)))
        per_model_probs[mname] = prob
        w = model_weights[mname]
        tta_tag = " +TTA" if did_tta else ""
        print(f"    {mname}: T={temp_cal:.2f}, w={w:.3f}{tta_tag}")

    # Step 4: Collect ensemble threshold predictor coefficients from metas
    # Use the most recent (highest-proxy) model's coefficients if available
    thresh_coefs = None
    best_proxy = -1.0
    for mn in models:
        meta_mn = metas.get(mn, {})
        proxy = float(meta_mn.get("best_val_score", meta_mn.get("combined_proxy", 0.0)))
        if proxy > best_proxy and "thresh_coefs" in meta_mn:
            thresh_coefs = meta_mn["thresh_coefs"]
            best_proxy = proxy

    # Step 5: Fuse with trimmed-mean logit fusion + learned threshold
    roles = {mn: metas.get(mn, {}).get("model_role", "generalist") for mn in models}
    return logit_fuse_ensemble(per_model_probs, model_weights,
                               model_roles=roles, thresh_coefs=thresh_coefs)
'''


CELL_PRED_MAIN = r'''
# ============================================================
# Main Inference Loop v3
# ============================================================
OUT_DIR = "/kaggle/working/preds"
os.makedirs(OUT_DIR, exist_ok=True)
ZIP_PATH = "/kaggle/working/submission.zip"

submission_meta = {
    "version": "v3.0",
    "models": list(models.keys()),
    "model_paths": {k: MODEL_PATHS[k] for k in models},
    "ens_mode": ENS_MODE,
    "ens_tl": ENS_TL, "ens_th": ENS_TH,
    "conf_temperature": CONF_TEMPERATURE,
    "default_roi": list(DEFAULT_ROI),
    "overlap": INF_OVERLAP,
    "pp_dust_min_3d": PP_DUST_MIN_3D,
    "pp_hole_max_2d": PP_HOLE_MAX_2D,
    "pp_bridge_kill": PP_BRIDGE_KILL,
    "auto_drop_thresh": AUTO_DROP_THRESH,
    "pytorch_version": torch.__version__,
    "metas": {k: {kk: vv for kk, vv in v.items()
                   if isinstance(vv, (int, float, str, bool))}
              for k, v in metas.items() if k in models},
}

n_test = len(test_ids)
time_per_vol_budget = (MAX_PRED_HOURS * 3600 - 300) / max(n_test, 1)
n_models = len(models)
print(f"[INF] {n_test} volumes, {time_per_vol_budget:.0f}s/vol budget, "
      f"{n_models} variants, overlap={INF_OVERLAP}, tta_thresh={TTA_DIFFICULTY_THRESH}")

# 6-level degradation ladder — applied proactively before each volume when
# rolling average × remaining_vols > remaining_budget × 0.80.
# Level 0 = best quality; each step sacrifices quality to preserve timing.
_DEGRADE_LADDER = [
    # (overlap, tta_eligible, patch_cap)
    (0.50,  True,  128),   # L0: full quality — TTA on difficult vols (gated by thresh)
    (0.50,  False, 128),   # L1: TTA completely off — 7x speedup vs L0
    (0.375, False, 128),   # L2: 40% fewer patches vs L1
    (0.25,  False, 128),   # L3: 64% fewer patches vs L0
    (0.25,  False,  96),   # L4: smaller patch window (lower VRAM, fewer patches)
    (0.25,  False,  64),   # L5: minimal patch window — last resort before empty
]
_degrade_level = [0]
_vol_rolling_times = []   # actual seconds per completed volume (last ≤6 kept for avg)

current_tta = USE_TTA
# Initial TTA check: if 5 variants × ~60s each already exceeds budget, start at L1.
_init_est_no_tta = n_models * 60
if _init_est_no_tta > time_per_vol_budget * 0.9:
    current_tta = False
    _degrade_level[0] = 1
    print(f"[INF] Starting at L1 (TTA off): est {_init_est_no_tta:.0f}s/vol "
          f"({n_models} variants×60s) vs budget {time_per_vol_budget:.0f}s")

# Pre-scan volume shapes so we can write correctly-shaped empty predictions
# for any volumes that can't be processed within the time budget.
# Uses tifffile metadata read (no decompression) — fast even for large volumes.
print("[INF] Pre-scanning volume shapes...")
vol_shapes = {}
for _vid, _vpath in zip(test_ids, test_paths):
    try:
        import tifffile as _tff
        with _tff.TiffFile(_vpath) as _tf:
            _pg = _tf.pages
            vol_shapes[_vid] = (len(_pg), int(_pg[0].shape[0]), int(_pg[0].shape[1]))
    except Exception:
        try:
            _v = read_tif(_vpath)
            vol_shapes[_vid] = _v.shape
            del _v
        except Exception:
            pass
print(f"[INF] Shape pre-scan: {len(vol_shapes)}/{n_test} volumes")

completed = 0
per_vol_stats = []

def _write_empty_vol(zf, vid, shape, reason="time_budget"):
    """Write a correctly-shaped all-zeros prediction into the zip."""
    try:
        empty = np.zeros(shape, dtype=np.uint8)
        _ep = os.path.join(OUT_DIR, f"{vid}.tif")
        write_tif(_ep, empty)
        zf.write(_ep, f"{vid}.tif")
        per_vol_stats.append({"id": vid, "time_s": 0, "fg_frac": 0.0, "error": reason})
        print(f"  [EMPTY] {vid}: wrote zero prediction ({reason})")
    except Exception as _ee:
        print(f"  [EMPTY-FAIL] {vid}: {_ee}")

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for vi, (vid, vpath) in enumerate(zip(test_ids, test_paths)):
        if not budget_ok(MAX_PRED_HOURS - 0.1):
            print(f"\n[TIME] Budget near limit. {completed}/{n_test} done.")
            print(f"[TIME] Writing empty predictions for {n_test - vi} remaining volumes.")
            for _vid, _vpath in zip(test_ids[vi:], test_paths[vi:]):
                if _vid in vol_shapes:
                    _write_empty_vol(zf, _vid, vol_shapes[_vid])
                else:
                    print(f"  [EMPTY-SKIP] {_vid}: no shape info, skipping")
            break

        t0 = time.time()
        print(f"\n[{vi+1}/{n_test}] {vid}...")
        orig_shape = None

        # Proactive degradation ladder: step down BEFORE attempting this volume if
        # rolling_avg_time × remaining_vols > remaining_budget × 0.80.
        # Uses rolling avg of last 6 volumes (more stable than single last-vol estimate).
        remaining = n_test - vi - 1
        if remaining > 0 and _vol_rolling_times:
            _rolling_avg = sum(_vol_rolling_times[-6:]) / len(_vol_rolling_times[-6:])
            _remain_budget = (MAX_PRED_HOURS - elapsed_h()) * 3600
            _est_remain = remaining * _rolling_avg
            if _est_remain > _remain_budget * 0.80 and _degrade_level[0] < len(_DEGRADE_LADDER) - 1:
                _next_level = _degrade_level[0] + 1
                _ovlp, _tta, _pcap = _DEGRADE_LADDER[_next_level]
                print(f"  [LADDER L{_degrade_level[0]}→{_next_level}] "
                      f"est {_est_remain:.0f}s > budget {_remain_budget*0.80:.0f}s: "
                      f"overlap={_ovlp}, tta={_tta}, pcap={_pcap}")
                INF_OVERLAP = _ovlp
                current_tta = _tta
                _INF_PATCH_CAP_GLOBAL[0] = _INF_PATCH_CAP_GLOBAL[1] = _INF_PATCH_CAP_GLOBAL[2] = _pcap
                _degrade_level[0] = _next_level

        # OOM-resilient inference: degrade patch cap and retry on CUDA OOM (up to 2 retries).
        # Ladder: 128³ (10.1 GB workspace) → 96³ (4.3 GB) → 64³ (1.3 GB).
        _oom_retries = 0
        _inference_done = False
        mask = None
        while not _inference_done:
            try:
                vol = read_tif(vpath)
                orig_shape = vol.shape
                print(f"  Shape: {orig_shape}, dtype: {vol.dtype}, cap={tuple(_INF_PATCH_CAP_GLOBAL)}")

                mask = ensemble_predict(vol, current_tta)

                assert mask.shape == orig_shape
                mask = mask.astype(np.uint8)
                mask = np.clip(mask, 0, 1)
                _inference_done = True

            except RuntimeError as _oom_e:
                _oom_msg = str(_oom_e).lower()
                if ("out of memory" in _oom_msg or "cuda" in _oom_msg) and _oom_retries < 2:
                    _oom_retries += 1
                    _new_cap = max(64, _INF_PATCH_CAP_GLOBAL[0] - 32)
                    print(f"  [OOM] Retry {_oom_retries}/2: patch cap {_INF_PATCH_CAP_GLOBAL[0]}→{_new_cap}, TTA off")
                    _INF_PATCH_CAP_GLOBAL[0] = _INF_PATCH_CAP_GLOBAL[1] = _INF_PATCH_CAP_GLOBAL[2] = _new_cap
                    current_tta = False
                    torch.cuda.empty_cache()
                    gc.collect()
                    orig_shape = None   # will be set on next read_tif
                else:
                    # Non-OOM error or retries exhausted → fall through to except
                    print(f"  [ERROR] {vid}: {_oom_e}")
                    traceback.print_exc()
                    _inference_done = True   # exit while loop, mask=None

            except Exception as e:
                print(f"  [ERROR] {vid}: {e}")
                traceback.print_exc()
                _inference_done = True   # exit while loop, mask=None

        # Write result or shape-correct empty fallback
        try:
            _shape_for_fallback = orig_shape if orig_shape is not None else vol_shapes.get(vid)
            if mask is not None:
                fg_frac = mask.sum() / mask.size
                if fg_frac == 0:
                    print(f"  [WARN] Empty prediction for {vid}!")
                out_path = os.path.join(OUT_DIR, f"{vid}.tif")
                write_tif(out_path, mask)
                zf.write(out_path, f"{vid}.tif")
                dt = time.time() - t0
                completed += 1
                _vol_rolling_times.append(dt)
                if len(_vol_rolling_times) > 12:   # keep last 12 for rolling avg
                    _vol_rolling_times.pop(0)
                per_vol_stats.append({"id": vid, "time_s": dt, "fg_frac": float(fg_frac),
                                      "shape": list(orig_shape), "degrade_level": _degrade_level[0]})
                print(f"  Done: {dt:.1f}s, FG={fg_frac:.4f}, L={_degrade_level[0]}")
            else:
                # Inference failed — write correctly-shaped empty prediction
                if _shape_for_fallback is not None:
                    _write_empty_vol(zf, vid, _shape_for_fallback, reason="inference_failed")
                    completed += 1
                else:
                    print(f"  [ERROR] {vid}: no shape info for fallback — volume skipped")
                    per_vol_stats.append({"id": vid, "time_s": 0, "fg_frac": 0.0,
                                          "error": "no_shape_for_fallback"})
        except Exception as _write_e:
            print(f"  [ERROR] Write failed for {vid}: {_write_e}")

        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

submission_meta["completed"] = completed
submission_meta["total_time_h"] = elapsed_h()
submission_meta["per_volume"] = per_vol_stats
meta_out = "/kaggle/working/submission_meta.json"
with open(meta_out, "w") as f:
    json.dump(submission_meta, f, indent=2)

print(f"\n[DONE] {completed}/{n_test} volumes in {elapsed_h():.2f}h")
print(f"  Submission: {ZIP_PATH}")
'''


CELL_PRED_VALIDATE = r'''
# ============================================================
# Submission Validation
# ============================================================
assert os.path.exists(ZIP_PATH), f"submission.zip not found"

with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    names = zf.namelist()
    assert len(names) > 0, "submission.zip is empty"
    print(f"[ZIP] {len(names)} files in submission.zip")
    for n in names[:5]:
        info = zf.getinfo(n)
        print(f"  {n}: {info.file_size/1e6:.1f} MB")

    for n in names[:3]:
        with zf.open(n) as fp:
            header = fp.read(4)
            assert header[:2] in (b'II', b'MM'), f"{n}: Not a valid TIFF"

    zip_basenames = set(os.path.basename(n).replace(".tif", "") for n in names)
    missing = [tid for tid in test_ids if tid not in zip_basenames]
    if missing:
        print(f"  [WARN] Missing predictions for {len(missing)} volumes")
    else:
        print(f"  All {len(test_ids)} test volumes have predictions")

sz = os.path.getsize(ZIP_PATH) / 1e6
print(f"\n[OK] submission.zip: {sz:.1f} MB, {len(names)} volumes")
print(f"[OK] Total time: {elapsed_h():.2f}h")

if os.path.exists("/kaggle/working/submission_meta.json"):
    with open("/kaggle/working/submission_meta.json") as f:
        sm = json.load(f)
    print(f"\n[META] Version: {sm.get('version')}")
    print(f"[META] Models: {sm.get('models')}")
    stats = sm.get("per_volume", [])
    if stats:
        fg_fracs = [s["fg_frac"] for s in stats if "fg_frac" in s]
        if fg_fracs:
            print(f"[META] FG fraction: min={min(fg_fracs):.4f}, max={max(fg_fracs):.4f}, "
                  f"mean={np.mean(fg_fracs):.4f}")

gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
'''


# ============================================================
# ASSEMBLY
# ============================================================

def build_train_notebook(model_name, seed, fold_idx, patch_size,
                         w_ce, w_dice, w_msr, w_bnd, w_topo=0.3,
                         w_surfdist=0.4, w_sdf=0.4, w_cldice=0.0,
                         w_compreg=0.0, w_topoguide=0.0,
                         w_gapneg=0.5, w_tversky=0.0,
                         tversky_alpha=0.3, tversky_beta=0.7, tversky_gamma=1.0,
                         epochs=300, grad_accum=2, ema_decay=0.999,
                         warmup_epochs=5, lr_floor=3e-4, label_smooth=0.02,
                         aug_scale=1.0, ring_width=3, ignore_reject=0.70,
                         n_epoch_vols=8, initial_lr=0.01, grad_clip=5.0,
                         p_boundary=0.40, p_ring=0.20, p_fg=0.25, p_bg=0.10,
                         p_hard=0.05, p_bridge_risk=0.05, p_thin_risk=0.05,
                         p_curvature=0.05, model_role="balanced"):
    cells = [
        CELL_TRAIN_ENV,
        cell_train_config(model_name, seed, fold_idx, patch_size,
                         w_ce, w_dice, w_msr, w_bnd, w_topo,
                         w_surfdist, w_sdf, w_cldice, w_compreg, w_topoguide,
                         w_gapneg, w_tversky,
                         tversky_alpha, tversky_beta, tversky_gamma,
                         epochs, grad_accum, ema_decay, warmup_epochs,
                         lr_floor, label_smooth, aug_scale,
                         ring_width, ignore_reject, n_epoch_vols,
                         initial_lr, grad_clip,
                         p_boundary, p_ring, p_fg, p_bg, p_hard, p_bridge_risk,
                         p_thin_risk, p_curvature, model_role),
        CELL_DATA_DISCOVERY,
        CELL_ARCHITECTURE,
        CELL_LOSS,
        CELL_DATASET,
        CELL_TRAINING,
        CELL_CALIBRATION,
    ]
    return make_nb(cells)


def build_predict_notebook():
    cells = [
        CELL_PRED_ENV,
        CELL_PRED_CONFIG,
        CELL_PRED_ARCHITECTURE,
        CELL_PRED_LOAD,
        CELL_PRED_POSTPROCESS,
        CELL_PRED_INFERENCE,
        CELL_PRED_MAIN,
        CELL_PRED_VALIDATE,
    ]
    return make_nb(cells)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print(f"Output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print()

    # v3: Specialist ensemble (brain doc: decompose failure modes)
    # A = Robust Generalist (anchor, conservative, best calibration)
    # B = Anti-Merge Specialist (strong topo/gap-neg, bridge-risk sampling)
    # C = Surface Polisher (strong SDF/SurfDist, high-res emphasis)
    configs = [
        {
            # SUPER MODEL X — Single unified model targeting 0.60 LB
            # Combines A's topo strength + B's bridge-killing + C's surface SDF quality.
            # Single model → 4 checkpoint variants (best_score/surfdice/voi/topo + swa) = 5-way
            # ensemble at inference with zero extra training cost.
            #
            # Loss design rationale (competition metric = 0.30×Topo + 0.35×SurfDice + 0.35×VOI):
            #   CE/Dice 0.8 — moderate base; headroom for topology/surface losses
            #   surfdist 0.40 — max safe with late_mult=1.5 (effective 0.60 in late phase)
            #   sdf 0.55 — strong surface shape signal (C had 0.6, A had 0.4; midpoint)
            #   topo 0.40 — A's proven TopoScore driver; late effective = 0.52
            #   gapneg 0.50 — B's bridge-killing weight; late effective = 0.65
            #   cldice 0.55 — strong skeleton continuity; late effective = 0.72
            #   compreg 0.20 — A's anti-fragmentation weight
            #   tversky (0.35/0.65) — recall-biased: penalise FN > FP (preserve sheets)
            #
            # Sampling: surface-first (0.35 boundary + 0.20 bridge-risk = 0.55 metric-critical)
            # n_epoch_vols=8 — halved from 16 to prevent OOM at late-phase K surge
            "filename": "v3_train_model_x.ipynb",
            "model_name": "model_x",
            "model_role": "balanced",
            "seed": 42, "fold_idx": 0,
            "patch_size": (192, 192, 192),
            "w_ce": 0.8, "w_dice": 0.8, "w_msr": 0.50, "w_bnd": 0.0, "w_topo": 0.40,
            "w_surfdist": 0.50, "w_sdf": 0.55, "w_cldice": 0.40,  # 0.50/0.40: SurfDice is 35% of metric
            "w_compreg": 0.30, "w_topoguide": 0.0,
            "w_gapneg": 0.50, "w_tversky": 0.30,
            "tversky_alpha": 0.35, "tversky_beta": 0.65, "tversky_gamma": 1.0,
            "epochs": 300,
            "initial_lr": 0.01, "grad_clip": 5.0,
            "grad_accum": 2, "ema_decay": 0.9995,
            "warmup_epochs": 5, "lr_floor": 3e-4, "label_smooth": 0.01,
            "aug_scale": 1.3, "ring_width": 3, "ignore_reject": 0.70,
            "n_epoch_vols": 8,   # was 16; K=16 at late phase transition caused OOM (2 workers × 16 large vols)
            "p_boundary": 0.35, "p_ring": 0.10, "p_fg": 0.15,
            "p_bg": 0.05, "p_hard": 0.05, "p_bridge_risk": 0.20, "p_thin_risk": 0.05,
            "p_curvature": 0.05,
        },
    ]

    for cfg in configs:
        fname = cfg.pop("filename")
        print(f"Building {fname}...")
        nb = build_train_notebook(**cfg)
        save_nb(nb, os.path.join(OUTPUT_DIR, fname))

    print(f"\nBuilding v3_predict_ensemble.ipynb...")
    nb = build_predict_notebook()
    save_nb(nb, os.path.join(OUTPUT_DIR, "v3_predict_ensemble.ipynb"))

    print("\n" + "="*60)
    print("All notebooks generated (v3.0 - Metric-Aligned Rewrite)")
    print("="*60)
    print("\nv3.0 changes (brain doc aligned):")
    print("  A) Aux heads: SDF + topology guidance on decoder")
    print("  B) 6 core losses: CE+Dice+SDF+SurfDist+MSR+TopoBridge (phase-gated)")
    print("  C) 3-phase schedule: early(0-25%) / mid(25-55%) / late(55-100%, 45% at full res)")
    print("  D) LR floor=3e-4 + restart at transitions + SWA in final 10%")
    print("  E) Fixed hard mining (actual vol index + actual center coords)")
    print("  F) Bridge-risk + split-risk sampling + SDF/topo GT precompute")
    print("  G) Mini-leaderboard validation every epoch in late phase")
    print("  H) Score-aligned calibration: 0.30*Topo + 0.35*SurfDice + 0.35*VOI")
    print("  I) Risk-aware fusion + trimmed mean + median logit options")
    print("  J) Bridge cutting via thickness+skeleton + smart dust removal + 26-CC")
    print("\nModel configs:")
    for cfg in configs:
        mn = cfg["model_name"]
        role = cfg.get("model_role", "balanced")
        ps = cfg["patch_size"]
        lr = cfg.get("initial_lr", 0.01)
        gc_val = cfg.get("grad_clip", 5.0)
        print(f"  {mn} ({role}): patch={ps}, lr={lr}, clip={gc_val}")
        print(f"    loss: CE={cfg['w_ce']},D={cfg['w_dice']},MSR={cfg['w_msr']},"
              f"BND={cfg['w_bnd']},TOPO={cfg['w_topo']},"
              f"SD={cfg['w_surfdist']},SDF={cfg['w_sdf']},"
              f"clD={cfg['w_cldice']},CR={cfg['w_compreg']},TG={cfg['w_topoguide']}")
