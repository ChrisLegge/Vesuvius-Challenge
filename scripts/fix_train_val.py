"""
Fix training validation in all 3 notebooks:
1. Add Gaussian blending to sliding_window_predict (all 3)
2. Replace argmax with hysteresis in validate_full_volume (Model A only)
"""
import json

BASE = r"C:\Users\aryaa\Documents\Vesuvius Notebooks"

notebooks = [
    f"{BASE}\\1_train_model_a.ipynb",
    f"{BASE}\\2_train_model_b.ipynb",
    f"{BASE}\\3_train_model_c.ipynb",
]

# ============================================================
# Old sliding_window_predict (same in all 3 notebooks)
# ============================================================
OLD_SWP = '''@torch.no_grad()
def sliding_window_predict(model, volume, patch_size, overlap=0.25, num_classes=2):
    """Sliding window inference on single volume. Returns logits (C,D,H,W)."""
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd = max(1, int(pd * (1 - overlap)))
    sh = max(1, int(ph * (1 - overlap)))
    sw = max(1, int(pw * (1 - overlap)))

    logits_sum = np.zeros((num_classes, D, H, W), dtype=np.float32)
    count = np.zeros((1, D, H, W), dtype=np.float32)

    d_starts = list(range(0, max(1, D - pd + 1), sd))
    h_starts = list(range(0, max(1, H - ph + 1), sh))
    w_starts = list(range(0, max(1, W - pw + 1), sw))
    if d_starts[-1] + pd < D: d_starts.append(max(0, D - pd))
    if h_starts[-1] + ph < H: h_starts.append(max(0, H - ph))
    if w_starts[-1] + pw < W: w_starts.append(max(0, W - pw))

    for d0 in d_starts:
        for h0 in h_starts:
            for w0 in w_starts:
                patch = volume[d0:d0+pd, h0:h0+ph, w0:w0+pw]
                inp = torch.from_numpy(patch[None, None]).float().to(DEVICE)
                with autocast("cuda"):
                    out = model(inp, deep_supervision=False)
                logit = out.squeeze(0).cpu().numpy()
                logits_sum[:, d0:d0+pd, h0:h0+ph, w0:w0+pw] += logit
                count[:, d0:d0+pd, h0:h0+ph, w0:w0+pw] += 1.0

    return logits_sum / np.maximum(count, 1.0)'''

# ============================================================
# New sliding_window_predict with Gaussian blending
# ============================================================
NEW_SWP = '''@torch.no_grad()
def sliding_window_predict(model, volume, patch_size, overlap=0.25, num_classes=2):
    """Sliding window inference with Gaussian blending. Returns logits (C,D,H,W)."""
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd = max(1, int(pd * (1 - overlap)))
    sh = max(1, int(ph * (1 - overlap)))
    sw = max(1, int(pw * (1 - overlap)))

    # Gaussian importance map for smooth blending
    sigma_scale = 0.125
    maps = []
    for s in (pd, ph, pw):
        ax = np.arange(s, dtype=np.float32)
        g = np.exp(-0.5 * ((ax - s / 2) / max(s * sigma_scale, 1e-6)) ** 2)
        maps.append(g)
    gauss = maps[0][:, None, None] * maps[1][None, :, None] * maps[2][None, None, :]
    gauss = np.clip(gauss, 1e-6, None)

    logits_sum = np.zeros((num_classes, D, H, W), dtype=np.float32)
    weight_sum = np.zeros((D, H, W), dtype=np.float32)

    d_starts = list(range(0, max(1, D - pd + 1), sd))
    h_starts = list(range(0, max(1, H - ph + 1), sh))
    w_starts = list(range(0, max(1, W - pw + 1), sw))
    if d_starts[-1] + pd < D: d_starts.append(max(0, D - pd))
    if h_starts[-1] + ph < H: h_starts.append(max(0, H - ph))
    if w_starts[-1] + pw < W: w_starts.append(max(0, W - pw))

    for d0 in d_starts:
        for h0 in h_starts:
            for w0 in w_starts:
                patch = volume[d0:d0+pd, h0:h0+ph, w0:w0+pw]
                inp = torch.from_numpy(patch[None, None]).float().to(DEVICE)
                with autocast("cuda"):
                    out = model(inp, deep_supervision=False)
                logit = out.squeeze(0).cpu().numpy()
                logits_sum[:, d0:d0+pd, h0:h0+ph, w0:w0+pw] += logit * gauss[None]
                weight_sum[d0:d0+pd, h0:h0+ph, w0:w0+pw] += gauss

    weight_sum = np.maximum(weight_sum, 1e-8)
    return logits_sum / weight_sum[None]'''

# ============================================================
# Old validate_full_volume (Model A only)
# ============================================================
OLD_VFV = '''@torch.no_grad()
def validate_full_volume(model, val_vols, patch_size, n_vols=3, overlap=0.25):
    """Full-volume validation: per-volume competition scores + mean + min."""
    model.eval()
    eval_vols = val_vols[:min(n_vols, len(val_vols))]
    vol_results = []

    for info in eval_vols:
        vol, lbl = _load_and_norm(info, patch_size)
        logits = sliding_window_predict(model, vol, patch_size, overlap)
        pred = logits.argmax(axis=0)
        valid = (lbl != 255)
        pred_m = ((pred == 1) & valid).astype(np.uint8)
        gt_m = ((lbl == 1) & valid).astype(np.uint8)
        score, sd, vi, ts = competition_score(pred_m, gt_m)
        vol_results.append({
            'id': info['id'], 'score': score, 'sd': sd, 'voi': vi, 'topo': ts,
        })

    model.train()
    scores = [r['score'] for r in vol_results]
    return {
        'mean': float(np.mean(scores)),
        'min': float(np.min(scores)),
        'per_vol': vol_results,
    }'''

# ============================================================
# New validate_full_volume with hysteresis (Model A only)
# ============================================================
NEW_VFV = '''@torch.no_grad()
def validate_full_volume(model, val_vols, patch_size, n_vols=3, overlap=0.25,
                         t_low=0.50, t_high=0.75):
    """Full-volume validation with hysteresis thresholding (matches inference)."""
    model.eval()
    eval_vols = val_vols[:min(n_vols, len(val_vols))]
    vol_results = []
    struct26 = ndi.generate_binary_structure(3, 3)

    for info in eval_vols:
        vol, lbl = _load_and_norm(info, patch_size)
        logits = sliding_window_predict(model, vol, patch_size, overlap)
        probs_fg = np_softmax(logits, axis=0)[1]
        seeds = probs_fg >= t_high
        candidates = probs_fg >= t_low
        if seeds.any():
            pred = ndi.binary_propagation(seeds, mask=candidates, structure=struct26)
        else:
            pred = candidates  # fallback: use t_low candidates
        valid = (lbl != 255)
        pred_m = (pred & valid).astype(np.uint8)
        gt_m = ((lbl == 1) & valid).astype(np.uint8)
        score, sd, vi, ts = competition_score(pred_m, gt_m)
        vol_results.append({
            'id': info['id'], 'score': score, 'sd': sd, 'voi': vi, 'topo': ts,
        })
        del vol, lbl, logits, probs_fg, seeds, candidates, pred
        torch.cuda.empty_cache()

    model.train()
    scores = [r['score'] for r in vol_results]
    return {
        'mean': float(np.mean(scores)),
        'min': float(np.min(scores)),
        'per_vol': vol_results,
    }'''

# ============================================================
# Apply patches
# ============================================================
for fp in notebooks:
    name = fp.split("\\")[-1]
    with open(fp, "r", encoding="utf-8") as f:
        nb = json.load(f)

    changed = False
    is_model_a = "model_a" in name

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])

        # Fix 1: Gaussian blending in sliding_window_predict
        if OLD_SWP in src:
            src = src.replace(OLD_SWP, NEW_SWP)
            changed = True
            print(f"  [OK] {name}: sliding_window_predict -> Gaussian blending")

        # Fix 2: hysteresis in validate_full_volume (Model A only)
        if is_model_a and OLD_VFV in src:
            src = src.replace(OLD_VFV, NEW_VFV)
            changed = True
            print(f"  [OK] {name}: validate_full_volume -> hysteresis")

        # Write back
        if changed:
            lines = src.split("\n")
            cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]

    if changed:
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        # Verify JSON valid
        with open(fp, "r", encoding="utf-8") as f:
            json.load(f)
        print(f"  [SAVED] {name}")
    else:
        print(f"  [SKIP] {name}: no matches found")

print("\n[DONE]")
