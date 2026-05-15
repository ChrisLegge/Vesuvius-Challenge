"""
Fix sliding_window_predict in Model B and C calibration cells:
Add Gaussian blending (no @torch.no_grad decorator version).
"""
import json

BASE = r"C:\Users\aryaa\Documents\Vesuvius Notebooks"

notebooks = [
    f"{BASE}\\2_train_model_b.ipynb",
    f"{BASE}\\3_train_model_c.ipynb",
]

# The B/C calibration cells have a compact version without decorator/docstring
OLD_SWP_BC = '''def sliding_window_predict(model, volume, patch_size, overlap=0.25, num_classes=2):
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

NEW_SWP_BC = '''def sliding_window_predict(model, volume, patch_size, overlap=0.25, num_classes=2):
    """Sliding window inference with Gaussian blending."""
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd = max(1, int(pd * (1 - overlap)))
    sh = max(1, int(ph * (1 - overlap)))
    sw = max(1, int(pw * (1 - overlap)))

    # Gaussian importance map for smooth blending
    sigma_scale = 0.125
    _maps = []
    for s in (pd, ph, pw):
        ax = np.arange(s, dtype=np.float32)
        g = np.exp(-0.5 * ((ax - s / 2) / max(s * sigma_scale, 1e-6)) ** 2)
        _maps.append(g)
    gauss = _maps[0][:, None, None] * _maps[1][None, :, None] * _maps[2][None, None, :]
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

for fp in notebooks:
    name = fp.split("\\")[-1]
    with open(fp, "r", encoding="utf-8") as f:
        nb = json.load(f)

    changed = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])

        if OLD_SWP_BC in src:
            src = src.replace(OLD_SWP_BC, NEW_SWP_BC)
            lines = src.split("\n")
            cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
            changed = True
            print(f"  [OK] {name}: sliding_window_predict -> Gaussian blending")

    if changed:
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        with open(fp, "r", encoding="utf-8") as f:
            json.load(f)
        print(f"  [SAVED] {name}")
    else:
        print(f"  [SKIP] {name}: no match")

print("\n[DONE]")
