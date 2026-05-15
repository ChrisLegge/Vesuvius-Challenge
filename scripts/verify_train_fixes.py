"""Verify fixes #3 and #4 are applied correctly."""
import json

BASE = r"C:\Users\aryaa\Documents\Vesuvius Notebooks"
ok = True

# Fix #3: Model A validate_full_volume uses hysteresis (not argmax)
with open(f"{BASE}\\1_train_model_a.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "def validate_full_volume" in src:
        if "argmax" in src:
            print("[FAIL] Model A validate_full_volume still uses argmax!")
            ok = False
        elif "binary_propagation" in src and "t_high" in src:
            print("[OK] Model A validate_full_volume uses hysteresis")
        else:
            print("[WARN] Model A validate_full_volume: unexpected content")
        break
else:
    print("[WARN] Model A: validate_full_volume not found")

# Fix #4: All 3 notebooks have Gaussian blending in sliding_window_predict
for fname in ["1_train_model_a.ipynb", "2_train_model_b.ipynb", "3_train_model_c.ipynb"]:
    with open(f"{BASE}\\{fname}", "r", encoding="utf-8") as f:
        nb = json.load(f)

    found_swp = False
    has_gauss = False
    has_old_count = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "def sliding_window_predict" in src:
            found_swp = True
            if "gauss" in src and "weight_sum" in src:
                has_gauss = True
            if "count = np.zeros" in src:
                has_old_count = True

    if not found_swp:
        print(f"[WARN] {fname}: sliding_window_predict not found")
    elif has_gauss and not has_old_count:
        print(f"[OK] {fname}: sliding_window_predict has Gaussian blending")
    elif has_gauss and has_old_count:
        print(f"[WARN] {fname}: has BOTH Gaussian and old count (multiple copies?)")
    else:
        print(f"[FAIL] {fname}: sliding_window_predict missing Gaussian blending")
        ok = False

if ok:
    print("\n[ALL CHECKS PASSED]")
else:
    print("\n[SOME CHECKS FAILED]")
