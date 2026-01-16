import numpy as np
from sklearn.metrics import f1_score

class_names = {
    0: "Squat (C)", 1: "Squat WT", 2: "Squat FL",
    3: "Ext (C)",   4: "Ext NF",   5: "Ext LL",
    6: "Gait (C)",  7: "Gait NF",  8: "Gait HA"
}

def per_class_f1(y_true, y_pred, n_classes=9):
    return f1_score(
        y_true, y_pred,
        labels=np.arange(n_classes),
        average=None,
        zero_division=0
    )

def bootstrap_ci_per_class(y_true, y_pred, B=1000, seed=0, n_classes=9):
    rng = np.random.default_rng(seed)
    N = len(y_true)
    samples = np.zeros((B, n_classes))

    for b in range(B):
        idx = rng.integers(0, N, size=N)
        samples[b] = per_class_f1(y_true[idx], y_pred[idx], n_classes=n_classes)

    point = per_class_f1(y_true, y_pred, n_classes=n_classes)
    lo = np.percentile(samples, 2.5, axis=0)
    hi = np.percentile(samples, 97.5, axis=0)
    return point, lo, hi

def bootstrap_ci_delta(y_true, y_pred_a, y_pred_b, B=1000, seed=0, n_classes=9):
    rng = np.random.default_rng(seed)
    N = len(y_true)
    deltas = np.zeros((B, n_classes))

    for b in range(B):
        idx = rng.integers(0, N, size=N)
        deltas[b] = per_class_f1(y_true[idx], y_pred_a[idx], n_classes) - per_class_f1(y_true[idx], y_pred_b[idx], n_classes)

    point = per_class_f1(y_true, y_pred_a, n_classes) - per_class_f1(y_true, y_pred_b, n_classes)
    lo = np.percentile(deltas, 2.5, axis=0)
    hi = np.percentile(deltas, 97.5, axis=0)
    return point, lo, hi

# load y_test (it is the same for all models)
y_test = np.load("models/fusion_late_y_test.npy")

# load predictions
preds = {
    "IMU": np.load("models/imu_y_pred.npy"),
    "sEMG": np.load("models/emg_y_pred.npy"),
    "Early": np.load("models/fusion_early_y_pred.npy"),
    "Late": np.load("models/fusion_late_y_pred.npy"),
    "Hybrid": np.load("models/fusion_hybrid_y_pred.npy"),
}

# per-class F1 CIs for each model
results = {}
for name, y_pred in preds.items():
    point, lo, hi = bootstrap_ci_per_class(y_test, y_pred, B=1000, seed=42)
    results[name] = (point, lo, hi)

# delta vs late
delta_vs_late = {}
for name, y_pred in preds.items():
    if name == "Late":
        continue
    d_point, d_lo, d_hi = bootstrap_ci_delta(y_test, y_pred, preds["Late"], B=1000, seed=42)
    delta_vs_late[name] = (d_point, d_lo, d_hi)

# printing the results
print("\nLate fusion per-class F1 with 95% bootstrap CI:")
late_point, late_lo, late_hi = results["Late"]
for c in range(9):
    print(f"{c} {class_names[c]:10s}  F1={late_point[c]:.3f}  [{late_lo[c]:.3f}, {late_hi[c]:.3f}]")

print("\nDelta F1 vs Late fusion:")
for name, (dp, dlo, dhi) in delta_vs_late.items():
    print(f"\n{name} - Late")
    for c in range(9):
        print(f"{c} {class_names[c]:10s}  Δ={dp[c]:+.3f}  [{dlo[c]:+.3f}, {dhi[c]:+.3f}]")
