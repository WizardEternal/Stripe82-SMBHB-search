"""
05_ml.py
Feature-based ML classifier to separate periodic candidates from stochastic quasars.

Features extracted per object (r-band):
  From light curves: mean, std, skewness, kurtosis, excess variance, n_obs, baseline
  From sf_results.csv: sf_at_300d, drw_at_300d, sf_excess, log_tau, log_sigma
  From ls_pass1.csv: peak_power, best_period, n_cycles

Labels:
  Positive (1): objects significant at >99% DRW level (44 objects from candidates.csv
    in the current pipeline run; actual count is read from the file at runtime, so
    this docstring figure is informational only).
  Negative (0): everything else
  Given the ~0.5% positive rate, we use class_weight='balanced' in RandomForest
  and evaluate with precision-recall rather than accuracy.

Note: The LS-based labels used here inherit the limitation that LS preferentially
recovers sinusoidal signals. Lin, Charisi & Haiman (2026) show that SMBHB signals
are sawtooth-shaped and LS recovers these at only ~1-9% efficiency. The ML
classifier trained here is therefore biased toward sinusoidal periodicities, but
the feature set (including SF excess, DRW parameters) is shape-agnostic and could
in principle capture non-sinusoidal periodicity if relabelled with better detections.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             roc_auc_score, RocCurveDisplay)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_DIR  = Path("data")
LC_DIR    = DATA_DIR / "lightcurves"
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

LC_COLS = [
    "u_mjd","u_mag","u_err",
    "g_mjd","g_mag","g_err",
    "r_mjd","r_mag","r_err",
    "i_mjd","i_mag","i_err",
    "z_mjd","z_mag","z_err",
    "ra","dec",
]

# ── Load existing results ────────────────────────────────────────────────────
print("Loading existing results...")
sf_res  = pd.read_csv(DATA_DIR / "sf_results.csv")
ls_pass = pd.read_csv(DATA_DIR / "ls_pass1.csv")
cands   = pd.read_csv(DATA_DIR / "candidates.csv")

print(f"  SF results:     {len(sf_res)} objects")
print(f"  LS pass1:       {len(ls_pass)} objects")
print(f"  Candidates:     {len(cands)} objects (positives)")


# ── Extract features from light curves ──────────────────────────────────────
print("Extracting light curve features (r-band)...")

def load_lc_r(dbid):
    fpath = LC_DIR / str(int(dbid))
    lc = pd.read_csv(fpath, sep=r"\s+", names=LC_COLS, comment="#")
    good = lc["r_mag"] > -99
    return (lc.loc[good, "r_mjd"].values,
            lc.loc[good, "r_mag"].values,
            lc.loc[good, "r_err"].values)

lc_features = []
lc_files = sorted(LC_DIR.iterdir())

for k, fpath in enumerate(lc_files):
    if k % 1000 == 0:
        print(f"  {k}/{len(lc_files)}")
    dbid = int(fpath.name)
    try:
        mjd, mag, err = load_lc_r(dbid)
    except Exception:
        continue
    if len(mjd) < 10:
        continue

    # Excess variance: intrinsic variance above measurement noise
    mean_err2  = np.mean(err ** 2)
    var_obs    = np.var(mag, ddof=1)
    excess_var = max(var_obs - mean_err2, 0.0)

    lc_features.append({
        "dbID":         dbid,
        "mean_mag":     mag.mean(),
        "std_mag":      mag.std(),
        "skewness":     float(skew(mag)),
        "kurt":         float(kurtosis(mag)),
        "excess_var":   excess_var,
        "n_obs":        len(mjd),
        "baseline":     mjd.max() - mjd.min(),
        "mean_err":     err.mean(),
    })

lc_feat = pd.DataFrame(lc_features)
print(f"  Extracted features for {len(lc_feat)} objects")


# ── Build full feature matrix ────────────────────────────────────────────────
print("Building feature matrix...")

feat = (lc_feat
        .merge(sf_res[["dbID","sf_at_300d","drw_at_300d","sf_excess",
                        "log_tau","log_sigma"]], on="dbID", how="left")
        .merge(ls_pass[["dbID","peak_power","best_period","n_cycles"]], on="dbID", how="left"))

# log-transform skewed features
for col in ["sf_at_300d","drw_at_300d","sf_excess","excess_var"]:
    feat[f"log_{col}"] = np.log10(feat[col].clip(lower=1e-6))

FEATURE_COLS = [
    "mean_mag", "std_mag", "skewness", "kurt",
    "log_excess_var", "n_obs", "baseline", "mean_err",
    "log_sf_at_300d", "log_drw_at_300d", "log_sf_excess",
    "log_tau", "log_sigma",
    "peak_power", "best_period", "n_cycles",
]

# Drop rows with any NaN in features
feat_clean = feat.dropna(subset=FEATURE_COLS).copy()
X = feat_clean[FEATURE_COLS].values
y = feat_clean["dbID"].isin(cands["dbID"]).astype(int).values

print(f"  Feature matrix: {X.shape[0]} objects × {X.shape[1]} features")
print(f"  Positives: {y.sum()} | Negatives: {(y==0).sum()}")

feat_clean.to_csv(DATA_DIR / "feature_matrix.csv", index=False)
print("  Saved data/feature_matrix.csv")


# ── Random Forest with cross-validation ─────────────────────────────────────
print("\nTraining Random Forest (5-fold CV)...")

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
proba_cv = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]

ap  = average_precision_score(y, proba_cv)
auc = roc_auc_score(y, proba_cv)
print(f"  Average Precision (PR-AUC): {ap:.3f}")
print(f"  ROC-AUC:                    {auc:.3f}")

# Fit on full dataset for feature importances only
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print(f"\n  Top 5 features by importance:")
print(importances.head(5).to_string())

# Use cross-validated predictions for stored scores (avoids data leakage)
feat_clean = feat_clean.copy()
feat_clean["rf_score"] = proba_cv
feat_clean["is_candidate"] = y
feat_clean.to_csv(DATA_DIR / "feature_matrix.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Feature importances
# ════════════════════════════════════════════════════════════════════════════
print("\nPlotting feature importances...")

fig, ax = plt.subplots(figsize=(9, 6))
colors = ["crimson" if f in ["peak_power","n_cycles","best_period"] else
          "darkorange" if "sf" in f or "drw" in f else
          "steelblue" for f in importances.index]
importances.plot(kind="barh", ax=ax, color=colors[::-1], edgecolor="k", linewidth=0.4)
ax.invert_yaxis()
ax.set_xlabel("Mean decrease in impurity (feature importance)", fontsize=11)
ax.set_title("Random Forest feature importances\n"
             "(red=LS features, orange=SF/DRW features, blue=LC statistics)", fontsize=12)
ax.axvline(1.0/len(FEATURE_COLS), color="gray", linestyle="--", lw=1, label="Uniform baseline")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "11_feature_importances.png", dpi=150)
plt.close()
print("  Saved plots/11_feature_importances.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Precision-recall curve + ROC curve
# ════════════════════════════════════════════════════════════════════════════
print("Plotting PR and ROC curves...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Random Forest classifier performance (5-fold CV)", fontsize=13)

# PR curve
ax = axes[0]
prec, rec, thr = precision_recall_curve(y, proba_cv)
ax.plot(rec, prec, color="steelblue", lw=1.5, label=f"RF (AP={ap:.3f})")
ax.axhline(y.mean(), color="gray", linestyle="--", lw=1,
           label=f"Random classifier ({y.mean():.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall curve")
ax.legend(fontsize=10); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# ROC curve
ax = axes[1]
RocCurveDisplay.from_predictions(y, proba_cv, ax=ax, color="steelblue",
                                  name=f"RF (AUC={auc:.3f})")
ax.plot([0,1],[0,1],"k--",lw=1,label="Random")
ax.set_title("ROC curve"); ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "12_classifier_performance.png", dpi=150)
plt.close()
print("  Saved plots/12_classifier_performance.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 3 — PCA scatter colored by RF score
# ════════════════════════════════════════════════════════════════════════════
print("Plotting PCA scatter...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"PCA of feature space (PC1={pca.explained_variance_ratio_[0]:.1%}, "
             f"PC2={pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=13)

# Colored by RF score
ax = axes[0]
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=feat_clean["rf_score"],
                s=3, alpha=0.5, cmap="YlOrRd", vmin=0, vmax=1)
plt.colorbar(sc, ax=ax, label="RF periodic score")
# Overlay candidates
cand_mask = feat_clean["is_candidate"] == 1
ax.scatter(X_pca[cand_mask, 0], X_pca[cand_mask, 1],
           s=40, color="blue", zorder=5, label=f"Candidates (N={cand_mask.sum()})",
           edgecolors="k", linewidths=0.5)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title("Colored by RF periodic score")
ax.legend(fontsize=9)

# Colored by LS peak power
ax = axes[1]
sc2 = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=feat_clean["peak_power"],
                 s=3, alpha=0.5, cmap="viridis")
plt.colorbar(sc2, ax=ax, label="LS peak power")
ax.scatter(X_pca[cand_mask, 0], X_pca[cand_mask, 1],
           s=40, color="red", zorder=5, label=f"Candidates (N={cand_mask.sum()})",
           edgecolors="k", linewidths=0.5)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title("Colored by LS peak power")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "13_pca_scatter.png", dpi=150)
plt.close()
print("  Saved plots/13_pca_scatter.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Key 2D diagnostic: peak_power vs sf_excess
# ════════════════════════════════════════════════════════════════════════════
print("Plotting peak power vs SF excess...")

fig, ax = plt.subplots(figsize=(9, 6))
non_cand = feat_clean[feat_clean["is_candidate"] == 0]
pos_cand = feat_clean[feat_clean["is_candidate"] == 1]

ax.scatter(non_cand["peak_power"], np.log10(non_cand["sf_excess"].clip(lower=1e-3)),
           s=2, alpha=0.3, color="steelblue", label="Stochastic")
ax.scatter(pos_cand["peak_power"], np.log10(pos_cand["sf_excess"].clip(lower=1e-3)),
           s=25, color="red", zorder=5, edgecolors="k", linewidths=0.5,
           label=f"Periodic candidates (N={len(pos_cand)})")

# Color by RF score for stochastic — overlay
sc = ax.scatter(non_cand["peak_power"], np.log10(non_cand["sf_excess"].clip(lower=1e-3)),
                c=non_cand["rf_score"], s=2, alpha=0.4, cmap="YlOrRd", vmin=0, vmax=1)
plt.colorbar(sc, ax=ax, label="RF periodic score")

ax.set_xlabel("LS peak power", fontsize=12)
ax.set_ylabel("log₁₀(SF excess at 300 d)", fontsize=12)
ax.set_title("LS peak power vs variability excess relative to DRW", fontsize=12)
ax.legend(fontsize=10)
ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.6, label="SF = DRW")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "14_power_vs_sfexcess.png", dpi=150)
plt.close()
print("  Saved plots/14_power_vs_sfexcess.png")

# ── Summary ──────────────────────────────────────────────────────────────────
top_rf = feat_clean.nlargest(10, "rf_score")[["dbID","rf_score","peak_power","best_period","n_cycles","sf_excess","is_candidate"]]
print(f"\nML summary (RF full):")
print(f"  RF average precision:  {ap:.3f}")
print(f"  RF ROC-AUC:            {auc:.3f}")
print(f"\n  Top 10 objects by RF score:")
print(top_rf.to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════
# MODEL 2 — RF WITHOUT LS features (breaks circularity)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Model 2: RF without LS features (non-circular)")
print("="*60)

from sklearn.inspection import permutation_importance

FEATURE_COLS_NOLS = [
    "mean_mag", "std_mag", "skewness", "kurt",
    "log_excess_var", "n_obs", "baseline", "mean_err",
    "log_sf_at_300d", "log_drw_at_300d", "log_sf_excess",
    "log_tau", "log_sigma",
]

X_nols = feat_clean[FEATURE_COLS_NOLS].values

rf_nols = RandomForestClassifier(
    n_estimators=500, max_depth=None, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
proba_nols = cross_val_predict(rf_nols, X_nols, y, cv=cv, method="predict_proba")[:, 1]
ap_nols  = average_precision_score(y, proba_nols)
auc_nols = roc_auc_score(y, proba_nols)
print(f"  AP (PR-AUC): {ap_nols:.3f}  (was {ap:.3f} with LS features)")
print(f"  ROC-AUC:     {auc_nols:.3f}  (was {auc:.3f} with LS features)")
# Interpretation caveat: AUC 0.81 means these features predict *which objects
# have high LS peak power*, not which objects are genuinely periodic. Features
# like sf_excess and kurtosis correlate with variability amplitude, which in
# turn raises LS power even for purely stochastic objects. The AUC is an upper
# bound on the classifier's ability to detect true periodicity without period
# searching; confirming it requires labels from an independent detection method.

rf_nols.fit(X_nols, y)
# Use CV predictions (already computed above) to avoid leakage
feat_clean["rf_nols_score"] = proba_nols
imp_nols = pd.Series(rf_nols.feature_importances_, index=FEATURE_COLS_NOLS).sort_values(ascending=False)
print(f"  Top 5 features (no LS):")
print(imp_nols.head(5).to_string())


# ════════════════════════════════════════════════════════════════════════════
# MODEL 3 — Isolation Forest (fully unsupervised, no labels)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Model 3: Isolation Forest (unsupervised)")
print("="*60)

# Scale no-LS features separately for the unsupervised model so it isn't
# dominated by LS-derived features (which would just recapitulate the LS result).
scaler_nols = StandardScaler()
X_nols_scaled = scaler_nols.fit_transform(X_nols)

iso = IsolationForest(n_estimators=500, contamination=0.01,
                      random_state=42, n_jobs=-1)
iso.fit(X_nols_scaled)
# score_samples returns negative anomaly score: more negative = more anomalous
# flip sign so higher = more anomalous
feat_clean["iso_score"] = -iso.score_samples(X_nols_scaled)

# Evaluate against LS candidate labels
auc_iso = roc_auc_score(y, feat_clean["iso_score"])
ap_iso  = average_precision_score(y, feat_clean["iso_score"])
print(f"  ROC-AUC vs LS labels: {auc_iso:.3f}")
print(f"  AP vs LS labels:      {ap_iso:.3f}")

feat_clean.to_csv(DATA_DIR / "feature_matrix.csv", index=False)
print("  Updated data/feature_matrix.csv with all three scores")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Feature importances: RF full vs RF no-LS (side by side)
# ════════════════════════════════════════════════════════════════════════════
print("\nPlotting comparative feature importances...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Feature importances: RF with LS features vs without", fontsize=13)

for ax, imp, title, cols in [
    (axes[0], importances,  "RF (all features — circular)",    FEATURE_COLS),
    (axes[1], imp_nols,     "RF (no LS features — non-circular)", FEATURE_COLS_NOLS),
]:
    colors = ["crimson"    if f in ["peak_power","n_cycles","best_period"] else
              "darkorange" if "sf" in f or "drw" in f else
              "steelblue"  for f in imp.index]
    imp.plot(kind="barh", ax=ax, color=colors[::-1], edgecolor="k", linewidth=0.4)
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance")
    ax.set_title(title, fontsize=11)
    ax.axvline(1.0/len(cols), color="gray", linestyle="--", lw=1, label="Uniform baseline")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "15_feature_importances_comparison.png", dpi=150)
plt.close()
print("  Saved plots/15_feature_importances_comparison.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 6 — PR and ROC curves: all three models overlaid
# ════════════════════════════════════════════════════════════════════════════
print("Plotting combined PR and ROC curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model comparison: RF (full) vs RF (no LS) vs Isolation Forest", fontsize=13)

ax = axes[0]
for proba, label, color in [
    (proba_cv,              f"RF full (AP={ap:.3f})",       "steelblue"),
    (proba_nols,            f"RF no-LS (AP={ap_nols:.3f})", "darkorange"),
    (feat_clean["iso_score"].values, f"IsoForest (AP={ap_iso:.3f})", "forestgreen"),
]:
    prec, rec, _ = precision_recall_curve(y, proba)
    ax.plot(rec, prec, lw=1.5, color=color, label=label)
ax.axhline(y.mean(), color="gray", linestyle="--", lw=1, label=f"Random ({y.mean():.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall curve"); ax.legend(fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

ax = axes[1]
for proba, label, color in [
    (proba_cv,              f"RF full (AUC={auc:.3f})",       "steelblue"),
    (proba_nols,            f"RF no-LS (AUC={auc_nols:.3f})", "darkorange"),
    (feat_clean["iso_score"].values, f"IsoForest (AUC={auc_iso:.3f})", "forestgreen"),
]:
    RocCurveDisplay.from_predictions(y, proba, ax=ax, name=label,
                                     color=color, plot_chance_level=False)
ax.plot([0,1],[0,1],"k--",lw=1,label="Random")
ax.set_title("ROC curve"); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "16_model_comparison_curves.png", dpi=150)
plt.close()
print("  Saved plots/16_model_comparison_curves.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 7 — Score correlation: RF full vs RF no-LS vs IsoForest
# ════════════════════════════════════════════════════════════════════════════
print("Plotting score correlations...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Score correlations across models\n"
             "(red = LS candidates, blue = other objects)", fontsize=12)

pairs = [
    ("rf_score",     "rf_nols_score", "RF full score",      "RF no-LS score"),
    ("rf_score",     "iso_score",     "RF full score",      "IsoForest score"),
    ("rf_nols_score","iso_score",     "RF no-LS score",     "IsoForest score"),
]

for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, pairs):
    nc = feat_clean[feat_clean["is_candidate"]==0]
    pc = feat_clean[feat_clean["is_candidate"]==1]
    ax.scatter(nc[xcol], nc[ycol], s=2, alpha=0.2, color="steelblue", label="Stochastic")
    ax.scatter(pc[xcol], pc[ycol], s=25, color="red", zorder=5,
               edgecolors="k", linewidths=0.4, label=f"LS candidates (N={len(pc)})")
    # Objects high in both — strong multi-method candidates
    both_high = feat_clean[(feat_clean[xcol] > feat_clean[xcol].quantile(0.98)) &
                            (feat_clean[ycol] > feat_clean[ycol].quantile(0.98)) &
                            (feat_clean["is_candidate"]==0)]
    if len(both_high) > 0:
        ax.scatter(both_high[xcol], both_high[ycol], s=40, color="gold", zorder=6,
                   edgecolors="k", linewidths=0.4,
                   label=f"New high-score (N={len(both_high)})")
    ax.set_xlabel(xlabel, fontsize=10); ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "17_score_correlations.png", dpi=150)
plt.close()
print("  Saved plots/17_score_correlations.png")


# ── New candidates flagged by IsoForest / RF-noLS but NOT by LS ─────────────
new_cands = feat_clean[
    (feat_clean["is_candidate"] == 0) &
    (feat_clean["iso_score"]     > feat_clean["iso_score"].quantile(0.99)) &
    (feat_clean["rf_nols_score"] > feat_clean["rf_nols_score"].quantile(0.95))
][["dbID","rf_score","rf_nols_score","iso_score","peak_power","best_period","sf_excess"]].sort_values("iso_score", ascending=False)

print(f"\n  Objects in top 1% IsoForest AND top 5% RF-noLS, but NOT LS candidates: {len(new_cands)}")
if len(new_cands) > 0:
    print(new_cands.head(10).to_string(index=False))

print("\nDone. Ready for 06_crossmatch.py")
