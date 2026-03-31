"""
02_eda.py
Exploratory data analysis of the Stripe 82 quasar light curve dataset.

Light curve file format (one row per epoch):
  u_MJD u_mag u_err  g_MJD g_mag g_err  r_MJD r_mag r_err
  i_MJD i_mag i_err  z_MJD z_mag z_err  RA Dec
Bad values flagged as -99.99.

DB_QSO_S82.dat columns:
  dbID ra dec SDR5ID M_i M_i_corr redshift mass_BH Lbol u g r i z Au
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("data")
LC_DIR    = DATA_DIR / "lightcurves"
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# ── Column definitions ───────────────────────────────────────────────────────
LC_COLS = [
    "u_mjd","u_mag","u_err",
    "g_mjd","g_mag","g_err",
    "r_mjd","r_mag","r_err",
    "i_mjd","i_mag","i_err",
    "z_mjd","z_mag","z_err",
    "ra","dec",
]

DB_COLS = [
    "dbID","ra","dec","SDR5ID","M_i","M_i_corr",
    "redshift","mass_BH","Lbol",
    "u_mean","g_mean","r_mean","i_mean","z_mean","Au",
]

# ── Load metadata ────────────────────────────────────────────────────────────
print("Loading metadata...")
db = pd.read_csv(
    DATA_DIR / "DB_QSO_S82.dat",
    comment="#", sep=r"\s+", names=DB_COLS,
)
print(f"  {len(db)} quasars loaded.")

# ── Load all light curves (summary stats only, not full data) ────────────────
print("Computing per-object statistics from all light curves...")

records = []
lc_files = sorted(LC_DIR.iterdir())

for fpath in lc_files:
    try:
        lc = pd.read_csv(fpath, sep=r"\s+", names=LC_COLS, comment="#")
    except Exception:
        continue

    dbid = int(fpath.name)

    for band, mjd_col, mag_col in [
        ("r", "r_mjd", "r_mag"),
        ("g", "g_mjd", "g_mag"),
    ]:
        good = lc[mag_col] > -99
        if good.sum() < 2:
            continue
        mjd  = lc.loc[good, mjd_col].values
        mag  = lc.loc[good, mag_col].values
        records.append({
            "dbID":      dbid,
            "band":      band,
            "n_obs":     good.sum(),
            "baseline":  mjd.max() - mjd.min(),
            "mean_mag":  mag.mean(),
            "std_mag":   mag.std(),
            "mjd_min":   mjd.min(),
            "mjd_max":   mjd.max(),
        })

stats = pd.DataFrame(records)
stats_r = stats[stats["band"] == "r"].copy()
stats_g = stats[stats["band"] == "g"].copy()
print(f"  Objects with r-band data: {len(stats_r)}")
print(f"  Objects with g-band data: {len(stats_g)}")

# Merge with metadata
stats_r = stats_r.merge(db[["dbID","redshift","mass_BH","M_i"]], on="dbID", how="left")

# ── Save summary for later scripts ──────────────────────────────────────────
stats.to_csv(DATA_DIR / "lc_stats.csv", index=False)
print("  Saved data/lc_stats.csv")

# ── Helper: load one light curve ─────────────────────────────────────────────
def load_lc(dbid, band="r"):
    fpath = LC_DIR / str(dbid)
    lc = pd.read_csv(fpath, sep=r"\s+", names=LC_COLS, comment="#")
    mjd_col = f"{band}_mjd"
    mag_col = f"{band}_mag"
    err_col = f"{band}_err"
    good = lc[mag_col] > -99
    return lc.loc[good, mjd_col].values, lc.loc[good, mag_col].values, lc.loc[good, err_col].values


# ════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Population distributions (4-panel)
# ════════════════════════════════════════════════════════════════════════════
print("Plotting population distributions...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Stripe 82 Quasar Population — r-band", fontsize=14)

# (a) Number of observations
ax = axes[0, 0]
ax.hist(stats_r["n_obs"], bins=50, color="steelblue", edgecolor="k", linewidth=0.3)
ax.set_xlabel("Number of r-band observations")
ax.set_ylabel("Count")
ax.set_title(f"Observations per object  (median={stats_r['n_obs'].median():.0f})")

# (b) Time baseline
ax = axes[0, 1]
ax.hist(stats_r["baseline"], bins=50, color="darkorange", edgecolor="k", linewidth=0.3)
ax.set_xlabel("Baseline (days)")
ax.set_ylabel("Count")
ax.set_title(f"Time baseline  (median={stats_r['baseline'].median():.0f} d)")

# (c) Redshift
ax = axes[1, 0]
z = db["redshift"]
z = z[z > 0]
ax.hist(z, bins=60, color="forestgreen", edgecolor="k", linewidth=0.3)
ax.set_xlabel("Redshift")
ax.set_ylabel("Count")
ax.set_title(f"Redshift distribution  (median z={z.median():.2f})")

# (d) RMS variability amplitude
ax = axes[1, 1]
ax.hist(stats_r["std_mag"], bins=60, color="orchid", edgecolor="k", linewidth=0.3)
ax.set_xlabel("r-band RMS magnitude (mag)")
ax.set_ylabel("Count")
ax.set_title(f"Variability amplitude  (median={stats_r['std_mag'].median():.3f} mag)")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_population_distributions.png", dpi=150)
plt.close()
print("  Saved plots/01_population_distributions.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Cadence / seasonal gaps from one representative object
# ════════════════════════════════════════════════════════════════════════════
print("Plotting cadence example...")

# Pick the object with the most r-band observations
example_id = stats_r.sort_values("n_obs", ascending=False).iloc[0]["dbID"]
mjd, mag, err = load_lc(int(example_id), band="r")

fig, axes = plt.subplots(2, 1, figsize=(12, 6))
fig.suptitle(f"Cadence structure — object {int(example_id):d} (r-band, {len(mjd)} obs)", fontsize=13)

ax = axes[0]
ax.errorbar(mjd, mag, yerr=err, fmt="o", ms=2.5, color="steelblue",
            ecolor="lightgray", elinewidth=0.8, capsize=0)
ax.invert_yaxis()
ax.set_xlabel("MJD")
ax.set_ylabel("r magnitude")
ax.set_title("Light curve")

ax = axes[1]
gaps = np.diff(np.sort(mjd))
ax.hist(gaps, bins=100, color="steelblue", edgecolor="k", linewidth=0.3)
ax.axvline(365, color="red", linestyle="--", label="1 yr (alias)")
ax.set_xlabel("Time gap between consecutive observations (days)")
ax.set_ylabel("Count")
ax.set_title("Gap distribution — note ~300 d seasonal gap")
ax.legend()
ax.set_xlim(0, 400)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_cadence_example.png", dpi=150)
plt.close()
print("  Saved plots/02_cadence_example.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 3 — 12 sample light curves (r and g band)
# ════════════════════════════════════════════════════════════════════════════
print("Plotting 12 sample light curves...")

# Pick 12 objects spread across the variability amplitude range
sample_ids = (
    stats_r.dropna(subset=["std_mag"])
    .sort_values("std_mag")
    .iloc[np.linspace(0, len(stats_r) - 1, 12, dtype=int)]["dbID"]
    .values
)

fig, axes = plt.subplots(4, 3, figsize=(15, 12))
fig.suptitle("Sample light curves (r=blue, g=orange) — sorted by variability amplitude", fontsize=13)

for ax, dbid in zip(axes.flat, sample_ids):
    row = stats_r[stats_r["dbID"] == dbid].iloc[0]
    try:
        mjd_r, mag_r, err_r = load_lc(int(dbid), "r")
        ax.errorbar(mjd_r, mag_r, yerr=err_r, fmt=".", ms=2, color="steelblue",
                    ecolor="lightgray", elinewidth=0.5)
    except Exception:
        pass
    try:
        mjd_g, mag_g, err_g = load_lc(int(dbid), "g")
        ax.errorbar(mjd_g, mag_g, yerr=err_g, fmt=".", ms=2, color="darkorange",
                    ecolor="wheat", elinewidth=0.5)
    except Exception:
        pass
    ax.invert_yaxis()
    ax.set_title(f"ID {int(dbid)} | z={row['redshift']:.2f} | σ={row['std_mag']:.3f}", fontsize=8)
    ax.tick_params(labelsize=7)

for ax in axes.flat:
    ax.set_xlabel("MJD", fontsize=7)
    ax.set_ylabel("mag", fontsize=7)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_sample_lightcurves.png", dpi=150)
plt.close()
print("  Saved plots/03_sample_lightcurves.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Variability vs physical properties
# ════════════════════════════════════════════════════════════════════════════
print("Plotting variability vs physical properties...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("r-band RMS variability vs physical properties", fontsize=13)

# vs redshift
ax = axes[0]
mask = (stats_r["redshift"] > 0) & (stats_r["std_mag"] > 0)
ax.scatter(stats_r.loc[mask, "redshift"], stats_r.loc[mask, "std_mag"],
           s=2, alpha=0.3, color="steelblue")
ax.set_xlabel("Redshift")
ax.set_ylabel("r-band RMS (mag)")
ax.set_title("Variability vs Redshift")

# vs mean magnitude (proxy for luminosity)
ax = axes[1]
mask2 = (stats_r["mean_mag"] > 0) & (stats_r["std_mag"] > 0)
ax.scatter(stats_r.loc[mask2, "mean_mag"], stats_r.loc[mask2, "std_mag"],
           s=2, alpha=0.3, color="darkorange")
ax.set_xlabel("Mean r magnitude")
ax.set_ylabel("r-band RMS (mag)")
ax.set_title("Variability vs Mean Magnitude\n(fainter = lower luminosity)")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_variability_vs_properties.png", dpi=150)
plt.close()
print("  Saved plots/04_variability_vs_properties.png")

print("\nEDA complete. Plots saved to plots/")
print(f"  Total objects: {len(db)}")
print(f"  Median r-band observations: {stats_r['n_obs'].median():.0f}")
print(f"  Median baseline: {stats_r['baseline'].median():.0f} days")
print(f"  Median r-band RMS: {stats_r['std_mag'].median():.3f} mag")
