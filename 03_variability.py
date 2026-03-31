"""
03_variability.py
Structure function computation for all objects + comparison against pre-fitted DRW models.

DRW model: SF(Δt) = SF_inf * sqrt(1 - exp(-Δt / tau))
  where SF_inf = sigma * sqrt(2 * tau_yr)   [sigma in mag/sqrt(yr), tau in yr]
DRW file columns (s82drw_r.dat):
  SDR5ID ra dec redshift M_i mass_BH chi2_pdf
  log10(tau [days]) log10(sigma [mag/sqrt(yr)])
  log10(tau_lim_lo) log10(tau_lim_hi) log10(sig_lim_lo) log10(sig_lim_hi)
  edge_flag Plike Pnoise Pinf mu npts
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

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

DRW_COLS = [
    "SDR5ID","ra","dec","redshift","M_i","mass_BH","chi2_pdf",
    "log_tau","log_sigma",
    "log_tau_lo","log_tau_hi","log_sig_lo","log_sig_hi",
    "edge_flag","Plike","Pnoise","Pinf","mu","npts",
]

DB_COLS = [
    "dbID","ra","dec","SDR5ID","M_i","M_i_corr",
    "redshift","mass_BH","Lbol",
    "u_mean","g_mean","r_mean","i_mean","z_mean","Au",
]

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading metadata and DRW parameters...")
db = pd.read_csv(DATA_DIR / "DB_QSO_S82.dat",
                 comment="#", sep=r"\s+", names=DB_COLS)

drw = pd.read_csv(DATA_DIR / "drw" / "s82drw_r.dat",
                  comment="#", sep=r"\s+", names=DRW_COLS)

# Match DRW params to DB via ra/dec (both files share the same catalog coords).
# Merge on rounded ra/dec to avoid float precision issues — 5 decimal places
# is ~1 arcsec precision, sufficient for unique matching.
db["_ra5"]  = db["ra"].round(5)
db["_dec5"] = db["dec"].round(5)
drw["_ra5"]  = drw["ra"].round(5)
drw["_dec5"] = drw["dec"].round(5)

db_drw = db.merge(
    drw[["_ra5","_dec5","log_tau","log_sigma","edge_flag","npts"]],
    on=["_ra5","_dec5"], how="left"
).drop(columns=["_ra5","_dec5"])

# Keep only objects with valid DRW fits (tau = -10 means < 10 obs, unusable)
valid_drw = db_drw[(db_drw["log_tau"] > -5) & (db_drw["log_sigma"] > -5)].copy()
print(f"  Objects with valid DRW fits: {len(valid_drw)} / {len(db)}")


# ── DRW model ────────────────────────────────────────────────────────────────
def drw_sf(dt, log_tau, log_sigma):
    """Expected DRW structure function at lag dt (days)."""
    tau = 10.0 ** log_tau                   # days
    sigma = 10.0 ** log_sigma               # mag / sqrt(yr)
    tau_yr = tau / 365.25
    SF_inf = sigma * np.sqrt(2.0 * tau_yr)  # mag
    return SF_inf * np.sqrt(1.0 - np.exp(-dt / tau))


# ── Structure function for one light curve ───────────────────────────────────
LAG_BINS = np.logspace(1, 3.6, 30)   # 10 to ~4000 days, log-spaced

def compute_sf(mjd, mag, bins=LAG_BINS):
    """
    Compute binned structure function SF(Δt) = sqrt(<|Δm|^2>) for all pairs.
    Returns (bin_centers, sf_values, n_pairs) — NaN where no pairs exist.
    """
    n = len(mjd)
    if n < 5:
        return np.full(len(bins)-1, np.nan), np.zeros(len(bins)-1, int)

    # All pairs
    i, j = np.triu_indices(n, k=1)
    dt  = np.abs(mjd[i] - mjd[j])
    dm2 = (mag[i] - mag[j]) ** 2

    centers  = np.sqrt(bins[:-1] * bins[1:])
    sf_vals  = np.full(len(centers), np.nan)
    n_pairs  = np.zeros(len(centers), int)

    for k in range(len(centers)):
        mask = (dt >= bins[k]) & (dt < bins[k+1])
        if mask.sum() > 0:
            sf_vals[k] = np.sqrt(dm2[mask].mean())
            n_pairs[k] = mask.sum()

    return sf_vals, n_pairs


# ── Load light curve helper ───────────────────────────────────────────────────
def load_lc(dbid, band="r"):
    fpath = LC_DIR / str(int(dbid))
    lc = pd.read_csv(fpath, sep=r"\s+", names=LC_COLS, comment="#")
    mjd_col, mag_col = f"{band}_mjd", f"{band}_mag"
    good = lc[mag_col] > -99
    return lc.loc[good, mjd_col].values, lc.loc[good, mag_col].values


# ════════════════════════════════════════════════════════════════════════════
# Compute SFs for ALL objects and compare to DRW
# Store per-object: SF at lag ~300 days (a representative timescale),
# and the DRW prediction at the same lag — excess = observed / predicted
# ════════════════════════════════════════════════════════════════════════════
print("Computing structure functions for all objects (r-band)...")

LAG_REF = 300.0   # days — reference lag for excess variability metric
bin_centers = np.sqrt(LAG_BINS[:-1] * LAG_BINS[1:])
ref_bin_idx = np.argmin(np.abs(bin_centers - LAG_REF))

results = []
all_sf_grids = []   # store full SF curves for population mean

lc_files = sorted(LC_DIR.iterdir())
for k, fpath in enumerate(lc_files):
    if k % 1000 == 0:
        print(f"  {k}/{len(lc_files)}")
    dbid = int(fpath.name)
    try:
        mjd, mag = load_lc(dbid, "r")
    except Exception:
        continue
    if len(mjd) < 10:
        continue

    sf_vals, n_pairs = compute_sf(mjd, mag)
    all_sf_grids.append(sf_vals)

    # DRW prediction for this object
    row = valid_drw[valid_drw["dbID"] == dbid]
    if len(row) == 1:
        lt, ls = row.iloc[0]["log_tau"], row.iloc[0]["log_sigma"]
        drw_pred = drw_sf(bin_centers, lt, ls)
        sf_ref   = sf_vals[ref_bin_idx]
        drw_ref  = drw_pred[ref_bin_idx]
        excess   = sf_ref / drw_ref if (drw_ref > 0 and not np.isnan(sf_ref)) else np.nan
    else:
        lt, ls, drw_ref, excess = np.nan, np.nan, np.nan, np.nan

    results.append({
        "dbID": dbid,
        "log_tau": lt, "log_sigma": ls,
        "sf_at_300d": sf_vals[ref_bin_idx],
        "drw_at_300d": drw_ref,
        "sf_excess": excess,
    })

sf_results = pd.DataFrame(results)
sf_results.to_csv(DATA_DIR / "sf_results.csv", index=False)
print(f"  Saved data/sf_results.csv  ({len(sf_results)} objects)")

# Population mean SF (ignore NaNs)
all_sf_arr   = np.array(all_sf_grids)
mean_sf      = np.nanmedian(all_sf_arr, axis=0)
p16_sf       = np.nanpercentile(all_sf_arr, 16, axis=0)
p84_sf       = np.nanpercentile(all_sf_arr, 84, axis=0)


# ════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Mean population SF + DRW population median
# ════════════════════════════════════════════════════════════════════════════
print("Plotting mean structure function...")

# Population median DRW prediction
valid2 = valid_drw[(valid_drw["log_tau"] > -5)]
drw_grid = np.array([drw_sf(bin_centers, r["log_tau"], r["log_sigma"])
                     for _, r in valid2.iterrows()])
median_drw = np.nanmedian(drw_grid, axis=0)
p16_drw    = np.nanpercentile(drw_grid, 16, axis=0)
p84_drw    = np.nanpercentile(drw_grid, 84, axis=0)

fig, ax = plt.subplots(figsize=(9, 6))

ax.fill_between(bin_centers, p16_sf, p84_sf, alpha=0.25, color="steelblue", label="Observed 16–84 pct")
ax.loglog(bin_centers, mean_sf, "o-", color="steelblue", ms=4, lw=1.5, label="Observed median SF")

ax.fill_between(bin_centers, p16_drw, p84_drw, alpha=0.25, color="darkorange", label="DRW model 16–84 pct")
ax.loglog(bin_centers, median_drw, "s--", color="darkorange", ms=4, lw=1.5, label="DRW model median")

ax.axvline(365, color="red", linestyle=":", lw=1.2, label="1 yr alias")
ax.set_xlabel("Time lag Δt (days)", fontsize=12)
ax.set_ylabel("Structure function SF(Δt) (mag)", fontsize=12)
ax.set_title("Population-averaged structure function — r-band (N=9258)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_mean_structure_function.png", dpi=150)
plt.close()
print("  Saved plots/05_mean_structure_function.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Individual SFs for 6 objects: 3 DRW-consistent, 3 high-excess
# ════════════════════════════════════════════════════════════════════════════
print("Plotting individual structure function examples...")

good = sf_results.dropna(subset=["sf_excess"])
# "Consistent" = excess closest to 1.0 (observed ≈ DRW prediction)
normal_ids = good.iloc[(good["sf_excess"] - 1.0).abs().argsort()[:3]]["dbID"].values
excess_ids = good.sort_values("sf_excess", ascending=False).iloc[:3]["dbID"].values

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Individual structure functions vs DRW model\n"
             "Top: DRW-consistent  |  Bottom: High excess variability", fontsize=12)

for row_idx, ids in enumerate([normal_ids, excess_ids]):
    for col_idx, dbid in enumerate(ids):
        ax = axes[row_idx, col_idx]
        try:
            mjd, mag = load_lc(int(dbid), "r")
            sf_vals, _ = compute_sf(mjd, mag)
            ax.loglog(bin_centers, sf_vals, "o-", ms=4, color="steelblue", label="Observed")
        except Exception:
            pass

        row = valid_drw[valid_drw["dbID"] == dbid]
        if len(row) == 1:
            lt, ls = row.iloc[0]["log_tau"], row.iloc[0]["log_sigma"]
            ax.loglog(bin_centers, drw_sf(bin_centers, lt, ls),
                      "--", color="darkorange", lw=1.5, label="DRW model")
            exc = sf_results.loc[sf_results["dbID"]==dbid, "sf_excess"]
            exc_val = exc.values[0] if len(exc) else np.nan
            ax.set_title(f"ID {int(dbid)} | excess={exc_val:.2f}x", fontsize=9)

        ax.axvline(365, color="red", linestyle=":", lw=0.8)
        ax.set_xlabel("Δt (days)", fontsize=8)
        ax.set_ylabel("SF (mag)", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "06_individual_structure_functions.png", dpi=150)
plt.close()
print("  Saved plots/06_individual_structure_functions.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Distribution of SF excess at 300 days
# ════════════════════════════════════════════════════════════════════════════
print("Plotting SF excess distribution...")

excess = sf_results["sf_excess"].dropna()
excess = excess[(excess > 0) & (excess < 20)]   # remove extreme outliers for display

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(np.log10(excess), bins=80, color="steelblue", edgecolor="k", linewidth=0.3)
ax.axvline(0, color="red", linestyle="--", lw=1.5, label="SF = DRW prediction")
ax.axvline(np.log10(2), color="darkorange", linestyle="--", lw=1.5, label="2× excess")
ax.set_xlabel("log₁₀(SF_observed / SF_DRW)  at Δt ≈ 300 days", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Distribution of variability excess relative to DRW model", fontsize=13)
ax.legend(fontsize=10)

n_excess = (excess > 2).sum()
ax.text(0.97, 0.95, f"Objects with >2× excess: {n_excess} ({100*n_excess/len(excess):.1f}%)",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
plt.savefig(PLOTS_DIR / "07_sf_excess_distribution.png", dpi=150)
plt.close()
print("  Saved plots/07_sf_excess_distribution.png")

# Summary
n_excess2 = int((sf_results["sf_excess"].dropna() > 2).sum())
print(f"\nVariability summary:")
print(f"  Objects with valid DRW fits: {len(valid_drw)}")
print(f"  Objects with >2× SF excess at 300 d: {n_excess2}")
print(f"  Top 5 excess objects: {sf_results.nlargest(5,'sf_excess')[['dbID','sf_excess']].to_string(index=False)}")
print("\nDone. Ready for 04_periodicity.py")
