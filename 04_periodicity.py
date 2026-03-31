"""
04_periodicity.py
Lomb-Scargle periodicity search on all 9258 Stripe 82 quasars (r-band).

Methodology:
  1. Run LS on all objects over a period grid of 200–1100 days
     (lower limit avoids the seasonal pair desert <270 d;
      upper limit enforces >= 3 full cycles in the ~3300 d baseline)
  2. Mask alias periods: 365 ± 25 d, 182.5 ± 15 d, 121.7 ± 10 d
  3. For the top 200 candidates by peak LS power, compute DRW-based
     significance via Monte Carlo (200 simulations per object):
     simulate DRW light curves at the same cadence, run LS, build
     the null distribution of max peak power.
  4. Report objects where the observed peak exceeds the 99th percentile
     of the DRW null distribution as periodic candidates.

Reference: Charisi et al. 2016 (MNRAS 463, 2145) for methodology.
Limitation: LS assumes sinusoidal signals; see Lin, Charisi & Haiman 2026
  (arXiv:2505.14778) for evidence that SMBHB signals are sawtooth-shaped
  and LS recovers these at only ~1-9% efficiency.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.timeseries import LombScargle

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

DB_COLS = [
    "dbID","ra","dec","SDR5ID","M_i","M_i_corr",
    "redshift","mass_BH","Lbol",
    "u_mean","g_mean","r_mean","i_mean","z_mean","Au",
]

DRW_COLS = [
    "SDR5ID","ra","dec","redshift","M_i","mass_BH","chi2_pdf",
    "log_tau","log_sigma",
    "log_tau_lo","log_tau_hi","log_sig_lo","log_sig_hi",
    "edge_flag","Plike","Pnoise","Pinf","mu","npts",
]

# ── Period grid ──────────────────────────────────────────────────────────────
P_MIN  = 200.0    # days — avoid the <270 d pair desert; keep some buffer
P_MAX  = 1100.0   # days — baseline ~3300 d / 3 cycles
N_FREQ = 2000     # frequency grid points
freqs  = np.linspace(1.0/P_MAX, 1.0/P_MIN, N_FREQ)
periods = 1.0 / freqs

# Alias masks: True = masked out
ALIAS_CENTERS = [365.25, 182.6, 121.7]
ALIAS_WIDTHS  = [25.0,   15.0,  10.0]

def alias_mask(periods):
    """Return boolean array: True where period is near a known alias."""
    mask = np.zeros(len(periods), dtype=bool)
    for cen, hw in zip(ALIAS_CENTERS, ALIAS_WIDTHS):
        mask |= (np.abs(periods - cen) < hw)
    return mask

period_mask = alias_mask(periods)
periods_clean = periods[~period_mask]
freqs_clean   = freqs[~period_mask]
print(f"Period grid: {P_MIN:.0f}–{P_MAX:.0f} d, {N_FREQ} points, "
      f"{period_mask.sum()} alias-masked")


# ── Load metadata and DRW ───────────────────────────────────────────────────
db  = pd.read_csv(DATA_DIR / "DB_QSO_S82.dat",  comment="#", sep=r"\s+", names=DB_COLS)
drw = pd.read_csv(DATA_DIR / "drw" / "s82drw_r.dat", comment="#", sep=r"\s+", names=DRW_COLS)

db["_ra5"]  = db["ra"].round(5);   db["_dec5"] = db["dec"].round(5)
drw["_ra5"] = drw["ra"].round(5);  drw["_dec5"] = drw["dec"].round(5)
db_drw = db.merge(drw[["_ra5","_dec5","log_tau","log_sigma"]],
                  on=["_ra5","_dec5"], how="left").drop(columns=["_ra5","_dec5"])


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_lc(dbid, band="r"):
    fpath = LC_DIR / str(int(dbid))
    lc = pd.read_csv(fpath, sep=r"\s+", names=LC_COLS, comment="#")
    mjd_col, mag_col, err_col = f"{band}_mjd", f"{band}_mag", f"{band}_err"
    good = lc[mag_col] > -99
    return (lc.loc[good, mjd_col].values,
            lc.loc[good, mag_col].values,
            lc.loc[good, err_col].values)


def run_ls(mjd, mag, err, freqs):
    """Run LS and return power array at given frequencies."""
    ls = LombScargle(mjd, mag, err)
    power = ls.power(freqs)
    return power


def simulate_drw(mjd, log_tau, log_sigma, mu, rng):
    """
    Simulate one DRW light curve sampled at the given MJD values.
    Uses the exact DRW covariance (Ornstein-Uhlenbeck) to draw a sample.
    Returns mag array (zero-meaned, same length as mjd).
    """
    tau   = 10.0 ** log_tau                  # days
    sigma = 10.0 ** log_sigma                # mag/sqrt(yr)
    tau_yr = tau / 365.25
    SF_inf = sigma * np.sqrt(2.0 * tau_yr)   # asymptotic amplitude (mag)
    var    = 0.5 * SF_inf ** 2               # process variance

    # Covariance matrix: C_ij = var * exp(-|t_i - t_j| / tau)
    dt  = np.abs(mjd[:, None] - mjd[None, :])
    cov = var * np.exp(-dt / tau)
    cov += np.eye(len(mjd)) * 1e-10          # numerical jitter

    try:
        L = np.linalg.cholesky(cov)
        z = rng.standard_normal(len(mjd))
        return L @ z
    except np.linalg.LinAlgError:
        # Fall back to eigendecomposition if Cholesky fails
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 0)
        return vecs @ (np.sqrt(vals) * rng.standard_normal(len(mjd)))


# ════════════════════════════════════════════════════════════════════════════
# PASS 1 — Run LS on all objects, store peak power and best period
# ════════════════════════════════════════════════════════════════════════════
print("\nPass 1: Running LS on all objects...")
results = []
lc_files = sorted(LC_DIR.iterdir())

for k, fpath in enumerate(lc_files):
    if k % 1000 == 0:
        print(f"  {k}/{len(lc_files)}")
    dbid = int(fpath.name)
    try:
        mjd, mag, err = load_lc(dbid, "r")
    except Exception:
        continue
    if len(mjd) < 15:
        continue

    power = run_ls(mjd, mag, err, freqs_clean)
    peak_idx  = np.argmax(power)
    peak_pwr  = power[peak_idx]
    best_per  = periods_clean[peak_idx]
    n_cycles  = (mjd.max() - mjd.min()) / best_per

    results.append({
        "dbID":      dbid,
        "peak_power": peak_pwr,
        "best_period": best_per,
        "n_cycles":   n_cycles,
        "n_obs":      len(mjd),
        "baseline":   mjd.max() - mjd.min(),
    })

ls_all = pd.DataFrame(results)
# Require >= 3 full cycles
ls_all = ls_all[ls_all["n_cycles"] >= 3.0].copy()
ls_all.to_csv(DATA_DIR / "ls_pass1.csv", index=False)
print(f"  {len(ls_all)} objects passed 3-cycle cut (from {len(results)})")
print(f"  Top peak powers: {ls_all.nlargest(5,'peak_power')[['dbID','peak_power','best_period','n_cycles']].to_string(index=False)}")


# ════════════════════════════════════════════════════════════════════════════
# PASS 2 — DRW Monte Carlo significance for top 200 by peak power
# ════════════════════════════════════════════════════════════════════════════
print("\nPass 2: DRW Monte Carlo significance for top 200 candidates...")
N_SIM = 200
top200 = ls_all.nlargest(200, "peak_power")
rng = np.random.default_rng(42)

mc_results = []
for i, (_, row) in enumerate(top200.iterrows()):
    if i % 50 == 0:
        print(f"  {i}/200")
    dbid = int(row["dbID"])
    try:
        mjd, mag, err = load_lc(dbid, "r")
    except Exception:
        continue

    # DRW parameters for this object
    drw_row = db_drw[db_drw["dbID"] == dbid]
    if len(drw_row) == 0 or pd.isna(drw_row.iloc[0]["log_tau"]):
        continue
    lt = drw_row.iloc[0]["log_tau"]
    ls_val = drw_row.iloc[0]["log_sigma"]
    if lt < -5 or ls_val < -5:
        continue

    mu = mag.mean()

    # Simulate N_SIM DRW light curves and run LS on each
    null_powers = []
    for _ in range(N_SIM):
        sim_mag = simulate_drw(mjd, lt, ls_val, mu, rng) + mu
        sim_mag += rng.normal(0, err)     # add photon noise
        pwr = run_ls(mjd, sim_mag, err, freqs_clean)
        null_powers.append(pwr.max())

    null_powers = np.array(null_powers)
    p99  = np.percentile(null_powers, 99)
    p999 = np.percentile(null_powers, 99.9)
    obs_power = row["peak_power"]

    mc_results.append({
        "dbID":        dbid,
        "peak_power":  obs_power,
        "best_period": row["best_period"],
        "n_cycles":    row["n_cycles"],
        "n_obs":       row["n_obs"],
        "baseline":    row["baseline"],
        "null_p99":    p99,
        "null_p999":   p999,
        "sig_99":      obs_power > p99,
        "sig_999":     obs_power > p999,
        "log_tau":     lt,
        "log_sigma":   ls_val,
    })

mc_df = pd.DataFrame(mc_results)
candidates = mc_df[mc_df["sig_99"]].copy()
mc_df.to_csv(DATA_DIR / "ls_mc_results.csv", index=False)
candidates.to_csv(DATA_DIR / "candidates.csv", index=False)

print(f"\n  Significant at 99% DRW level:   {mc_df['sig_99'].sum()}")
print(f"  Significant at 99.9% DRW level: {mc_df['sig_999'].sum()}")
print(f"\n  Candidates (>99%):")
print(candidates[["dbID","best_period","n_cycles","peak_power","null_p99","sig_999"]].to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════
# PLOT 1 — LS peak power distribution (all objects)
# ════════════════════════════════════════════════════════════════════════════
print("\nPlotting LS peak power distribution...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Lomb-Scargle pass 1 results — r-band, 200–1100 d, ≥3 cycles", fontsize=13)

ax = axes[0]
ax.hist(ls_all["peak_power"], bins=80, color="steelblue", edgecolor="k", lw=0.3)
ax.axvline(mc_df["null_p99"].median(), color="darkorange", lw=1.5,
           linestyle="--", label=f"Median DRW 99th pct ({mc_df['null_p99'].median():.3f})")
ax.set_xlabel("Peak LS power")
ax.set_ylabel("Count")
ax.set_title("Distribution of peak LS power")
ax.legend(fontsize=9)

ax = axes[1]
ax.scatter(ls_all["best_period"], ls_all["peak_power"],
           s=1, alpha=0.3, color="steelblue")
for cen in ALIAS_CENTERS:
    if P_MIN <= cen <= P_MAX:
        ax.axvline(cen, color="red", lw=1, linestyle=":")
ax.set_xlabel("Best-fit period (days)")
ax.set_ylabel("Peak LS power")
ax.set_title("Peak power vs best-fit period")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "08_ls_pass1.png", dpi=150)
plt.close()
print("  Saved plots/08_ls_pass1.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 2 — LS periodograms + phase-folded light curves for top candidates
# ════════════════════════════════════════════════════════════════════════════
print("Plotting candidate periodograms...")

plot_cands = candidates.nlargest(min(6, len(candidates)), "peak_power")
if len(plot_cands) == 0:
    # Fall back to top 6 by power even if not formally significant
    plot_cands = mc_df.nlargest(6, "peak_power")
    print("  No formally significant candidates — plotting top 6 by power")

n_cands = len(plot_cands)
fig, axes = plt.subplots(n_cands, 2, figsize=(14, 3.5 * n_cands))
if n_cands == 1:
    axes = axes[np.newaxis, :]
fig.suptitle("Candidate periodograms and phase-folded light curves", fontsize=13)

for idx, (_, row) in enumerate(plot_cands.iterrows()):
    dbid = int(row["dbID"])
    try:
        mjd, mag, err = load_lc(dbid, "r")
    except Exception:
        continue

    power_full = run_ls(mjd, mag, err, freqs)

    # Periodogram
    ax = axes[idx, 0]
    ax.plot(periods[~period_mask], power_full[~period_mask],
            lw=0.8, color="steelblue")
    for cen, hw in zip(ALIAS_CENTERS, ALIAS_WIDTHS):
        ax.axvspan(cen - hw, cen + hw, alpha=0.15, color="red")
    ax.axhline(row["null_p99"],  color="darkorange", lw=1.2, linestyle="--",
               label=f"DRW 99% ({row['null_p99']:.3f})")
    ax.axhline(row["null_p999"], color="red", lw=1.0, linestyle=":",
               label=f"DRW 99.9% ({row['null_p999']:.3f})")
    ax.axvline(row["best_period"], color="forestgreen", lw=1.2, linestyle="-",
               label=f"P={row['best_period']:.1f} d")
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("LS power")
    ax.set_title(f"ID {dbid} | P={row['best_period']:.1f} d | {row['n_cycles']:.1f} cycles | "
                 f"sig99={'Y' if row['sig_99'] else 'N'} sig999={'Y' if row['sig_999'] else 'N'}",
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.set_xlim(100, P_MAX + 100)

    # Phase-folded light curve — plot two cycles (0 to 2) with step lines
    ax = axes[idx, 1]
    phase = ((mjd - mjd.min()) % row["best_period"]) / row["best_period"]
    order = np.argsort(phase)
    ph_sorted  = phase[order]
    mag_sorted = mag[order]
    err_sorted = err[order]
    # Duplicate for second cycle
    ph2  = np.concatenate([ph_sorted, ph_sorted + 1.0])
    mag2 = np.concatenate([mag_sorted, mag_sorted])
    err2 = np.concatenate([err_sorted, err_sorted])
    ax.step(ph2, mag2, where="mid", color="steelblue", lw=1.0, zorder=2)
    ax.errorbar(ph2, mag2, yerr=err2, fmt="o", ms=2.0, color="steelblue",
                ecolor="lightgray", elinewidth=0.7, capsize=0, zorder=3)
    ax.axvline(1.0, color="gray", lw=0.8, linestyle="--", alpha=0.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 2)
    ax.set_xlabel("Phase (0–2)")
    ax.set_ylabel("r magnitude")
    ax.set_title(f"Phase-folded at P={row['best_period']:.1f} d", fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "09_candidate_periodograms.png", dpi=150)
plt.close()
print("  Saved plots/09_candidate_periodograms.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Observed peak power vs DRW 99th percentile (MC validation)
# ════════════════════════════════════════════════════════════════════════════
print("Plotting MC significance summary...")

fig, ax = plt.subplots(figsize=(8, 6))
sig   = mc_df["sig_99"]
ax.scatter(mc_df.loc[~sig, "null_p99"], mc_df.loc[~sig, "peak_power"],
           s=8, alpha=0.5, color="steelblue", label="Not significant")
ax.scatter(mc_df.loc[sig, "null_p99"], mc_df.loc[sig, "peak_power"],
           s=30, alpha=0.9, color="red", zorder=5, label="Significant (>99% DRW)")
lim = max(mc_df["null_p99"].max(), mc_df["peak_power"].max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=1, label="Obs = DRW 99%")
ax.set_xlabel("DRW null 99th percentile power")
ax.set_ylabel("Observed peak LS power")
ax.set_title("Observed LS peak vs DRW null distribution\n"
             "(points above diagonal = significant candidates)", fontsize=12)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "10_mc_significance.png", dpi=150)
plt.close()
print("  Saved plots/10_mc_significance.png")

print(f"\nPeriodicity analysis complete.")
print(f"  Objects searched: {len(ls_all)}")
print(f"  Top 200 MC-tested: {len(mc_df)}")
print(f"  Periodic candidates (>99% DRW): {candidates['dbID'].nunique()}")
print(f"  Saved: data/candidates.csv")
print("\nDone. Ready for 05_ml.py and 06_crossmatch.py")
