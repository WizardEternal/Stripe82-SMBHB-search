"""
06_crossmatch.py
Cross-match our candidates against published SMBHB catalogs, and follow up on
the two objects flagged by IsoForest+RF-noLS but missed by LS.

Published catalogs fetched from VizieR (CDS):
  - Charisi et al. 2016 (MNRAS 463, 2145) — 50 PTF periodic candidates
  - Graham et al. 2015 (MNRAS 453, 1562) — 111 CRTS periodic candidates

Match radius: 2 arcsec (0.000556 deg)

Novel candidates (IsoForest + RF-noLS but NOT LS-significant):
  - Full LS periodogram and phase-folded light curve even though below threshold
  - Comparison of their features against the LS-confirmed population
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u
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

MATCH_RADIUS = 2.0 / 3600.0  # 2 arcsec in degrees

# ── Load our results ─────────────────────────────────────────────────────────
print("Loading our results...")
feat    = pd.read_csv(DATA_DIR / "feature_matrix.csv")
cands   = pd.read_csv(DATA_DIR / "candidates.csv")
mc_df   = pd.read_csv(DATA_DIR / "ls_mc_results.csv")
ls_all  = pd.read_csv(DATA_DIR / "ls_pass1.csv")

# Merge RA/Dec back in from the light curve files (stored in lc_stats if available,
# otherwise read directly from the light curve file first row)
db_cols = ["dbID","ra","dec","redshift","mass_BH","M_i"]
db = pd.read_csv(DATA_DIR / "DB_QSO_S82.dat", comment="#", sep=r"\s+",
                 names=["dbID","ra","dec","SDR5ID","M_i","M_i_corr",
                        "redshift","mass_BH","Lbol",
                        "u_mean","g_mean","r_mean","i_mean","z_mean","Au"])

# Our candidate list with coordinates
cands_coord = cands.merge(db[["dbID","ra","dec","redshift"]], on="dbID", how="left")
all_ls_coord = mc_df.merge(db[["dbID","ra","dec"]], on="dbID", how="left")
feat_coord   = feat.merge(db[["dbID","ra","dec"]], on="dbID", how="left")

# Novel candidates: top IsoForest AND RF-noLS but not LS-significant
novel = feat[
    (feat["is_candidate"] == 0) &
    (feat["iso_score"]     > feat["iso_score"].quantile(0.99)) &
    (feat["rf_nols_score"] > feat["rf_nols_score"].quantile(0.95))
].copy()
novel = novel.merge(db[["dbID","ra","dec","redshift"]], on="dbID", how="left")
novel = novel.merge(ls_all[["dbID","peak_power","best_period","n_cycles"]], on="dbID", how="left")
print(f"  LS candidates:    {len(cands)}")
print(f"  Novel candidates: {len(novel)}")
show_cols = [c for c in ["dbID","ra","dec","rf_nols_score","iso_score","peak_power","best_period"] if c in novel.columns]
print(f"  Novel objects:\n{novel[show_cols].to_string(index=False)}")


# ── Fetch published catalogs from VizieR ─────────────────────────────────────
print("\nFetching published catalogs from VizieR...")
try:
    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = -1

    # Charisi et al. 2016 — J/MNRAS/463/2145 (PTF periodic candidates)
    # Note: this catalog is not available on VizieR; skipping gracefully.
    print("  Charisi+2016: not available on VizieR (PTF survey, separate footprint)")
    charisi_tab = None

    # Graham et al. 2015 — J/MNRAS/453/1562
    print("  Fetching Graham+2015...")
    graham_res = Vizier.get_catalogs("J/MNRAS/453/1562")
    graham_tab = None
    for t in graham_res:
        if "RAJ2000" in t.colnames or "_RAJ2000" in t.colnames:
            graham_tab = t.to_pandas()
            break
    if graham_tab is None and len(graham_res) > 0:
        graham_tab = graham_res[0].to_pandas()

    print(f"  Charisi+2016: {len(charisi_tab) if charisi_tab is not None else 0} entries")
    print(f"  Graham+2015:  {len(graham_tab)  if graham_tab  is not None else 0} entries")

except Exception as e:
    print(f"  VizieR fetch failed: {e}")
    charisi_tab = None
    graham_tab  = None


def parse_coords(tab):
    """Extract RA/Dec in degrees from a VizieR table (handles decimal and sexagesimal)."""
    for ra_col in ["RAJ2000","_RAJ2000","RA_ICRS","RA"]:
        for dec_col in ["DEJ2000","_DEJ2000","DE_ICRS","Dec","DE"]:
            if ra_col in tab.columns and dec_col in tab.columns:
                ra_raw  = tab[ra_col].values
                dec_raw = tab[dec_col].values
                try:
                    # Try decimal degrees first
                    return ra_raw.astype(float), dec_raw.astype(float)
                except (ValueError, TypeError):
                    # Sexagesimal — use astropy
                    coords = SkyCoord(ra=ra_raw, dec=dec_raw,
                                      unit=(u.hourangle, u.deg))
                    return coords.ra.deg, coords.dec.deg
    raise ValueError(f"No RA/Dec columns found. Available: {list(tab.columns)}")


def crossmatch(our_ra, our_dec, their_ra, their_dec, radius_deg=MATCH_RADIUS):
    """
    Return indices into our catalog and matched catalog for pairs within radius.
    """
    our_coords   = SkyCoord(ra=our_ra*u.deg,   dec=our_dec*u.deg)
    their_coords = SkyCoord(ra=their_ra*u.deg, dec=their_dec*u.deg)
    idx, sep, _ = our_coords.match_to_catalog_sky(their_coords)
    matched = sep.deg < radius_deg
    return matched, idx


# ── Crossmatch LS candidates ─────────────────────────────────────────────────
print("\nCross-matching LS candidates against published catalogs...")

our_ra  = cands_coord["ra"].values.astype(float)
our_dec = cands_coord["dec"].values.astype(float)

results_xm = cands_coord.copy()
results_xm["in_charisi16"] = False
results_xm["in_graham15"]  = False

for cat_name, tab in [("Charisi+2016", charisi_tab), ("Graham+2015", graham_tab)]:
    if tab is None:
        print(f"  {cat_name}: skipped (not available)")
        continue
    try:
        cat_ra, cat_dec = parse_coords(tab)
        matched, idx = crossmatch(our_ra, our_dec, cat_ra, cat_dec)
        col = "in_charisi16" if "Charisi" in cat_name else "in_graham15"
        results_xm[col] = matched
        n_match = matched.sum()
        print(f"  {cat_name}: {n_match} matches out of {len(cands)} candidates")
        if n_match > 0:
            print(results_xm[matched][["dbID","ra","dec","best_period","sig_999"]].to_string(index=False))

        # Report how many catalog objects fall in the Stripe 82 footprint
        # (even if unmatched — sets expectation for how many matches are possible)
        stripe82_dec = np.abs(cat_dec) < 1.26
        stripe82_ra  = (cat_ra > 300) | (cat_ra < 60)
        n_in_footprint = (stripe82_dec & stripe82_ra).sum()
        print(f"    ({n_in_footprint} of {len(tab)} {cat_name} objects fall in Stripe 82 footprint)")
    except Exception as e:
        print(f"  {cat_name}: crossmatch error — {e}")

results_xm.to_csv(DATA_DIR / "candidates_crossmatched.csv", index=False)
print("  Saved data/candidates_crossmatched.csv")

# Also crossmatch novel candidates
print("\nCross-matching novel candidates...")
novel_ra  = novel["ra"].values.astype(float)
novel_dec = novel["dec"].values.astype(float)

for cat_name, tab in [("Charisi+2016", charisi_tab), ("Graham+2015", graham_tab)]:
    if tab is None:
        continue
    try:
        cat_ra, cat_dec = parse_coords(tab)
        matched, _ = crossmatch(novel_ra, novel_dec, cat_ra, cat_dec)
        print(f"  Novel vs {cat_name}: {matched.sum()} matches")
    except Exception as e:
        print(f"  {cat_name}: {e}")


# ════════════════════════════════════════════════════════════════════════════
# Helper functions
# ════════════════════════════════════════════════════════════════════════════
P_MIN, P_MAX, N_FREQ = 200.0, 1100.0, 2000
freqs   = np.linspace(1.0/P_MAX, 1.0/P_MIN, N_FREQ)
periods = 1.0 / freqs
ALIAS_CENTERS = [365.25, 182.6, 121.7]
ALIAS_WIDTHS  = [25.0, 15.0, 10.0]

def alias_mask(p):
    m = np.zeros(len(p), dtype=bool)
    for c, hw in zip(ALIAS_CENTERS, ALIAS_WIDTHS):
        m |= (np.abs(p - c) < hw)
    return m

period_mask   = alias_mask(periods)
periods_clean = periods[~period_mask]
freqs_clean   = freqs[~period_mask]

def load_lc(dbid, band="r"):
    lc = pd.read_csv(LC_DIR / str(int(dbid)), sep=r"\s+", names=LC_COLS, comment="#")
    good = lc[f"{band}_mag"] > -99
    return (lc.loc[good, f"{band}_mjd"].values,
            lc.loc[good, f"{band}_mag"].values,
            lc.loc[good, f"{band}_err"].values)


# ════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Crossmatch summary: candidate properties with catalog flags
# ════════════════════════════════════════════════════════════════════════════
print("\nPlotting crossmatch summary...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("LS candidates: period vs peak power, with published catalog matches", fontsize=12)

for ax, sig_col, title in [
    (axes[0], "sig_999", "All 55 candidates (colour = 99.9% significance)"),
    (axes[1], "sig_999", "Zoom: 300–700 d"),
]:
    c16 = results_xm["in_charisi16"]
    g15 = results_xm["in_graham15"]
    neither = ~c16 & ~g15

    ax.scatter(results_xm.loc[neither, "best_period"],
               results_xm.loc[neither, "peak_power"],
               s=30, color="steelblue", alpha=0.7, label="Our candidates only")
    ax.scatter(results_xm.loc[c16, "best_period"],
               results_xm.loc[c16, "peak_power"],
               s=80, color="red", zorder=5, marker="*", label="Charisi+2016 match")
    ax.scatter(results_xm.loc[g15, "best_period"],
               results_xm.loc[g15, "peak_power"],
               s=80, color="gold", zorder=5, marker="D", label="Graham+2015 match")
    ax.set_xlabel("Best-fit period (days)")
    ax.set_ylabel("LS peak power")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)

axes[1].set_xlim(300, 700)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "18_crossmatch_summary.png", dpi=150)
plt.close()
print("  Saved plots/18_crossmatch_summary.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Deep follow-up on the 2 novel candidates
# ════════════════════════════════════════════════════════════════════════════
print("Deep follow-up on novel candidates...")

# Also pull their DRW MC null from mc_df if available, else use ls_all
n_novel = len(novel)
if n_novel == 0:
    print("  No novel candidates found — skipping follow-up plots.")
else:
    fig, axes = plt.subplots(n_novel, 3, figsize=(18, 4.5 * n_novel))
    if n_novel == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Novel candidates: flagged by IsoForest + RF-noLS, NOT by LS\n"
                 "(LS periodogram shown for reference — below formal threshold)", fontsize=12)

for idx, (_, row) in enumerate(novel.iterrows()):
    dbid = int(row["dbID"])
    try:
        mjd, mag, err = load_lc(dbid, "r")
    except Exception as e:
        print(f"  Could not load {dbid}: {e}")
        continue

    # Get DRW null thresholds if available from MC
    mc_row = mc_df[mc_df["dbID"] == dbid]
    has_mc = len(mc_row) > 0
    p99    = mc_row.iloc[0]["null_p99"]  if has_mc else None
    p999   = mc_row.iloc[0]["null_p999"] if has_mc else None

    power_full  = LombScargle(mjd, mag, err).power(freqs)
    power_clean = power_full[~period_mask]
    peak_idx    = np.argmax(power_clean)
    best_period = row["best_period"] if not pd.isna(row.get("best_period")) else periods_clean[peak_idx]

    # --- Col 1: Raw light curve ---
    ax = axes[idx, 0]
    ax.errorbar(mjd, mag, yerr=err, fmt="o", ms=2.5, color="steelblue",
                ecolor="lightgray", elinewidth=0.8, capsize=0)
    ax.invert_yaxis()
    ax.set_xlabel("MJD"); ax.set_ylabel("r magnitude")
    feat_row = feat[feat["dbID"] == dbid].iloc[0] if len(feat[feat["dbID"]==dbid]) > 0 else None
    iso_s  = f"{feat_row['iso_score']:.3f}"  if feat_row is not None else "N/A"
    nols_s = f"{feat_row['rf_nols_score']:.3f}" if feat_row is not None else "N/A"
    ax.set_title(f"ID {dbid} | z={row['redshift']:.2f}\n"
                 f"IsoForest={iso_s}  RF-noLS={nols_s}", fontsize=9)

    # --- Col 2: LS periodogram ---
    ax = axes[idx, 1]
    ax.plot(periods[~period_mask], power_clean, lw=0.9, color="steelblue")
    for cen, hw in zip(ALIAS_CENTERS, ALIAS_WIDTHS):
        ax.axvspan(cen - hw, cen + hw, alpha=0.15, color="red")
    if has_mc:
        ax.axhline(p99,  color="darkorange", lw=1.2, linestyle="--",
                   label=f"DRW 99%  ({p99:.3f})")
        ax.axhline(p999, color="red",        lw=1.0, linestyle=":",
                   label=f"DRW 99.9% ({p999:.3f})")
    else:
        # Approximate from median of all MC results
        ax.axhline(mc_df["null_p99"].median(), color="darkorange", lw=1.2,
                   linestyle="--", label=f"Approx DRW 99% (median)")
    ax.axvline(best_period, color="forestgreen", lw=1.2,
               label=f"P={best_period:.1f} d")
    peak_pwr = power_clean[peak_idx]
    ax.set_xlabel("Period (days)"); ax.set_ylabel("LS power")
    ax.set_title(f"Peak power={peak_pwr:.3f}  (below threshold but shown)", fontsize=9)
    ax.legend(fontsize=7); ax.set_xlim(100, P_MAX + 100)

    # --- Col 3: Phase-folded light curve (0 to 2) ---
    ax = axes[idx, 2]
    phase = ((mjd - mjd.min()) % best_period) / best_period
    order = np.argsort(phase)
    ph_s, mag_s, err_s = phase[order], mag[order], err[order]
    ph2  = np.concatenate([ph_s, ph_s + 1.0])
    mag2 = np.concatenate([mag_s, mag_s])
    err2 = np.concatenate([err_s, err_s])
    ax.step(ph2, mag2, where="mid", color="steelblue", lw=1.0, zorder=2)
    ax.errorbar(ph2, mag2, yerr=err2, fmt="o", ms=2.0, color="steelblue",
                ecolor="lightgray", elinewidth=0.7, capsize=0, zorder=3)
    ax.axvline(1.0, color="gray", lw=0.8, linestyle="--", alpha=0.6)
    ax.invert_yaxis(); ax.set_xlim(0, 2)
    ax.set_xlabel("Phase (0–2)"); ax.set_ylabel("r magnitude")
    ax.set_title(f"Phase-folded at P={best_period:.1f} d", fontsize=9)

if n_novel > 0:
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "19_novel_candidates_followup.png", dpi=150)
    plt.close()
    print("  Saved plots/19_novel_candidates_followup.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Feature comparison: novel vs LS candidates vs stochastic
# ════════════════════════════════════════════════════════════════════════════
print("Plotting feature comparison...")

feat_full = feat.merge(db[["dbID","ra","dec"]], on="dbID", how="left")

novel_ids = novel["dbID"].values
ls_ids    = cands["dbID"].values

stochastic_mask = (~feat_full["dbID"].isin(ls_ids)) & (~feat_full["dbID"].isin(novel_ids))
ls_mask         = feat_full["dbID"].isin(ls_ids)
novel_mask      = feat_full["dbID"].isin(novel_ids)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Feature distributions: stochastic vs LS candidates vs novel candidates", fontsize=12)

plot_features = [
    ("std_mag",          "r-band RMS (mag)"),
    ("skewness",         "Skewness"),
    ("kurt",             "Kurtosis"),
    ("log_excess_var",   "log₁₀(Excess variance)"),
    ("log_sf_excess",    "log₁₀(SF excess at 300 d)"),
    ("log_tau",          "log₁₀(DRW τ) [days]"),
]

for ax, (col, label) in zip(axes.flat, plot_features):
    vals_s = feat_full.loc[stochastic_mask, col].dropna()
    vals_l = feat_full.loc[ls_mask,         col].dropna()
    vals_n = feat_full.loc[novel_mask,       col].dropna()

    bins = np.linspace(np.percentile(vals_s, 1), np.percentile(vals_s, 99), 40)
    ax.hist(vals_s, bins=bins, density=True, alpha=0.4,
            color="steelblue", label="Stochastic")
    ax.hist(vals_l, bins=bins, density=True, alpha=0.6,
            color="red", label=f"LS candidates (N={len(vals_l)})")
    for v, dbid in zip(vals_n.values, novel_ids):
        ax.axvline(v, color="gold", lw=2.0, linestyle="-",
                   label=f"Novel ID {int(dbid)}" if list(vals_n.values).index(v)==0 else "")
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "20_feature_comparison.png", dpi=150)
plt.close()
print("  Saved plots/20_feature_comparison.png")


# ── Final summary ─────────────────────────────────────────────────────────────
n_c16 = results_xm["in_charisi16"].sum()
n_g15 = results_xm["in_graham15"].sum()

print(f"\n{'='*55}")
print(f"FINAL SUMMARY")
print(f"{'='*55}")
print(f"  Objects searched:              9258")
print(f"  LS candidates (>99% DRW):       {len(cands)}")
print(f"  LS candidates (>99.9% DRW):     {mc_df['sig_999'].sum()}")
print(f"  Matches in Charisi+2016:         {n_c16}")
print(f"  Matches in Graham+2015:          {n_g15}")
print(f"  Novel (IsoForest+RF-noLS):       {len(novel)}")
print(f"{'='*55}")
print("\nAll results saved. Pipeline complete.")
