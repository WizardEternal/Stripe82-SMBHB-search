# Stripe 82 SMBHB Search

An end-to-end pipeline for searching supermassive black hole binary (SMBHB) candidates in the SDSS Stripe 82 quasar dataset using optical periodicity analysis and machine learning. Built as a hands-on exploration of the methodology underpinning the [MMMonsters ERC project](https://www.ia.forth.gr/).

---

## Scientific Background

Active galactic nuclei (quasars) are powered by accretion onto supermassive black holes and show stochastic optical variability. This variability is well-modelled as a **Damped Random Walk (DRW)** — a red-noise process characterised by a decorrelation timescale τ and variability amplitude SF∞. A SMBHB should imprint *periodic* variability on top of this background due to its orbital motion, potentially via Doppler boosting, accretion rate modulation, or circumbinary disk dynamics.

The core challenge: **red noise can mimic periodicity** over a finite observing baseline. A naive Lomb-Scargle search with a white-noise false alarm probability will produce many spurious detections. The correct approach is to assess significance against the DRW process itself — using the pre-fitted DRW parameters for each quasar to simulate realistic null light curves and build an empirical noise floor.

---

## Dataset

**SDSS Stripe 82** — MacLeod et al. (2012): 9,258 spectroscopically confirmed quasars with ~10 year optical light curves in ugriz bands. Observations are organised in yearly ~3-month seasons with roughly nightly cadence, creating large (~9 month) seasonal gaps. This irregular, gappy cadence is a major challenge for period finding.

Downloaded automatically by `01_download_data.py` from the [MacLeod et al. catalog](http://faculty.washington.edu/ivezic/macleod/qso_dr7/Southern.html).

---

## Pipeline

| Script | What it does |
|--------|-------------|
| `01_download_data.py` | Downloads and extracts all three data archives (~30 MB) |
| `02_eda.py` | Exploratory analysis — light curves, cadence, population distributions |
| `03_variability.py` | Structure functions for all objects; comparison against DRW model |
| `04_periodicity.py` | Lomb-Scargle search with DRW Monte Carlo null; alias masking |
| `05_ml.py` | Three ML approaches for classifying periodic vs stochastic objects |
| `06_crossmatch.py` | Cross-match candidates against Graham+2015; follow-up on novel detections |

---

## Methodology

### DRW as the null hypothesis
Most published searches assess LS peak significance against white noise. Quasar variability is red noise — it has more power at low frequencies, making the white-noise FAP severely underestimated. Here, for each candidate, we simulate 200 DRW light curves sampled at the same cadence as the observed data (using the MacLeod et al. pre-fitted τ and σ), run LS on each, and use the 99th/99.9th percentile of the resulting peak power distribution as the significance threshold.

### Seasonal alias masking
Stripe 82's ~9-month seasonal gaps create strong aliasing in the LS periodogram at 365 d, 182.5 d, and 121.7 d. Any peak within ±25/15/10 days of these is explicitly masked before reporting candidates.

### 3-cycle requirement
Only periods where at least 3 full cycles fit within the object's observing baseline are considered. A claimed "period" of 4000 days in a 3000-day dataset is noise. This is the Vaughan et al. (2016) criterion now standard in the field.

### Three independent ML approaches
The ML section deliberately uses three methods to avoid the circularity of training a classifier whose labels and dominant feature are both derived from the same LS periodogram:

- **RF (full features)**: Includes LS peak power — effectively reproduces the LS selection, demonstrates the circularity problem.
- **RF (no LS features)**: Trained only on light curve statistics and DRW parameters. ROC-AUC = 0.81, showing that photometric variability properties *alone* carry real information about periodicity — a non-trivial, non-circular result.
- **Isolation Forest (unsupervised)**: No labels required. Finds anomalous objects purely from feature-space density. Identifies objects missed by LS that are photometrically unusual.

---

## Results

### Population variability (plots 01–07)
- Median of 60 r-band observations per object over a ~3300-day baseline
- Observed structure functions broadly consistent with DRW but systematically ~20% below the DRW prediction at intermediate lags (~100–300 days), caused by the near-absence of observation pairs during the seasonal gap
- 75% of objects fall below their DRW prediction at 300 days — the DRW model slightly overestimates variability at these timescales

### Periodicity search (plots 08–10)
- 8,896 objects passed the 3-cycle cut over the 200–1100 day period grid
- **55 candidates at >99% DRW significance**
- **27 candidates at >99.9% DRW significance**
- Candidates cluster at periods of 300–500 days with no excess at the alias periods, confirming the masking is working correctly

### ML results (plots 11–17)
- RF (full): ROC-AUC = 0.996, AP = 0.488 — inflated by circularity (peak power dominates feature importance at 62%)
- RF (no LS features): ROC-AUC = 0.808, AP = 0.040 — honest non-circular result; kurtosis, skewness, and excess variance are the leading discriminants
- Isolation Forest: ROC-AUC = 0.661 vs LS labels — partially independent ranking, flags a different population
- **2 novel candidates** appear in both the top 1% of Isolation Forest scores and top 5% of RF-noLS scores, while falling below the formal LS threshold. Both show unusually heavy-tailed magnitude distributions (high kurtosis) and are at z~1.7. Their LS periodograms have peaks just below the DRW 99% threshold, and their phase-folded light curves show broadly coherent structure at ~340 d and ~409 d respectively.

### Crossmatch (plots 18–20)
- 0 matches with Graham+2015 (CRTS): only 2 of 111 Graham objects fall in the Stripe 82 footprint at all, and neither matches our candidates. This is expected given the different survey cadence and baseline, and consistent with the known low persistence rate of candidates from that era — many of which have since been shown to be red-noise artefacts over extended baselines.
- Charisi+2016 (PTF) is not available on VizieR; PTF and Stripe 82 have limited footprint overlap.

---

## A Note on LS Limitations

Lomb-Scargle assumes sinusoidal signals. Lin, Charisi & Haiman (2026) show that hydrodynamical simulations of circumbinary accretion disks predict **sawtooth-shaped** light curve variations. LS recovers these at only ~1–9% efficiency compared to ~24% for sinusoids — meaning the majority of real SMBHBs are likely missed by any LS-based search. The Isolation Forest component here is a first step toward a shape-agnostic approach, motivated directly by this finding.

---

## Requirements

```bash
pip install numpy pandas matplotlib astropy scikit-learn scipy astroquery
```

Python 3.9+ recommended.

## Usage

```bash
python 01_download_data.py   # ~1 min
python 02_eda.py             # ~3 min
python 03_variability.py     # ~10 min
python 04_periodicity.py     # ~20 min
python 05_ml.py              # ~15 min
python 06_crossmatch.py      # ~2 min
```

Plots → `plots/`   |   Intermediate CSVs → `data/`

---

## References

- MacLeod et al. 2012, ApJ 753, 106 — Stripe 82 DRW light curves and parameters
- Charisi et al. 2016, MNRAS 463, 2145 — PTF periodic quasar candidates
- Graham et al. 2015, MNRAS 453, 1562 — CRTS periodic quasar candidates
- Vaughan et al. 2016, MNRAS 461, 3145 — red noise and spurious periodicity
- Lin, Charisi & Haiman 2026, arXiv:2505.14778 — LS efficiency for non-sinusoidal SMBHB signals
