# Stripe 82 Quasar Periodicity Pipeline

A complete end-to-end pipeline for searching supermassive black hole binary (SMBHB) candidates in the SDSS Stripe 82 quasar light curve dataset via optical periodicity analysis and machine learning.

## Scientific Background

Quasars show stochastic optical variability well-described by a Damped Random Walk (DRW) process. A SMBHB should imprint *periodic* variability on top of this stochastic background due to its orbital motion. The challenge is distinguishing genuine periodicity from red-noise fluctuations that can mimic it over a finite baseline.

This pipeline applies a statistically rigorous search to **9,258 spectroscopically confirmed quasars** from the MacLeod et al. Stripe 82 dataset, using DRW simulations as the null hypothesis rather than white noise — a critical methodological distinction.

## Pipeline

| Script | Description |
|--------|-------------|
| `01_download_data.py` | Download and extract all raw data (~30 MB) |
| `02_eda.py` | Exploratory data analysis — light curves, cadence, population statistics |
| `03_variability.py` | Structure function computation and DRW model comparison |
| `04_periodicity.py` | Lomb-Scargle periodicity search with DRW Monte Carlo significance |
| `05_ml.py` | Feature-based ML: RF (full), RF (no LS features), Isolation Forest |
| `06_crossmatch.py` | Cross-match against published SMBHB catalogs (Graham+2015) |

## Key Methodological Points

- **DRW null hypothesis**: Significance assessed against 200 DRW simulations per object (not white noise FAP), using pre-fitted DRW parameters from MacLeod et al.
- **Seasonal alias masking**: Peaks at 365 ± 25 d, 182.5 ± 15 d, and 121.7 ± 10 d are explicitly excluded as Stripe 82 cadence artefacts.
- **3-cycle requirement**: Only periods where ≥ 3 full cycles fit within the baseline are reported (Vaughan et al. 2016 criterion).
- **Three independent ML approaches**: A standard RF classifier, a non-circular RF trained without any LS-derived features, and an unsupervised Isolation Forest — allowing cross-validation of candidates.

## Results Summary

- **9,258 quasars** searched over a period grid of 200–1100 days
- **55 periodic candidates** at >99% DRW significance; **27 at >99.9%**
- **2 novel candidates** flagged by Isolation Forest + LS-feature-free RF but below the formal LS threshold — potential non-sinusoidal signals missed by standard period-finding
- **0 matches** with Graham+2015 (CRTS): only 2 Graham objects fall in the Stripe 82 footprint; consistent with the known low persistence rate of early SMBHB candidates

## LS Limitation Note

This pipeline uses Lomb-Scargle, which assumes sinusoidal signals. Lin, Charisi & Haiman (2026) show that hydrodynamical simulations predict sawtooth-shaped light curves from circumbinary disk interactions, which LS recovers at only ~1–9% efficiency. The unsupervised Isolation Forest component is intended as a step toward more shape-agnostic detection.

## Data

Data is downloaded automatically by `01_download_data.py` from the [MacLeod et al. Stripe 82 catalog](http://faculty.washington.edu/ivezic/macleod/qso_dr7/Southern.html):

- `QSO_S82.tar.gz` — 9,258 individual light curve files
- `DB_QSO_S82.dat.gz` — quasar metadata (redshift, BH mass, luminosity)
- `s82drw.tar.gz` — pre-fitted DRW parameters in all 5 SDSS bands

## Requirements

```bash
pip install numpy pandas matplotlib astropy scikit-learn scipy astroquery
```

Python 3.9+ recommended.

## Usage

```bash
python 01_download_data.py   # ~1 min, downloads ~30 MB
python 02_eda.py             # ~3 min
python 03_variability.py     # ~10 min
python 04_periodicity.py     # ~20 min (LS on 9k objects + 200 MC simulations)
python 05_ml.py              # ~15 min
python 06_crossmatch.py      # ~2 min
```

All plots are saved to `plots/`. Intermediate results (CSVs) are saved to `data/`.

## References

- MacLeod et al. 2012 — Stripe 82 DRW light curves
- Charisi et al. 2016, MNRAS 463, 2145 — PTF periodic quasar candidates
- Graham et al. 2015, MNRAS 453, 1562 — CRTS periodic quasar candidates
- Vaughan et al. 2016 — red noise and spurious periodicity
- Lin, Charisi & Haiman 2026 — LS efficiency for non-sinusoidal SMBHB signals
