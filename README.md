# Stripe 82 SMBHB Search

A pipeline for searching for supermassive black hole binary (SMBHB) candidates in the SDSS Stripe 82 quasar dataset using optical periodicity analysis and machine learning. Built as part of exploring the methodology behind multi-messenger searches for SMBHBs, relevant to projects like [MMMonsters](https://www.ia.forth.gr/).

---

## What is this actually doing?

Quasars vary in brightness over time in a random-looking way that is well described by a stochastic process called a Damped Random Walk (DRW). If a quasar hosts two supermassive black holes orbiting each other (an SMBHB), you would expect to see some periodic signal on top of that random variability, caused by things like the orbital motion, Doppler boosting, or the dynamics of the gas around the binary.

The problem is that random noise, if you stare at it long enough, will occasionally look periodic just by chance. This is especially bad for DRW-type red noise which has more power at long timescales and can easily fool a naive period search. So the main challenge here is not finding periodic signals, it is figuring out which ones are real.

The dataset is the MacLeod et al. Stripe 82 catalog: 9,258 spectroscopically confirmed quasars with optical light curves spanning roughly 10 years, from the SDSS. Observations happen in yearly seasons of about 3 months with nightly cadence, so there are large gaps of about 9 months between seasons every year. This makes period finding harder than it sounds.

---

## Pipeline overview

| Script | What it does |
|--------|-------------|
| `01_download_data.py` | Downloads and extracts the raw data (~30 MB) |
| `02_eda.py` | Basic exploration of the light curves and the quasar population |
| `03_variability.py` | Computes structure functions and compares them to the DRW model |
| `04_periodicity.py` | Runs the Lomb-Scargle period search with proper DRW-based significance |
| `05_ml.py` | Three ML approaches for classifying periodic vs stochastic objects |
| `06_crossmatch.py` | Cross-matches candidates against Graham+2015; follows up on novel detections |

---

## The significance problem

Most published searches assess how significant a Lomb-Scargle peak is by comparing it against white noise. Quasar variability is not white noise, it is red noise, and using the wrong null model means your false alarm probabilities are way off. Here, for each candidate, we simulate 200 DRW light curves for that specific object (sampled at exactly the same times as the real data, using its fitted DRW parameters) and use the distribution of peak powers from those simulations as the significance threshold. A detection has to beat the 99th or 99.9th percentile of that distribution to count.

Two other things that matter a lot: the yearly seasonal gaps create fake peaks in the periodogram at 365 days and its harmonics, so those period ranges are masked out. And any claimed period has to fit at least 3 full cycles within the data baseline, otherwise it is not really constrained.

---

## Results

**Variability:** The population structure functions broadly follow the DRW model but are systematically about 20% lower than predicted at lags around 100-300 days. This is because there are almost no observation pairs at those timescales due to the seasonal gaps, so the structure function estimates there are unreliable.

**Period search:** Searching 8,896 objects over a period grid of 200-1100 days, we find 55 candidates significant at the 99% DRW level and 27 at 99.9%. Candidates mostly cluster around periods of 300-500 days.

**ML:** Three separate approaches were used on purpose to deal with a circularity problem. If you train a classifier whose labels come from the LS peak power and then include LS peak power as a feature, the classifier just learns to replicate what you already computed (ROC-AUC 0.996, but meaningless). The more interesting result is the RF trained without any LS-derived features at all, which still reaches ROC-AUC 0.81 using only light curve shape and variability statistics. This means there is genuine information about periodicity in the photometric properties independent of the period search itself. Kurtosis, skewness, and excess variance are the leading features. The unsupervised Isolation Forest adds a third independent ranking.

Two objects come out as high-scoring in both the LS-free RF and the Isolation Forest while falling just below the formal LS threshold. Both are at z~1.7, have unusually heavy-tailed magnitude distributions, and show rough phase coherence at periods of ~340 and ~409 days respectively. These are worth following up.

**Crossmatch:** Zero matches with Graham+2015, but this is not surprising. Only 2 of their 111 objects even fall in the Stripe 82 footprint to begin with. More importantly, many candidates from that era have since been shown not to persist when observed over longer baselines, which is a known issue with LS-based searches on short datasets.

---

## A known limitation worth mentioning

Lomb-Scargle looks for sinusoidal signals. Recent hydrodynamical simulations of circumbinary accretion disks (Lin, Charisi & Haiman 2026) suggest that real SMBHB light curves are more sawtooth-shaped, and LS only recovers those at about 1-9% efficiency. This means most real SMBHBs would be missed by this pipeline regardless of significance thresholds, which motivates the Isolation Forest approach as a first step toward something more shape-agnostic.

---

## Requirements

```bash
pip install numpy pandas matplotlib astropy scikit-learn scipy astroquery
```

Python 3.9+.

## Running it

```bash
python 01_download_data.py   # ~1 min
python 02_eda.py             # ~3 min
python 03_variability.py     # ~10 min
python 04_periodicity.py     # ~20 min
python 05_ml.py              # ~15 min
python 06_crossmatch.py      # ~2 min
```

Plots go to `plots/`. Intermediate results (CSVs) go to `data/`.

---

## References

- MacLeod et al. 2012, ApJ 753, 106
- Charisi et al. 2016, MNRAS 463, 2145
- Graham et al. 2015, MNRAS 453, 1562
- Vaughan et al. 2016, MNRAS 461, 3145
- Lin, Charisi & Haiman 2026, arXiv:2505.14778
