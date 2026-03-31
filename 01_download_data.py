"""
01_download_data.py
Download and extract the Stripe 82 quasar light curve data from MacLeod et al.
"""

import urllib.request
import tarfile
import gzip
import shutil
import os
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

FILES = {
    "QSO_S82.tar.gz":    "http://faculty.washington.edu/ivezic/macleod/qso_dr7/QSO_S82.tar.gz",
    "DB_QSO_S82.dat.gz": "http://faculty.washington.edu/ivezic/macleod/qso_dr7/DB_QSO_S82.dat.gz",
    "s82drw.tar.gz":     "http://faculty.washington.edu/ivezic/macleod/qso_dr7/s82drw.tar.gz",
}


def download(url, dest):
    print(f"  Downloading {dest.name} ...", end=" ", flush=True)
    urllib.request.urlretrieve(url, dest)
    size_mb = dest.stat().st_size / 1e6
    print(f"done ({size_mb:.1f} MB)")


def extract_tar(path, extract_to):
    print(f"  Extracting {path.name} ...", end=" ", flush=True)
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(extract_to)
    print("done")


def extract_gz(path, dest):
    print(f"  Extracting {path.name} ...", end=" ", flush=True)
    with gzip.open(path, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print("done")


# --- Download ---
for filename, url in FILES.items():
    dest = DATA_DIR / filename
    if dest.exists():
        print(f"  {filename} already exists, skipping download.")
    else:
        download(url, dest)

# --- Extract QSO_S82.tar.gz -> data/lightcurves/ ---
# The tar extracts to a subdirectory named QSO_S82; light curve files have no extension.
lc_dir = DATA_DIR / "lightcurves"
if lc_dir.exists():
    print(f"  {lc_dir} already exists, skipping extraction.")
else:
    extract_tar(DATA_DIR / "QSO_S82.tar.gz", DATA_DIR)
    extracted_subdir = DATA_DIR / "QSO_S82"
    if extracted_subdir.exists():
        extracted_subdir.rename(lc_dir)
        print(f"  Renamed QSO_S82 -> lightcurves/")
    else:
        # Fallback: find any new directory with many files
        candidates = [d for d in DATA_DIR.iterdir()
                      if d.is_dir() and d.name not in ("lightcurves", "drw")]
        for d in candidates:
            if len(list(d.iterdir())) > 100:
                d.rename(lc_dir)
                print(f"  Renamed {d.name} -> lightcurves/")
                break

# --- Extract DB_QSO_S82.dat.gz ---
db_dest = DATA_DIR / "DB_QSO_S82.dat"
if db_dest.exists():
    print(f"  {db_dest.name} already exists, skipping extraction.")
else:
    extract_gz(DATA_DIR / "DB_QSO_S82.dat.gz", db_dest)

# --- Extract s82drw.tar.gz -> data/drw/ ---
# The tar extracts 5 loose files (s82drw_u.dat etc.) directly into the target dir.
drw_dir = DATA_DIR / "drw"
if drw_dir.exists():
    print(f"  {drw_dir} already exists, skipping extraction.")
else:
    drw_dir.mkdir()
    extract_tar(DATA_DIR / "s82drw.tar.gz", drw_dir)
    print(f"  Extracted DRW files -> drw/")

# --- Verify ---
print("\n--- Verification ---")
n_lc = len(list(lc_dir.glob("*"))) if lc_dir.exists() else 0
print(f"  Light curve files : {n_lc}")
print(f"  DB metadata file  : {db_dest.exists()}")
print(f"  DRW directory     : {drw_dir.exists()}, files: {len(list(drw_dir.glob('*'))) if drw_dir.exists() else 0}")
print("\nDone. Ready for 02_eda.py")
