"""
Download 'denizbilginn/google-maps-restaurant-reviews' from Kaggle via API.

Usage:
    pip install kaggle
    python download_kaggle_dataset.py
"""

import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "denizbilginn/google-maps-restaurant-reviews"
OUTPUT_DIR = Path("data/google_maps_restaurant_reviews")

def ensure_permissions_unix(kaggle_json: Path):
    """Ensure ~/.kaggle/kaggle.json has 0o600 on Unix-like systems."""
    try:
        if os.name != "nt" and kaggle_json.exists():
            os.chmod(kaggle_json, 0o600)
    except Exception as e:
        print(f"Warning: could not set permissions on {kaggle_json}: {e}")

def main():
    # Make sure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Optional: fix permissions for Unix-like systems
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    ensure_permissions_unix(kaggle_json)

    # Authenticate with Kaggle
    api = KaggleApi()
    api.authenticate()  # Will use ~/.kaggle/kaggle.json or env vars

    # Download & unzip
    print(f"Downloading '{DATASET}' to: {OUTPUT_DIR.resolve()}")
    api.dataset_download_files(
        DATASET,
        path=str(OUTPUT_DIR),
        unzip=True,
        quiet=False,
    )

    print("\n Done!")
    print(f"Files are in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()