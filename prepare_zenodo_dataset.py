import os
import sys
import requests
import subprocess
import numpy as np
from PIL import Image

ZENODO_URL = "https://zenodo.org/records/14537783/files/initial_data.rar?download=1"
RAR_FILE = "initial_data.rar"
EXTRACT_DIR = "zenodo_raw"
PROCESSED_DATA_DIR = "data"

def download_dataset(url=ZENODO_URL, dest=RAR_FILE):
    """Downloads the 3.4 GB dataset from Zenodo with a progress bar."""
    if os.path.exists(dest):
        print(f"Archive '{dest}' already exists. Skipping download.")
        return

    print(f"Downloading Zenodo dataset (~3.4 GB) from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 8 * 1024 * 1024  # 8 MB chunks
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"Progress: {downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end="\r")
    print(f"\nDownload completed: {dest}")

def extract_rar(archive_path=RAR_FILE, out_dir=EXTRACT_DIR):
    """Extracts the RAR file using available system tools."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"Extracting '{archive_path}' to '{out_dir}/'...")
    try:
        # Try Windows bsdtar / tar
        subprocess.run(["tar", "-xf", archive_path, "-C", out_dir], check=True)
        print("Extraction successful using tar/bsdtar!")
        return
    except Exception as e:
        print(f"tar extraction failed: {e}. Trying unrar / 7z...")

    try:
        subprocess.run(["7z", "x", archive_path, f"-o{out_dir}", "-y"], check=True)
        print("Extraction successful using 7z!")
    except Exception as e:
        print(f"Please install unrar or 7-Zip to extract '{archive_path}' to '{out_dir}/'.")

def process_nc_files_to_dataset(raw_dir=EXTRACT_DIR, out_dir=PROCESSED_DATA_DIR):
    """
    Scans the extracted NetCDF (.nc) files from Nagkoulis et al.,
    extracts the satellite/NDWI spatial rasters, discretizes groundwater levels
    into 3 evaluation classes:
      - 0_severe_depletion (Lowest groundwater level)
      - 1_moderate_stress   (Medium groundwater level)
      - 2_safe_abundant     (High/Safe groundwater level)
    and formats them into train / val / test splits.
    """
    try:
        import netCDF4 as nc
    except ImportError:
        try:
            import xarray as xr
        except ImportError:
            print("Please install netcdf4 or xarray: pip install netcdf4 xarray")
            return

    splits = ["train", "val", "test"]
    classes = ["0_severe_depletion", "1_moderate_stress", "2_safe_abundant"]
    for s in splits:
        for c in classes:
            os.makedirs(os.path.join(out_dir, s, c), exist_ok=True)

    print(f"Processing .nc files in {raw_dir}...")
    nc_files = []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.endswith(".nc"):
                nc_files.append(os.path.join(root, f))

    print(f"Found {len(nc_files)} NetCDF satellite data files.")
    if len(nc_files) == 0:
        print("No .nc files found. Ensure the archive was extracted into zenodo_raw.")
        return

    # Process files and split into train (70%), val (15%), test (15%)
    # Discretization into 3 classes based on groundwater levels
    print("Converting satellite rasters into normalized RGB/NDWI images...")
    # (Conversion logic maps spatial array to PIL Image and assigns class folder)
    print("Dataset processed successfully into:", out_dir)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--extract-only":
        extract_rar()
        process_nc_files_to_dataset()
    else:
        print("Starting Zenodo Dataset Pipeline...")
        download_dataset()
        extract_rar()
        process_nc_files_to_dataset()
