#!/usr/bin/env python3
"""
Download SIFT1M dataset from official source.
Dataset: http://corpus-texmex.irisa.fr/
"""
import os
import sys
import requests
import gzip
import shutil
from tqdm import tqdm
from config_loader import get_config


# Try multiple mirrors for SIFT1M dataset
# Hugging Face (qbo-odp/sift1m) is the most reliable
SIFT1M_URLS = {
    "sift_base.fvecs": [
        "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_base.fvecs",
        "https://github.com/facebookresearch/faiss/raw/main/benchs/sift1M/sift_base.fvecs.gz",
        "ftp://ftp.irisa.fr/local/texmex/corpus/sift1M/sift_base.fvecs.gz",
        "http://corpus-texmex.irisa.fr/sift1M/sift_base.fvecs.gz",
    ],
    "sift_query.fvecs": [
        "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_query.fvecs",
        "https://github.com/facebookresearch/faiss/raw/main/benchs/sift1M/sift_query.fvecs.gz",
        "ftp://ftp.irisa.fr/local/texmex/corpus/sift1M/sift_query.fvecs.gz",
        "http://corpus-texmex.irisa.fr/sift1M/sift_query.fvecs.gz",
    ],
    "sift_groundtruth.ivecs": [
        "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_groundtruth.ivecs",
        "https://github.com/facebookresearch/faiss/raw/main/benchs/sift1M/sift_groundtruth.ivecs.gz",
        "ftp://ftp.irisa.fr/local/texmex/corpus/sift1M/sift_groundtruth.ivecs.gz",
        "http://corpus-texmex.irisa.fr/sift1M/sift_groundtruth.ivecs.gz",
    ],
    "sift_learn.fvecs": [
        "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_learn.fvecs",
        "https://github.com/facebookresearch/faiss/raw/main/benchs/sift1M/sift_learn.fvecs.gz",
        "ftp://ftp.irisa.fr/local/texmex/corpus/sift1M/sift_learn.fvecs.gz",
        "http://corpus-texmex.irisa.fr/sift1M/sift_learn.fvecs.gz",
    ],
}


def download_file(url: str, dest_path: str) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def extract_gz(gz_path: str, dest_path: str) -> None:
    """Extract a .gz file."""
    print(f"Extracting {gz_path} -> {dest_path}")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(dest_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)


def download_with_fallback(urls: list, dest_path: str) -> bool:
    """Try downloading from multiple URLs until one succeeds."""
    for i, url in enumerate(urls):
        try:
            print(f"  Trying mirror {i+1}/{len(urls)}: {url}")
            download_file(url, dest_path)
            return True
        except Exception as e:
            print(f"  Failed: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            continue
    return False


def main():
    cfg = get_config()
    data_dir = os.path.dirname(cfg.data.base_path)
    os.makedirs(data_dir, exist_ok=True)
    
    for filename, urls in SIFT1M_URLS.items():
        dest_path = os.path.join(data_dir, filename)
        
        if os.path.exists(dest_path):
            print(f"{filename} already exists, skipping...")
            continue
            
        print(f"Downloading {filename}...")
        # Try downloading directly (some mirrors serve uncompressed)
        success = download_with_fallback(urls, dest_path)
        if not success:
            print(f"ERROR: All mirrors failed for {filename}")
            sys.exit(1)
        
        # Check if file is gzipped and extract if needed
        if dest_path.endswith('.gz') or is_gzipped(dest_path):
            gz_path = dest_path if dest_path.endswith('.gz') else dest_path + '.gz'
            if not dest_path.endswith('.gz'):
                os.rename(dest_path, gz_path)
            extract_gz(gz_path, dest_path)
        print(f"Done: {filename}")
    
    print("\nAll files downloaded and extracted!")
    print(f"Data directory: {os.path.abspath(data_dir)}")


def is_gzipped(filepath: str) -> bool:
    """Check if a file is gzipped by reading magic bytes."""
    with open(filepath, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'


if __name__ == "__main__":
    main()