"""
Download and prepare a subset of SPACEV-1B.

SPACEV-1B binary format (.i8bin / .ibin):
  - bytes 0-3:  uint32 num_vectors
  - bytes 4-7:  uint32 dimension
  - bytes 8+:   raw int8 (base/query) or int32 (ground truth) data

For the base file (~93 GB total), we use an HTTP Range request to fetch
only the bytes needed for SUBSET_SIZE vectors, avoiding a full download.
"""

import os
import struct
import urllib.request
import numpy as np

# ---------------------------------------------------------------------------
# Configurable constants
# ---------------------------------------------------------------------------

_S3 = "https://bigger-ann.s3.amazonaws.com/spacev-1b"
BASE_URL = f"{_S3}/base.1B.i8bin"
QUERY_URL = f"{_S3}/query.30K.i8bin"
GT_URL = f"{_S3}/groundtruth.30K.i32bin"

DATA_DIR = os.environ.get("SPACEV_DATA_DIR", "./data")
DIM = 100
SUBSET_SIZE = 10_000_000  # 10M for e2e testing; scale up later


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _progress(downloaded: int, total: int) -> None:
    if total > 0:
        pct = min(downloaded / total * 100, 100)
        print(f"\r  {pct:.1f}%  ({downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB)",
              end="", flush=True)


def _download_full(url: str, dest: str) -> None:
    """Download a full file with a progress hook."""
    if os.path.exists(dest):
        print(f"  already exists: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {url}")

    def _hook(blocks, block_size, total):
        _progress(blocks * block_size, total)

    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print()


def _download_range(url: str, dest: str, num_bytes: int) -> None:
    """Download only the first num_bytes of url using HTTP Range."""
    if os.path.exists(dest):
        print(f"  already exists: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading first {num_bytes / 1e6:.1f} MB from {url}")

    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{num_bytes - 1}"})
    chunk = 1 << 20  # 1 MB
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        downloaded = 0
        while True:
            data = resp.read(chunk)
            if not data:
                break
            f.write(data)
            downloaded += len(data)
            _progress(downloaded, num_bytes)
    print()


def read_i8bin(path: str, max_rows: int | None = None) -> np.ndarray:
    """Read a .i8bin file into an int8 ndarray of shape (N, D)."""
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
    n = min(n, max_rows) if max_rows is not None else n
    data = np.memmap(path, dtype="int8", mode="r", offset=8, shape=(n, d))
    return np.array(data)


def read_ibin(path: str, max_rows: int | None = None) -> np.ndarray:
    """Read a .ibin file into an int32 ndarray of shape (N, K)."""
    with open(path, "rb") as f:
        n, k = struct.unpack("<II", f.read(8))
    n = min(n, max_rows) if max_rows is not None else n
    data = np.memmap(path, dtype="int32", mode="r", offset=8, shape=(n, k))
    return np.array(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def download_and_prepare(
    data_dir: str = DATA_DIR,
    subset_size: int = SUBSET_SIZE,
) -> dict[str, str]:
    """
    Fetch SPACEV-1B base (partial), query, and ground-truth files, then save
    as .npy for fast reloading.

    Returns {'base': path, 'query': path, 'gt': path}.
    """
    os.makedirs(data_dir, exist_ok=True)
    raw_dir = os.path.join(data_dir, "raw")

    raw_base = os.path.join(raw_dir, f"spacev_base_{subset_size // 1_000_000}M.i8bin")
    raw_query = os.path.join(raw_dir, "query.i8bin")
    raw_gt = os.path.join(raw_dir, "groundtruth.30K.i32bin")

    out_base = os.path.join(data_dir, f"base_{subset_size // 1_000_000}M.npy")
    out_query = os.path.join(data_dir, "query.npy")
    out_gt = os.path.join(data_dir, "gt100.npy")

    # Base: use Range request — header (8 bytes) + subset_size * DIM bytes
    base_bytes = 8 + subset_size * DIM
    _download_range(BASE_URL, raw_base, base_bytes)

    # Query + GT: small, download in full
    _download_full(QUERY_URL, raw_query)
    _download_full(GT_URL, raw_gt)

    # Convert to .npy
    if not os.path.exists(out_base):
        print(f"Loading {subset_size:,} base vectors ...")
        base = read_i8bin(raw_base, max_rows=subset_size)
        assert base.shape == (subset_size, DIM), f"Unexpected shape: {base.shape}"
        np.save(out_base, base)
        print(f"Saved {base.shape} int8 → {out_base}")

    if not os.path.exists(out_query):
        query = read_i8bin(raw_query)
        np.save(out_query, query)
        print(f"Saved {query.shape} int8 → {out_query}")

    if not os.path.exists(out_gt):
        gt = read_ibin(raw_gt)
        np.save(out_gt, gt)
        print(f"Saved {gt.shape} int32 → {out_gt}")

    return {"base": out_base, "query": out_query, "gt": out_gt}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--subset-size", type=int, default=SUBSET_SIZE)
    args = parser.parse_args()
    paths = download_and_prepare(args.data_dir, args.subset_size)
    print("Ready:", paths)
