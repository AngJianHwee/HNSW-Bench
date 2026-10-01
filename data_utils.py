#!/usr/bin/env python3
"""
Utility functions for reading .fvecs and .ivecs files (SIFT1M format).
"""
import numpy as np
import os


def read_fvecs(filepath: str, fraction: float = 1.0) -> np.ndarray:
    """
    Read .fvecs file (float32 vectors).
    
    Format (SIFT1M):
    - For each vector: [dim (int32), v1 (float32), v2 (float32), ..., vdim (float32)]
    - All vectors have the same dimension
    
    Args:
        filepath: Path to .fvecs file
        fraction: Fraction of vectors to read (0.0-1.0). Default 1.0 reads all.
        
    Returns:
        numpy array of shape (n_vectors, dimension) with dtype float32
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0.0, 1.0], got {fraction}")
    
    # Read the entire file as bytes
    with open(filepath, 'rb') as f:
        data = f.read()
    
    if len(data) == 0:
        return np.empty((0, 0), dtype=np.float32)
    
    # Read first 4 bytes to get dimension
    dim = np.frombuffer(data[:4], dtype=np.int32)[0]
    
    if dim <= 0:
        return np.empty((0, 0), dtype=np.float32)
    
    # Each vector takes (1 + dim) * 4 bytes = 4 + dim * 4 bytes
    vector_size = 4 + dim * 4
    n_vectors = len(data) // vector_size
    
    # Apply fraction
    n_vectors_to_read = int(n_vectors * fraction)
    if n_vectors_to_read == 0:
        n_vectors_to_read = 1
    
    # Pre-allocate output array
    vectors = np.empty((n_vectors_to_read, dim), dtype=np.float32)
    
    # Parse each vector
    offset = 0
    for i in range(n_vectors_to_read):
        # Skip dimension (4 bytes), read vector data
        vector_data = np.frombuffer(data[offset + 4:offset + vector_size], dtype=np.float32)
        vectors[i] = vector_data
        offset += vector_size
    
    return vectors


def read_ivecs(filepath: str, fraction: float = 1.0) -> np.ndarray:
    """
    Read .ivecs file (int32 vectors, typically ground truth indices).
    
    Format (SIFT1M):
    - For each vector: [dim (int32), v1 (int32), v2 (int32), ..., vdim (int32)]
    - All vectors have the same dimension
    
    Args:
        filepath: Path to .ivecs file
        fraction: Fraction of vectors to read (0.0-1.0). Default 1.0 reads all.
        
    Returns:
        numpy array of shape (n_vectors, dimension) with dtype int32
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0.0, 1.0], got {fraction}")
    
    # Read the entire file as bytes
    with open(filepath, 'rb') as f:
        data = f.read()
    
    if len(data) == 0:
        return np.empty((0, 0), dtype=np.int32)
    
    # Read first 4 bytes to get dimension
    dim = np.frombuffer(data[:4], dtype=np.int32)[0]
    
    if dim <= 0:
        return np.empty((0, 0), dtype=np.int32)
    
    # Each vector takes (1 + dim) * 4 bytes = 4 + dim * 4 bytes
    vector_size = 4 + dim * 4
    n_vectors = len(data) // vector_size
    
    # Apply fraction
    n_vectors_to_read = int(n_vectors * fraction)
    if n_vectors_to_read == 0:
        n_vectors_to_read = 1
    
    # Pre-allocate output array
    vectors = np.empty((n_vectors_to_read, dim), dtype=np.int32)
    
    # Parse each vector
    offset = 0
    for i in range(n_vectors_to_read):
        # Skip dimension (4 bytes), read vector data
        vector_data = np.frombuffer(data[offset + 4:offset + vector_size], dtype=np.int32)
        vectors[i] = vector_data
        offset += vector_size
    
    return vectors


def write_fvecs(filepath: str, vectors: np.ndarray) -> None:
    """
    Write vectors to .fvecs file (SIFT1M format).
    
    Format:
    - For each vector: [dim (int32), v1 (float32), v2 (float32), ..., vdim (float32)]
    - All vectors have the same dimension
    
    Args:
        filepath: Output path
        vectors: numpy array of shape (n_vectors, dimension) with dtype float32
    """
    vectors = vectors.astype(np.float32)
    n_vectors, dim = vectors.shape
    
    with open(filepath, 'wb') as f:
        # Write each vector with its dimension prefix (SIFT1M format)
        for i in range(n_vectors):
            np.array([dim], dtype=np.int32).tofile(f)
            vectors[i].tofile(f)


def write_ivecs(filepath: str, vectors: np.ndarray) -> None:
    """
    Write vectors to .ivecs file (SIFT1M format).
    
    Format:
    - For each vector: [dim (int32), v1 (int32), v2 (int32), ..., vdim (int32)]
    - All vectors have the same dimension
    
    Args:
        filepath: Output path
        vectors: numpy array of shape (n_vectors, dimension) with dtype int32
    """
    vectors = vectors.astype(np.int32)
    n_vectors, dim = vectors.shape
    
    with open(filepath, 'wb') as f:
        # Write each vector with its dimension prefix (SIFT1M format)
        for i in range(n_vectors):
            np.array([dim], dtype=np.int32).tofile(f)
            vectors[i].tofile(f)


def get_dataset_info(data_dir: str = "data") -> dict:
    """
    Get information about the SIFT1M dataset files.
    
    Args:
        data_dir: Directory containing the dataset files
        
    Returns:
        Dictionary with file info
    """
    files = {
        "base": "sift_base.fvecs",
        "query": "sift_query.fvecs",
        "groundtruth": "sift_groundtruth.ivecs",
        "learn": "sift_learn.fvecs",
    }
    
    info = {}
    for key, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            if key == "groundtruth":
                data = read_ivecs(filepath)
            else:
                data = read_fvecs(filepath)
            info[key] = {
                "path": filepath,
                "shape": data.shape,
                "dtype": str(data.dtype),
                "size_mb": data.nbytes / (1024 * 1024)
            }
        else:
            info[key] = {"path": filepath, "exists": False}
    
    return info


if __name__ == "__main__":
    # Test the utilities
    info = get_dataset_info("data")
    for key, val in info.items():
        if "shape" in val:
            print(f"{key}: {val['shape']} ({val['dtype']}) - {val['size_mb']:.2f} MB")
        else:
            print(f"{key}: NOT FOUND")