#!/usr/bin/env python3
"""
Phase 2: Build Flat index (brute-force) and compute Ground Truth.
Also includes Recall@k calculation function.
"""
import os
import time
import numpy as np
import faiss
import psutil
from data_utils import read_fvecs, read_ivecs, write_ivecs
from config_loader import get_config


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def recall_at_k(groundtruth: np.ndarray, predictions: np.ndarray, k: int = 10) -> float:
    """
    Calculate Recall@k.
    
    Args:
        groundtruth: Shape (n_queries, k) - true nearest neighbor indices
        predictions: Shape (n_queries, k) - predicted nearest neighbor indices
        k: Number of top results to consider
        
    Returns:
        Recall@k as float (0.0 to 1.0)
    """
    n_queries = groundtruth.shape[0]
    total_recall = 0.0
    
    for i in range(n_queries):
        gt_set = set(groundtruth[i, :k])
        pred_set = set(predictions[i, :k])
        intersection = gt_set & pred_set
        total_recall += len(intersection) / k
    
    return total_recall / n_queries


def recall_at_k_vectorized(groundtruth: np.ndarray, predictions: np.ndarray, k: int = 10) -> float:
    """
    Vectorized Recall@k calculation (faster for large datasets).
    
    Args:
        groundtruth: Shape (n_queries, k) - true nearest neighbor indices
        predictions: Shape (n_queries, k) - predicted nearest neighbor indices
        k: Number of top results to consider
        
    Returns:
        Recall@k as float (0.0 to 1.0)
        
    Raises:
        ValueError: If groundtruth and predictions have different number of queries
    """
    # Validate that both arrays have the same number of queries
    if groundtruth.shape[0] != predictions.shape[0]:
        raise ValueError(
            f"Query count mismatch: groundtruth has {groundtruth.shape[0]} queries, "
            f"but predictions has {predictions.shape[0]} queries. "
            f"Ensure ground truth was computed for all queries."
        )
    
    # Use broadcasting to compare each prediction against all groundtruth for each query
    # Shape: (n_queries, k, k) -> True where prediction matches groundtruth
    matches = predictions[:, :k, np.newaxis] == groundtruth[:, np.newaxis, :k]
    # Any match along the groundtruth axis (axis=2)
    hits = matches.any(axis=2).sum(axis=1)  # Shape: (n_queries,)
    return hits.mean() / k


def build_flat_index_and_compute_gt(
    base_path: str | None = None,
    query_path: str | None = None,
    gt_path: str | None = None,
    k: int | None = None
) -> tuple:
    """
    Build IndexFlatL2, compute ground truth for all queries.
    
    Args:
        base_path: Path to base vectors (uses config if None)
        query_path: Path to query vectors (uses config if None)
        gt_path: Output path for ground truth (uses config if None)
        k: Number of nearest neighbors (uses config if None)
        
    Returns:
        (groundtruth_labels, groundtruth_distances, build_time, search_time, peak_memory_mb)
    """
    cfg = get_config()
    
    # Resolve config values
    resolved_base_path = base_path or cfg.data.base_path
    resolved_query_path = query_path or cfg.data.query_path
    resolved_gt_path = gt_path or cfg.data.gt_path
    resolved_k = k or cfg.groundtruth.k
    data_fraction = cfg.data.data_fraction
    
    print("=" * 60)
    print("Phase 2: Building Flat Index & Computing Ground Truth")
    print("=" * 60)
    print(f"data_fraction: {data_fraction}")
    
    # Load data
    print("Loading base vectors...")
    base_vectors = read_fvecs(resolved_base_path, fraction=data_fraction)
    print(f"Base shape: {base_vectors.shape}")
    
    print("Loading query vectors...")
    query_vectors = read_fvecs(resolved_query_path, fraction=data_fraction)
    print(f"Query shape: {query_vectors.shape}")
    
    d = base_vectors.shape[1]
    
    # Build Flat index
    print("\nBuilding IndexFlatL2...")
    mem_before = get_memory_usage_mb()
    start_build = time.time()
    
    index = faiss.IndexFlatL2(d)
    index.add(base_vectors)
    
    build_time = time.time() - start_build
    mem_after = get_memory_usage_mb()
    peak_mem = mem_after  # For flat index, memory is stable after build
    
    print(f"Index built: {index.ntotal} vectors")
    print(f"Build time: {build_time:.2f}s")
    print(f"Memory usage: {mem_after - mem_before:.2f} MB (total: {mem_after:.2f} MB)")
    
    # Search for ground truth
    print(f"\nSearching Top-{resolved_k} for {query_vectors.shape[0]} queries...")
    start_search = time.time()
    
    distances, labels = index.search(query_vectors, resolved_k)
    
    search_time = time.time() - start_search
    qps = query_vectors.shape[0] / search_time
    
    print(f"Search time: {search_time:.2f}s")
    print(f"QPS: {qps:.2f}")
    
    # Save ground truth
    print(f"\nSaving ground truth to {resolved_gt_path}...")
    write_ivecs(resolved_gt_path, labels)
    
    # Also save distances for reference
    dist_path = resolved_gt_path.replace('.ivecs', '_distances.fvecs')
    # Convert distances to float32 and save with dimension header
    with open(dist_path, 'wb') as f:
        np.array([resolved_k], dtype=np.int32).tofile(f)
        distances.astype(np.float32).tofile(f)
    print(f"Saved distances to {dist_path}")
    
    return labels, distances, build_time, search_time, peak_mem


def load_groundtruth(gt_path: str | None = None) -> np.ndarray:
    """Load precomputed ground truth."""
    cfg = get_config()
    resolved_gt_path = gt_path or cfg.data.gt_path
    return read_ivecs(resolved_gt_path)


def evaluate_recall(
    predictions: np.ndarray,
    groundtruth: np.ndarray,
    k: int | None = None
) -> float:
    """
    Evaluate Recall@k for given predictions against ground truth.
    
    Args:
        predictions: Shape (n_queries, k) - predicted indices
        groundtruth: Shape (n_queries, k) - true indices
        k: Top-k to evaluate (uses config if None)
        
    Returns:
        Recall@k score
        
    Raises:
        ValueError: If groundtruth and predictions have different number of queries
    """
    cfg = get_config()
    resolved_k = k or cfg.groundtruth.k
    
    # Validate query count match
    if groundtruth.shape[0] != predictions.shape[0]:
        raise ValueError(
            f"Query count mismatch: groundtruth has {groundtruth.shape[0]} queries, "
            f"but predictions has {predictions.shape[0]} queries. "
            f"Ensure ground truth was computed for all queries. "
            f"Re-run Phase 2 with the full query set."
        )
    
    return recall_at_k_vectorized(groundtruth, predictions, resolved_k)


if __name__ == "__main__":
    # Run ground truth computation
    labels, distances, build_t, search_t, mem = build_flat_index_and_compute_gt()
    
    print("\n" + "=" * 60)
    print("GROUND TRUTH COMPUTATION COMPLETE")
    print("=" * 60)
    print(f"Build time: {build_t:.2f}s")
    print(f"Search time: {search_t:.2f}s")
    print(f"QPS: {10000/search_t:.2f}")
    print(f"Peak memory: {mem:.2f} MB")
    print(f"Ground truth shape: {labels.shape}")
    print(f"First query top-5: {labels[0, :5]}")
    print(f"First query distances: {distances[0, :5]}")