#!/usr/bin/env python3
"""
Phase 3: HNSW Index Construction and Parameter Sweep.
Tests ef_search parameter sweep with fixed M=16, ef_construction=64.
"""
import os
import time
import numpy as np
import faiss
import psutil
import pandas as pd
from data_utils import read_fvecs
from phase2_groundtruth import recall_at_k_vectorized, load_groundtruth
from config_loader import get_config


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_hnsw_experiment(
    base_path: str | None = None,
    query_path: str | None = None,
    gt_path: str | None = None,
    M: int | None = None,
    ef_construction: int | None = None,
    ef_search_values: list | None = None,
    k: int | None = None
) -> pd.DataFrame:
    """
    Run HNSW parameter sweep over ef_search values.
    
    Args:
        base_path: Path to base vectors (uses config if None)
        query_path: Path to query vectors (uses config if None)
        gt_path: Path to ground truth (uses config if None)
        M: HNSW M parameter (uses config if None)
        ef_construction: HNSW ef_construction parameter (uses config if None)
        ef_search_values: List of ef_search values to test (uses config if None)
        k: Top-k for recall calculation (uses config if None)
        
    Returns:
        DataFrame with results for each ef_search
    """
    cfg = get_config()
    
    # Resolve config values
    resolved_base_path = base_path or cfg.data.base_path
    resolved_query_path = query_path or cfg.data.query_path
    resolved_gt_path = gt_path or cfg.data.gt_path
    resolved_M = M or cfg.hnsw.M
    resolved_ef_construction = ef_construction or cfg.hnsw.ef_construction
    resolved_ef_search_values = ef_search_values or cfg.hnsw.ef_search_values
    resolved_k = k or cfg.hnsw.k
    
    print("=" * 60)
    print(f"Phase 3: HNSW Index (M={resolved_M}, ef_construction={resolved_ef_construction})")
    print("=" * 60)
    
    # Load data
    print("Loading base vectors...")
    base_vectors = read_fvecs(resolved_base_path)
    print(f"Base shape: {base_vectors.shape}")
    
    print("Loading query vectors...")
    query_vectors = read_fvecs(resolved_query_path)
    print(f"Query shape: {query_vectors.shape}")
    
    print("Loading ground truth...")
    groundtruth = load_groundtruth(resolved_gt_path)
    print(f"Ground truth shape: {groundtruth.shape}")
    
    d = base_vectors.shape[1]
    n_queries = query_vectors.shape[0]
    
    # Build HNSW index
    print(f"\nBuilding IndexHNSWFlat (M={resolved_M}, ef_construction={resolved_ef_construction})...")
    mem_before = get_memory_usage_mb()
    start_build = time.time()
    
    index = faiss.IndexHNSWFlat(d, resolved_M)
    index.hnsw.efConstruction = resolved_ef_construction
    index.add(base_vectors)
    
    build_time = time.time() - start_build
    mem_after_build = get_memory_usage_mb()
    index_mem = mem_after_build - mem_before
    
    print(f"Index built: {index.ntotal} vectors")
    print(f"Build time: {build_time:.2f}s")
    print(f"Index memory: {index_mem:.2f} MB")
    
    # Run parameter sweep
    results = []
    
    # Validate query count match between groundtruth and query vectors
    n_gt_queries = groundtruth.shape[0]
    if n_gt_queries != n_queries:
        print(f"\nWARNING: Ground truth has {n_gt_queries} queries, but query set has {n_queries} queries.")
        print(f"Truncating query vectors to match ground truth ({n_gt_queries} queries).")
        query_vectors = query_vectors[:n_gt_queries]
        n_queries = n_gt_queries
    
    for ef_search in resolved_ef_search_values:
        print(f"\n--- ef_search = {ef_search} ---")
        index.hnsw.efSearch = ef_search
        
        # Measure peak memory during search
        mem_before_search = get_memory_usage_mb()
        start_search = time.time()
        
        distances, labels = index.search(query_vectors, resolved_k)
        
        search_time = time.time() - start_search
        mem_after_search = get_memory_usage_mb()
        peak_mem = max(mem_after_build, mem_after_search)
        
        qps = n_queries / search_time
        recall = recall_at_k_vectorized(groundtruth, labels, resolved_k)
        
        print(f"  Search time: {search_time:.3f}s")
        print(f"  QPS: {qps:.2f}")
        print(f"  Recall@{resolved_k}: {recall:.4f} ({recall*100:.2f}%)")
        print(f"  Peak memory: {peak_mem:.2f} MB")
        
        results.append({
            'algorithm': 'HNSW',
            'M': resolved_M,
            'ef_construction': resolved_ef_construction,
            'ef_search': ef_search,
            'build_time_s': build_time,
            'search_time_s': search_time,
            'qps': qps,
            f'recall@{resolved_k}': recall,
            'peak_memory_mb': peak_mem,
            'index_memory_mb': index_mem
        })
    
    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    df = run_hnsw_experiment()
    print("\n" + "=" * 60)
    print("HNSW RESULTS SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    
    # Save to CSV
    cfg = get_config()
    output_path = cfg.output.hnsw_results
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")