#!/usr/bin/env python3
"""
Phase 3: HNSW Index Construction and Parameter Sweep.
Tests grid search over M, ef_construction, and ef_search parameters.
"""
import os
import time
import gc
import numpy as np
import pandas as pd
import faiss
from itertools import product
from data_utils import read_fvecs
from phase2_groundtruth import recall_at_k_vectorized, load_groundtruth
from config_loader import get_config


def get_index_memory_mb(index) -> float:
    """Get index memory in MB by serializing to bytes."""
    serialized = faiss.serialize_index(index)
    return serialized.nbytes / (1024 * 1024)


def run_hnsw_experiment(
    base_path: str | None = None,
    query_path: str | None = None,
    gt_path: str | None = None,
    M_values: list | None = None,
    ef_construction_values: list | None = None,
    ef_search_values: list | None = None,
    k: int | None = None
) -> pd.DataFrame:
    """
    Run HNSW grid search over M, ef_construction, and ef_search values.
    
    Args:
        base_path: Path to base vectors (uses config if None)
        query_path: Path to query vectors (uses config if None)
        gt_path: Path to ground truth (uses config if None)
        M_values: List of HNSW M parameters to test (uses config if None)
        ef_construction_values: List of HNSW ef_construction parameters to test (uses config if None)
        ef_search_values: List of ef_search values to test (uses config if None)
        k: Top-k for recall calculation (uses config if None)
        
    Returns:
        DataFrame with results for each parameter combination
    """
    cfg = get_config()
    
    # Resolve config values
    resolved_base_path = base_path or cfg.data.base_path
    resolved_query_path = query_path or cfg.data.query_path
    resolved_gt_path = gt_path or cfg.data.gt_path
    resolved_M_values = M_values or cfg.hnsw.M_values
    resolved_ef_construction_values = ef_construction_values or cfg.hnsw.ef_construction_values
    resolved_ef_search_values = ef_search_values or cfg.hnsw.ef_search_values
    resolved_k = k or cfg.hnsw.k
    data_fraction = cfg.data.data_fraction
    
    print("=" * 70)
    print(f"Phase 3: HNSW Grid Search")
    print(f"  M values: {resolved_M_values}")
    print(f"  ef_construction values: {resolved_ef_construction_values}")
    print(f"  ef_search values: {resolved_ef_search_values}")
    print(f"  k: {resolved_k}")
    print(f"  data_fraction: {data_fraction}")
    print("=" * 70)
    
    # Load data once
    print("Loading base vectors...")
    base_vectors = read_fvecs(resolved_base_path, fraction=data_fraction)
    print(f"  Base shape: {base_vectors.shape}")
    
    print("Loading query vectors...")
    query_vectors = read_fvecs(resolved_query_path, fraction=data_fraction)
    print(f"  Query shape: {query_vectors.shape}")
    
    print("Loading ground truth...")
    groundtruth = load_groundtruth(resolved_gt_path)
    print(f"  Ground truth shape: {groundtruth.shape}")
    
    n_queries = query_vectors.shape[0]
    n_gt_queries = groundtruth.shape[0]
    if n_gt_queries != n_queries:
        print(f"  WARNING: Ground truth has {n_gt_queries} queries, query set has {n_queries} queries.")
        print(f"  Will truncate query vectors to match ground truth ({n_gt_queries} queries).")
        query_vectors = query_vectors[:n_gt_queries]
        n_queries = n_gt_queries
    
    d = base_vectors.shape[1]
    
    # Run grid search
    results = []
    total_combinations = len(resolved_M_values) * len(resolved_ef_construction_values) * len(resolved_ef_search_values)
    combo_idx = 0
    
    for M, ef_construction in product(resolved_M_values, resolved_ef_construction_values):
        combo_idx += 1
        print(f"\n{'=' * 70}")
        print(f"[{combo_idx}/{len(resolved_M_values) * len(resolved_ef_construction_values)}] Building IndexHNSWFlat (M={M}, ef_construction={ef_construction})")
        print(f"{'=' * 70}")
        
        # Build index once per (M, ef_construction) combination
        gc.collect()
        start_build = time.time()
        
        index = faiss.IndexHNSWFlat(d, M)
        index.hnsw.efConstruction = ef_construction
        index.add(base_vectors)
        
        build_time = time.time() - start_build
        
        # Get index memory by serialization
        index_mem = get_index_memory_mb(index)
        
        print(f"  Build time: {build_time:.2f}s")
        print(f"  Index memory: {index_mem:.2f} MB")
        
        # Sweep ef_search for this index
        for ef_search in resolved_ef_search_values:
            print(f"\n  [ef_search={ef_search}] Running search...")
            try:
                index.hnsw.efSearch = ef_search
                start_search = time.time()
                
                distances, labels = index.search(query_vectors, resolved_k)
                
                search_time = time.time() - start_search
                
                qps = n_queries / search_time
                recall = recall_at_k_vectorized(groundtruth, labels, resolved_k)
                
                print(f"    Search time: {search_time:.3f}s")
                print(f"    QPS: {qps:.2f}")
                print(f"    Recall@{resolved_k}: {recall:.4f} ({recall*100:.2f}%)")
                
                results.append({
                    'algorithm': 'HNSW',
                    'M': M,
                    'ef_construction': ef_construction,
                    'ef_search': ef_search,
                    'build_time_s': build_time,
                    'search_time_s': search_time,
                    'qps': qps,
                    f'recall@{resolved_k}': recall,
                    'index_memory_mb': index_mem
                })
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append({
                    'algorithm': 'HNSW',
                    'M': M,
                    'ef_construction': ef_construction,
                    'ef_search': ef_search,
                    'build_time_s': None,
                    'search_time_s': None,
                    'qps': None,
                    f'recall@{resolved_k}': None,
                    'index_memory_mb': None,
                    'error': str(e)
                })
    
    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    df = run_hnsw_experiment()
    print("\n" + "=" * 70)
    print("HNSW RESULTS SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))
    
    # Save to CSV
    cfg = get_config()
    output_path = cfg.output.hnsw_results
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")