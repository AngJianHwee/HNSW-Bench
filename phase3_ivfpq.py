#!/usr/bin/env python3
"""
Phase 3: IVF-PQ Index Construction and Parameter Sweep.
Tests nprobe parameter sweep with fixed nlist=1024, m=8, nbits=8.
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


def run_ivfpq_experiment(
    base_path: str | None = None,
    query_path: str | None = None,
    learn_path: str | None = None,
    gt_path: str | None = None,
    nlist: int | None = None,
    m: int | None = None,
    nbits: int | None = None,
    nprobe_values: list | None = None,
    k: int | None = None
) -> pd.DataFrame:
    """
    Run IVF-PQ parameter sweep over nprobe values.
    
    Args:
        base_path: Path to base vectors (uses config if None)
        query_path: Path to query vectors (uses config if None)
        learn_path: Path to training vectors (uses config if None)
        gt_path: Path to ground truth (uses config if None)
        nlist: Number of inverted lists (uses config if None)
        m: Number of subquantizers (uses config if None)
        nbits: Bits per subquantizer (uses config if None)
        nprobe_values: List of nprobe values to test (uses config if None)
        k: Top-k for recall calculation (uses config if None)
        
    Returns:
        DataFrame with results for each nprobe
    """
    cfg = get_config()
    
    # Resolve config values
    resolved_base_path = base_path or cfg.data.base_path
    resolved_query_path = query_path or cfg.data.query_path
    resolved_learn_path = learn_path or cfg.data.learn_path
    resolved_gt_path = gt_path or cfg.data.gt_path
    resolved_nlist = nlist or cfg.ivfpq.nlist
    resolved_m = m or cfg.ivfpq.m
    resolved_nbits = nbits or cfg.ivfpq.nbits
    resolved_nprobe_values = nprobe_values or cfg.ivfpq.nprobe_values
    resolved_k = k or cfg.ivfpq.k
    
    print("=" * 60)
    print(f"Phase 3: IVF-PQ Index (nlist={resolved_nlist}, m={resolved_m}, nbits={resolved_nbits})")
    print("=" * 60)
    
    # Load data
    print("Loading base vectors...")
    base_vectors = read_fvecs(resolved_base_path)
    print(f"Base shape: {base_vectors.shape}")
    
    print("Loading query vectors...")
    query_vectors = read_fvecs(resolved_query_path)
    print(f"Query shape: {query_vectors.shape}")
    
    print("Loading training vectors...")
    learn_vectors = read_fvecs(resolved_learn_path)
    print(f"Learn shape: {learn_vectors.shape}")
    
    print("Loading ground truth...")
    groundtruth = load_groundtruth(resolved_gt_path)
    print(f"Ground truth shape: {groundtruth.shape}")
    
    d = base_vectors.shape[1]
    n_queries = query_vectors.shape[0]
    
    # Build IVF-PQ index
    print(f"\nBuilding IndexIVFPQ (nlist={resolved_nlist}, m={resolved_m}, nbits={resolved_nbits})...")
    mem_before = get_memory_usage_mb()
    start_total = time.time()
    
    # Create quantizer (coarse quantizer)
    quantizer = faiss.IndexFlatL2(d)
    
    # Create IVF-PQ index
    index = faiss.IndexIVFPQ(quantizer, d, resolved_nlist, resolved_m, resolved_nbits)
    
    # Train the index (learns PQ codebooks and coarse quantizer centroids)
    print("Training index...")
    start_train = time.time()
    index.train(learn_vectors)
    train_time = time.time() - start_train
    print(f"Training time: {train_time:.2f}s")
    
    # Add base vectors
    print("Adding base vectors...")
    start_add = time.time()
    index.add(base_vectors)
    add_time = time.time() - start_add
    print(f"Add time: {add_time:.2f}s")
    
    total_build_time = time.time() - start_total
    mem_after_build = get_memory_usage_mb()
    index_mem = mem_after_build - mem_before
    
    print(f"Index built: {index.ntotal} vectors")
    print(f"Total build time: {total_build_time:.2f}s (train: {train_time:.2f}s, add: {add_time:.2f}s)")
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
    
    for nprobe in resolved_nprobe_values:
        print(f"\n--- nprobe = {nprobe} ---")
        index.nprobe = nprobe
        
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
            'algorithm': 'IVF-PQ',
            'nlist': resolved_nlist,
            'm': resolved_m,
            'nbits': resolved_nbits,
            'nprobe': nprobe,
            'train_time_s': train_time,
            'add_time_s': add_time,
            'build_time_s': total_build_time,
            'search_time_s': search_time,
            'qps': qps,
            f'recall@{resolved_k}': recall,
            'peak_memory_mb': peak_mem,
            'index_memory_mb': index_mem
        })
    
    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    df = run_ivfpq_experiment()
    print("\n" + "=" * 60)
    print("IVF-PQ RESULTS SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    
    # Save to CSV
    cfg = get_config()
    output_path = cfg.output.ivfpq_results
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")