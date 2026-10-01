#!/usr/bin/env python3
"""
Phase 3: HNSW Index Construction and Parameter Sweep.
Tests grid search over M, ef_construction, and ef_search parameters.
Uses subprocess for accurate memory measurement.
"""
import os
import time
import json
import subprocess
import sys
import gc
import numpy as np
import pandas as pd
from itertools import product
from data_utils import read_fvecs
from phase2_groundtruth import recall_at_k_vectorized, load_groundtruth
from config_loader import get_config


def run_hnsw_single_experiment(
    base_path: str,
    query_path: str,
    gt_path: str,
    M: int,
    ef_construction: int,
    ef_search: int,
    k: int
) -> dict:
    """
    Run a single HNSW experiment in a subprocess for accurate memory measurement.
    
    Returns:
        Dictionary with build_time_s, search_time_s, qps, recall, peak_memory_mb, index_memory_mb
    """
    # Create a temporary script that runs the experiment and outputs JSON
    script = f'''
import os
import time
import json
import gc
import sys
import numpy as np
import faiss
import resource
import platform
from data_utils import read_fvecs
from phase2_groundtruth import recall_at_k_vectorized, load_groundtruth
from config_loader import get_config

def get_peak_memory_mb():
    """Get peak memory usage in MB using resource module (ru_maxrss is peak RSS)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    ru_maxrss = usage.ru_maxrss
    # On Linux, ru_maxrss is in KB; on macOS, it's in bytes
    if platform.system() == "Darwin":
        return ru_maxrss / (1024 * 1024)
    else:
        return ru_maxrss / 1024

def get_index_memory_mb(index):
    """Get index memory in MB by serializing to bytes."""
    # Serialize index to bytes using faiss.serialize_index
    serialized = faiss.serialize_index(index)
    return serialized.nbytes / (1024 * 1024)

# Force garbage collection before measuring
gc.collect()

# Load config for data_fraction
cfg = get_config()
data_fraction = cfg.data.data_fraction

# Load data
base_vectors = read_fvecs("{base_path}", fraction=data_fraction)
query_vectors = read_fvecs("{query_path}", fraction=data_fraction)
groundtruth = load_groundtruth("{gt_path}")

d = base_vectors.shape[1]
n_queries = query_vectors.shape[0]

# Validate query count
n_gt_queries = groundtruth.shape[0]
if n_gt_queries != n_queries:
    query_vectors = query_vectors[:n_gt_queries]
    n_queries = n_gt_queries

# Build index
start_build = time.time()

index = faiss.IndexHNSWFlat(d, {M})
index.hnsw.efConstruction = {ef_construction}
index.add(base_vectors)

build_time = time.time() - start_build

# Get peak memory after build phase
build_peak_mem = get_peak_memory_mb()

# Get index memory by serialization
index_mem = get_index_memory_mb(index)

# Search
index.hnsw.efSearch = {ef_search}
start_search = time.time()

distances, labels = index.search(query_vectors, {k})

search_time = time.time() - start_search

# Get peak memory after search phase (overall peak)
overall_peak_mem = get_peak_memory_mb()

qps = n_queries / search_time
recall = recall_at_k_vectorized(groundtruth, labels, {k})

result = {{
    "build_time_s": build_time,
    "search_time_s": search_time,
    "qps": qps,
    "recall": recall,
    "peak_memory_mb": overall_peak_mem,
    "build_peak_memory_mb": build_peak_mem,
    "index_memory_mb": index_mem
}}

print(json.dumps(result))
'''
    
    # Run in subprocess
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    if result.returncode != 0:
        print(f"  ERROR: Subprocess failed with return code {result.returncode}")
        print(f"  stderr: {result.stderr}")
        raise RuntimeError(f"Subprocess failed: {result.stderr}")
    
    # Parse JSON output (last line)
    output_lines = result.stdout.strip().split('\n')
    json_line = output_lines[-1]
    return json.loads(json_line)


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
    
    # Load data once to validate
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
    
    # Run grid search
    results = []
    total_combinations = len(resolved_M_values) * len(resolved_ef_construction_values) * len(resolved_ef_search_values)
    combo_idx = 0
    
    for M, ef_construction in product(resolved_M_values, resolved_ef_construction_values):
        combo_idx += 1
        print(f"\n{'=' * 70}")
        print(f"[{combo_idx}/{len(resolved_M_values) * len(resolved_ef_construction_values)}] Building IndexHNSWFlat (M={M}, ef_construction={ef_construction})")
        print(f"{'=' * 70}")
        
        # Sweep ef_search for this index
        for ef_search in resolved_ef_search_values:
            print(f"\n  [ef_search={ef_search}] Running experiment...")
            try:
                exp_result = run_hnsw_single_experiment(
                    resolved_base_path,
                    resolved_query_path,
                    resolved_gt_path,
                    M,
                    ef_construction,
                    ef_search,
                    resolved_k
                )
                
                print(f"    Build time: {exp_result['build_time_s']:.2f}s")
                print(f"    Search time: {exp_result['search_time_s']:.3f}s")
                print(f"    QPS: {exp_result['qps']:.2f}")
                print(f"    Recall@{resolved_k}: {exp_result['recall']:.4f} ({exp_result['recall']*100:.2f}%)")
                print(f"    Index memory: {exp_result['index_memory_mb']:.2f} MB")
                print(f"    Build peak memory: {exp_result['build_peak_memory_mb']:.2f} MB")
                print(f"    Overall peak memory: {exp_result['peak_memory_mb']:.2f} MB")
                
                results.append({
                    'algorithm': 'HNSW',
                    'M': M,
                    'ef_construction': ef_construction,
                    'ef_search': ef_search,
                    'build_time_s': exp_result['build_time_s'],
                    'search_time_s': exp_result['search_time_s'],
                    'qps': exp_result['qps'],
                    f'recall@{resolved_k}': exp_result['recall'],
                    'peak_memory_mb': exp_result['peak_memory_mb'],
                    'build_peak_memory_mb': exp_result['build_peak_memory_mb'],
                    'index_memory_mb': exp_result['index_memory_mb']
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
                    'peak_memory_mb': None,
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