#!/usr/bin/env python3
"""
Main runner script for HNSW benchmark on SIFT1M.
Orchestrates all phases based on config.yaml - no command line args needed.
"""
import os
import sys
import subprocess
from config_loader import get_config


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    cfg = get_config()
    
    print("=" * 60)
    print("HNSW Benchmark on SIFT1M")
    print("=" * 60)
    print("Execution plan from config.yaml:")
    print(f"  Phase 1 (Download):     {cfg.run.phase1_download}")
    print(f"  Phase 2 (Ground Truth): {cfg.run.phase2_groundtruth}")
    print(f"  Phase 3 (HNSW):         {cfg.run.phase3_hnsw}")
    print(f"  Phase 4 (Visualize):    {cfg.run.phase4_visualize}")
    
    success = True
    
    # Phase 1: Environment & Data
    if cfg.run.phase1_download:
        success &= run_command([sys.executable, "download_data.py"], "Phase 1: Download SIFT1M dataset")
    
    # Phase 2: Ground Truth
    if cfg.run.phase2_groundtruth and success:
        success &= run_command([sys.executable, "phase2_groundtruth.py"], "Phase 2: Build Flat index & compute Ground Truth")
    
    # Phase 3: HNSW
    if cfg.run.phase3_hnsw and success:
        success &= run_command([sys.executable, "phase3_hnsw.py"], "Phase 3: HNSW parameter sweep")
    
    # Phase 4: Visualization
    if cfg.run.phase4_visualize and success:
        success &= run_command([sys.executable, "phase4_visualize.py"], "Phase 4: Results visualization")
    
    if success:
        print("\n" + "=" * 60)
        print("ALL PHASES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Output files:")
        for f in [cfg.output.hnsw_results, cfg.output.combined_results,
                  cfg.output.plot_recall_vs_qps, cfg.output.plot_recall_vs_efsearch,
                  cfg.output.plot_memory_3d_bar]:
            if os.path.exists(f):
                print(f"  ✓ {f}")
    else:
        print("\n" + "=" * 60)
        print("SOME PHASES FAILED - Check output above")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()