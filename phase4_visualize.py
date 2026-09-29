#!/usr/bin/env python3
"""
Phase 4: Results Export and Visualization.
Combines HNSW and IVF-PQ results, creates comparison plots.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from config_loader import get_config


def load_and_combine_results(
    hnsw_path: str | None = None,
    ivfpq_path: str | None = None
) -> pd.DataFrame:
    """Load and combine results from both algorithms."""
    cfg = get_config()
    
    resolved_hnsw_path = hnsw_path or cfg.output.hnsw_results
    resolved_ivfpq_path = ivfpq_path or cfg.output.ivfpq_results
    
    dfs = []
    
    if os.path.exists(resolved_hnsw_path):
        df_hnsw = pd.read_csv(resolved_hnsw_path)
        dfs.append(df_hnsw)
        print(f"Loaded HNSW results: {len(df_hnsw)} rows")
    
    if os.path.exists(resolved_ivfpq_path):
        df_ivfpq = pd.read_csv(resolved_ivfpq_path)
        dfs.append(df_ivfpq)
        print(f"Loaded IVF-PQ results: {len(df_ivfpq)} rows")
    
    if not dfs:
        raise FileNotFoundError("No result files found. Run Phase 3 first.")
    
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def print_results_table(df: pd.DataFrame) -> None:
    """Print formatted results table."""
    print("\n" + "=" * 100)
    print("COMBINED RESULTS")
    print("=" * 100)
    
    # Select key columns for display
    display_cols = ['algorithm']
    
    # Add algorithm-specific params
    for col in ['M', 'ef_construction', 'ef_search', 'nlist', 'm', 'nbits', 'nprobe']:
        if col in df.columns:
            display_cols.append(col)
    
    display_cols += ['build_time_s', 'search_time_s', 'qps', 'recall@10', 'peak_memory_mb', 'index_memory_mb']
    display_cols = [c for c in display_cols if c in df.columns]
    
    print(df[display_cols].to_string(index=False, float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x)))


def plot_recall_vs_qps(df: pd.DataFrame, output_path: str | None = None) -> None:
    """Plot Recall@10 vs QPS for both algorithms."""
    cfg = get_config()
    output_path = output_path or cfg.output.plot_recall_qps
    
    plt.figure(figsize=(10, 7))
    
    # HNSW
    hnsw = df[df['algorithm'] == 'HNSW']
    if len(hnsw) > 0:
        # Get HNSW params for label
        M = hnsw['M'].iloc[0] if 'M' in hnsw.columns else '?'
        ef_c = hnsw['ef_construction'].iloc[0] if 'ef_construction' in hnsw.columns else '?'
        plt.plot(hnsw['recall@10'] * 100, hnsw['qps'], 'o-', 
                 label=f'HNSW (M={M}, ef_construction={ef_c})', 
                 linewidth=2, markersize=8, color='#2E86AB')
        # Annotate ef_search values
        for _, row in hnsw.iterrows():
            plt.annotate(f"ef={int(row['ef_search'])}", 
                        (row['recall@10'] * 100, row['qps']),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
    
    # IVF-PQ
    ivfpq = df[df['algorithm'] == 'IVF-PQ']
    if len(ivfpq) > 0:
        nlist = ivfpq['nlist'].iloc[0] if 'nlist' in ivfpq.columns else '?'
        m = ivfpq['m'].iloc[0] if 'm' in ivfpq.columns else '?'
        nbits = ivfpq['nbits'].iloc[0] if 'nbits' in ivfpq.columns else '?'
        plt.plot(ivfpq['recall@10'] * 100, ivfpq['qps'], 's-', 
                 label=f'IVF-PQ (nlist={nlist}, m={m}, nbits={nbits})', 
                 linewidth=2, markersize=8, color='#A23B72')
        # Annotate nprobe values
        for _, row in ivfpq.iterrows():
            plt.annotate(f"nprobe={int(row['nprobe'])}", 
                        (row['recall@10'] * 100, row['qps']),
                        textcoords="offset points", xytext=(5, -15), fontsize=9)
    
    plt.xlabel('Recall@10 (%)', fontsize=12)
    plt.ylabel('QPS (queries/second)', fontsize=12)
    plt.title('HNSW vs IVF-PQ: Recall@10 vs QPS Trade-off (SIFT1M)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 105)
    plt.ylim(bottom=0)
    
    # Add Pareto frontier annotation
    plt.text(0.02, 0.98, 'Higher is better →', transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=cfg.visualization.dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_memory_comparison(df: pd.DataFrame, flat_mem_mb: float | None = None, output_path: str | None = None) -> None:
    """Plot memory usage comparison."""
    cfg = get_config()
    output_path = output_path or cfg.output.plot_memory
    flat_mem_mb = flat_mem_mb or cfg.visualization.flat_memory_mb
    
    plt.figure(figsize=(10, 6))
    
    algorithms = []
    memories = []
    colors = []
    
    # Flat index memory (if provided)
    if flat_mem_mb is not None:
        algorithms.append('Flat\n(Exact)')
        memories.append(flat_mem_mb)
        colors.append('#E8E8E8')
    
    # HNSW
    hnsw = df[df['algorithm'] == 'HNSW']
    if len(hnsw) > 0:
        # Use average index memory across ef_search values
        hnsw_mem = hnsw['index_memory_mb'].mean()
        M = hnsw['M'].iloc[0] if 'M' in hnsw.columns else '?'
        algorithms.append(f'HNSW\n(M={M})')
        memories.append(hnsw_mem)
        colors.append('#2E86AB')
    
    # IVF-PQ
    ivfpq = df[df['algorithm'] == 'IVF-PQ']
    if len(ivfpq) > 0:
        ivfpq_mem = ivfpq['index_memory_mb'].mean()
        nlist = ivfpq['nlist'].iloc[0] if 'nlist' in ivfpq.columns else '?'
        m = ivfpq['m'].iloc[0] if 'm' in ivfpq.columns else '?'
        algorithms.append(f'IVF-PQ\n(nlist={nlist}, m={m})')
        memories.append(ivfpq_mem)
        colors.append('#A23B72')
    
    bars = plt.bar(algorithms, memories, color=colors, edgecolor='black', linewidth=1.2, width=0.6)
    
    # Add value labels on bars
    for bar, mem in zip(bars, memories):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(memories)*0.01,
                f'{mem:.0f} MB', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylabel('Memory Usage (MB)', fontsize=12)
    plt.title('Index Memory Comparison: Flat vs HNSW vs IVF-PQ (SIFT1M)', fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add savings annotation if we have all three
    if flat_mem_mb and len(algorithms) == 3:
        hnsw_saving = (1 - memories[1]/flat_mem_mb) * 100
        ivfpq_saving = (1 - memories[2]/flat_mem_mb) * 100
        plt.text(0.5, 0.95, f'HNSW saves {hnsw_saving:.0f}% vs Flat\nIVF-PQ saves {ivfpq_saving:.0f}% vs Flat',
                transform=plt.gca().transAxes, ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=cfg.visualization.dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_build_time_comparison(df: pd.DataFrame, output_path: str | None = None) -> None:
    """Plot build time comparison."""
    cfg = get_config()
    output_path = output_path or cfg.output.plot_build_time
    
    plt.figure(figsize=(10, 6))
    
    algorithms = []
    times = []
    colors = []
    
    hnsw = df[df['algorithm'] == 'HNSW']
    if len(hnsw) > 0:
        algorithms.append('HNSW')
        times.append(hnsw['build_time_s'].iloc[0])
        colors.append('#2E86AB')
    
    ivfpq = df[df['algorithm'] == 'IVF-PQ']
    if len(ivfpq) > 0:
        algorithms.append('IVF-PQ')
        times.append(ivfpq['build_time_s'].iloc[0])
        colors.append('#A23B72')
    
    bars = plt.bar(algorithms, times, color=colors, edgecolor='black', linewidth=1.2, width=0.5)
    
    for bar, t in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                f'{t:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylabel('Build Time (seconds)', fontsize=12)
    plt.title('Index Build Time Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=cfg.visualization.dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def export_combined_csv(df: pd.DataFrame, output_path: str | None = None) -> None:
    """Export combined results to CSV."""
    cfg = get_config()
    output_path = output_path or cfg.output.combined_results
    df.to_csv(output_path, index=False)
    print(f"Saved combined results to: {output_path}")


def main():
    print("=" * 60)
    print("Phase 4: Results Export & Visualization")
    print("=" * 60)
    
    # Load results
    df = load_and_combine_results()
    
    # Print table
    print_results_table(df)
    
    # Export combined CSV
    export_combined_csv(df)
    
    # Create plots
    print("\nGenerating plots...")
    plot_recall_vs_qps(df)
    plot_memory_comparison(df)
    plot_build_time_comparison(df)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    cfg = get_config()
    print("Generated files:")
    print(f"  - {cfg.output.combined_results}")
    print(f"  - {cfg.output.plot_recall_qps}")
    print(f"  - {cfg.output.plot_memory}")
    print(f"  - {cfg.output.plot_build_time}")


if __name__ == "__main__":
    main()