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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection
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


def plot_3d_recall_qps_memory(df: pd.DataFrame, output_path: str | None = None) -> None:
    """3D plot: Recall@10 vs QPS vs Peak Memory for both algorithms.
    
    This shows the full trade-off surface: higher recall typically requires
    more memory and/or lower QPS. The parameter (ef_search for HNSW, nprobe
    for IVF-PQ) moves you along each algorithm's curve.
    """
    cfg = get_config()
    output_path = output_path or cfg.output.plot_3d_recall_qps_memory
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # HNSW
    hnsw = df[df['algorithm'] == 'HNSW']
    if len(hnsw) > 0:
        M = hnsw['M'].iloc[0] if 'M' in hnsw.columns else '?'
        ef_c = hnsw['ef_construction'].iloc[0] if 'ef_construction' in hnsw.columns else '?'
        
        x = hnsw['recall@10'] * 100
        y = hnsw['qps']
        z = hnsw['peak_memory_mb']
        
        # Plot line connecting points in order of ef_search
        hnsw_sorted = hnsw.sort_values('ef_search')
        ax.plot(hnsw_sorted['recall@10'] * 100, hnsw_sorted['qps'], hnsw_sorted['peak_memory_mb'],
                'o-', label=f'HNSW (M={M}, ef_construction={ef_c})',
                linewidth=2, markersize=8, color='#2E86AB')
        
        # Annotate ef_search values
        for _, row in hnsw_sorted.iterrows():
            ax.text(row['recall@10'] * 100, row['qps'], row['peak_memory_mb'],
                    f"ef={int(row['ef_search'])}", fontsize=9, color='#2E86AB')
    
    # IVF-PQ
    ivfpq = df[df['algorithm'] == 'IVF-PQ']
    if len(ivfpq) > 0:
        nlist = ivfpq['nlist'].iloc[0] if 'nlist' in ivfpq.columns else '?'
        m = ivfpq['m'].iloc[0] if 'm' in ivfpq.columns else '?'
        nbits = ivfpq['nbits'].iloc[0] if 'nbits' in ivfpq.columns else '?'
        
        ivfpq_sorted = ivfpq.sort_values('nprobe')
        ax.plot(ivfpq_sorted['recall@10'] * 100, ivfpq_sorted['qps'], ivfpq_sorted['peak_memory_mb'],
                's-', label=f'IVF-PQ (nlist={nlist}, m={m}, nbits={nbits})',
                linewidth=2, markersize=8, color='#A23B72')
        
        # Annotate nprobe values
        for _, row in ivfpq_sorted.iterrows():
            ax.text(row['recall@10'] * 100, row['qps'], row['peak_memory_mb'],
                    f"nprobe={int(row['nprobe'])}", fontsize=9, color='#A23B72')
    
    ax.set_xlabel('Recall@10 (%)', fontsize=11, labelpad=10)
    ax.set_ylabel('QPS (queries/sec)', fontsize=11, labelpad=10)
    ax.set_zlabel('Peak Memory (MB)', fontsize=11, labelpad=10)
    ax.set_title('3D Trade-off: Recall@10 vs QPS vs Memory\n(HNSW: ef_search, IVF-PQ: nprobe)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='upper left')
    
    # Set axis limits for better visualization
    ax.set_xlim(0, 105)
    ax.set_ylim(bottom=0)
    ax.set_zlim(bottom=0)
    
    # Add direction annotations
    ax.text2D(0.02, 0.98, '← Higher Recall\n← Higher QPS\n← Lower Memory',
              transform=ax.transAxes, fontsize=9, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=cfg.visualization.dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_3d_param_effects(df: pd.DataFrame, output_path: str | None = None) -> None:
    """3D plot: Parameter (ef_search/nprobe) vs Recall@10 vs QPS.
    
    Shows how the search parameter directly affects the recall-QPS trade-off.
    Memory is shown as color/size for additional dimension.
    """
    cfg = get_config()
    output_path = output_path or cfg.output.plot_3d_param_effects
    
    fig = plt.figure(figsize=(14, 10))
    
    # Subplot 1: HNSW - ef_search vs Recall vs QPS
    ax1 = fig.add_subplot(121, projection='3d')
    hnsw = df[df['algorithm'] == 'HNSW']
    if len(hnsw) > 0:
        hnsw_sorted = hnsw.sort_values('ef_search')
        x = hnsw_sorted['ef_search']
        y = hnsw_sorted['recall@10'] * 100
        z = hnsw_sorted['qps']
        c = hnsw_sorted['peak_memory_mb']
        
        scatter = ax1.scatter(x, y, z, c=c, cmap='Blues', s=100, edgecolor='black', linewidth=1, depthshade=True)
        ax1.plot(x, y, z, 'o-', color='#2E86AB', linewidth=2, alpha=0.7)
        
        for _, row in hnsw_sorted.iterrows():
            ax1.text(row['ef_search'], row['recall@10'] * 100, row['qps'],
                    f"ef={int(row['ef_search'])}", fontsize=8, color='#2E86AB')
        
        ax1.set_xlabel('ef_search', fontsize=11, labelpad=10)
        ax1.set_ylabel('Recall@10 (%)', fontsize=11, labelpad=10)
        ax1.set_zlabel('QPS (queries/sec)', fontsize=11, labelpad=10)
        ax1.set_title('HNSW: ef_search → Recall vs QPS\n(Color = Peak Memory)',
                      fontsize=12, fontweight='bold', pad=15)
        ax1.set_xscale('log', base=2)
        
        # Colorbar
        cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.6, pad=0.1)
        cbar1.set_label('Peak Memory (MB)', fontsize=9)
    
    # Subplot 2: IVF-PQ - nprobe vs Recall vs QPS
    ax2 = fig.add_subplot(122, projection='3d')
    ivfpq = df[df['algorithm'] == 'IVF-PQ']
    if len(ivfpq) > 0:
        ivfpq_sorted = ivfpq.sort_values('nprobe')
        x = ivfpq_sorted['nprobe']
        y = ivfpq_sorted['recall@10'] * 100
        z = ivfpq_sorted['qps']
        c = ivfpq_sorted['peak_memory_mb']
        
        scatter = ax2.scatter(x, y, z, c=c, cmap='Reds', s=100, edgecolor='black', linewidth=1, depthshade=True)
        ax2.plot(x, y, z, 's-', color='#A23B72', linewidth=2, alpha=0.7)
        
        for _, row in ivfpq_sorted.iterrows():
            ax2.text(row['nprobe'], row['recall@10'] * 100, row['qps'],
                    f"nprobe={int(row['nprobe'])}", fontsize=8, color='#A23B72')
        
        ax2.set_xlabel('nprobe', fontsize=11, labelpad=10)
        ax2.set_ylabel('Recall@10 (%)', fontsize=11, labelpad=10)
        ax2.set_zlabel('QPS (queries/sec)', fontsize=11, labelpad=10)
        ax2.set_title('IVF-PQ: nprobe → Recall vs QPS\n(Color = Peak Memory)',
                      fontsize=12, fontweight='bold', pad=15)
        ax2.set_xscale('log', base=2)
        
        # Colorbar
        cbar2 = plt.colorbar(scatter, ax=ax2, shrink=0.6, pad=0.1)
        cbar2.set_label('Peak Memory (MB)', fontsize=9)
    
    fig.suptitle('Parameter Sensitivity: How ef_search / nprobe Controls the Recall-QPS Trade-off',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=cfg.visualization.dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_3d_bar_recall_method_memory(df: pd.DataFrame, output_path: str | None = None) -> None:
    """3D bar plot: Method vs Memory vs Recall@10.
    
    Shows recall as bar height, with method (HNSW/IVF-PQ) and memory usage
    as the two floor axes. Each bar represents a specific parameter setting
    (ef_search for HNSW, nprobe for IVF-PQ).
    """
    cfg = get_config()
    output_path = output_path or cfg.output.plot_3d_bar_recall_method_memory
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare data: each row is a bar
    # x-axis: method (0=HNSW, 1=IVF-PQ)
    # y-axis: peak memory (MB)
    # z-axis: recall@10 (%)
    
    bars_data = []
    
    # HNSW bars
    hnsw = df[df['algorithm'] == 'HNSW']
    if len(hnsw) > 0:
        hnsw_sorted = hnsw.sort_values('ef_search')
        for _, row in hnsw_sorted.iterrows():
            bars_data.append({
                'method': 0,
                'method_name': 'HNSW',
                'memory': row['peak_memory_mb'],
                'recall': row['recall@10'] * 100,
                'param': f"ef={int(row['ef_search'])}",
                'color': '#2E86AB',
                'alpha': 0.7
            })
    
    # IVF-PQ bars
    ivfpq = df[df['algorithm'] == 'IVF-PQ']
    if len(ivfpq) > 0:
        ivfpq_sorted = ivfpq.sort_values('nprobe')
        for _, row in ivfpq_sorted.iterrows():
            bars_data.append({
                'method': 1,
                'method_name': 'IVF-PQ',
                'memory': row['peak_memory_mb'],
                'recall': row['recall@10'] * 100,
                'param': f"nprobe={int(row['nprobe'])}",
                'color': '#A23B72',
                'alpha': 0.7
            })
    
    if not bars_data:
        print("No data for 3D bar plot")
        return
    
    # Bar dimensions
    dx = 0.4  # width along method axis
    dy = 5    # width along memory axis (MB)
    
    # Plot bars
    for i, bar in enumerate(bars_data):
        x = bar['method']
        y = bar['memory']
        z = 0
        dz = bar['recall']
        
        ax.bar3d(x, y, z, dx, dy, dz,
                 color=bar['color'], alpha=bar['alpha'],
                 edgecolor='black', linewidth=0.5, shade=True)
        
        # Add parameter label on top of bar
        ax.text(x + dx/2, y + dy/2, dz + 1, bar['param'],
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                color=bar['color'], rotation=0)
        
        # Add recall value on top
        ax.text(x + dx/2, y + dy/2, dz + 3, f"{dz:.1f}%",
                ha='center', va='bottom', fontsize=7, color='black')
    
    # Set labels and ticks
    ax.set_xlabel('Method', fontsize=12, labelpad=15)
    ax.set_ylabel('Peak Memory (MB)', fontsize=12, labelpad=15)
    ax.set_zlabel('Recall@10 (%)', fontsize=12, labelpad=15)
    
    ax.set_xticks([0.2, 1.2])
    ax.set_xticklabels(['HNSW', 'IVF-PQ'], fontsize=11)
    
    ax.set_title('3D Bar Plot: Recall@10 by Method and Memory Usage\n(Bar height = Recall, Labels = ef_search / nprobe)',
                 fontsize=13, fontweight='bold', pad=20)
    
    ax.set_zlim(0, 110)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', alpha=0.7, edgecolor='black', label='HNSW (ef_search)'),
        Patch(facecolor='#A23B72', alpha=0.7, edgecolor='black', label='IVF-PQ (nprobe)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Add annotation
    ax.text2D(0.02, 0.98, 'Higher bars = Better recall\nWider bars = Memory range',
              transform=ax.transAxes, fontsize=9, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
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
    plot_3d_recall_qps_memory(df)
    plot_3d_param_effects(df)
    plot_3d_bar_recall_method_memory(df)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    cfg = get_config()
    print("Generated files:")
    print(f"  - {cfg.output.combined_results}")
    print(f"  - {cfg.output.plot_recall_qps}")
    print(f"  - {cfg.output.plot_memory}")
    print(f"  - {cfg.output.plot_build_time}")
    print(f"  - {cfg.output.plot_3d_recall_qps_memory}")
    print(f"  - {cfg.output.plot_3d_param_effects}")
    print(f"  - {cfg.output.plot_3d_bar_recall_method_memory}")


if __name__ == "__main__":
    main()