#!/usr/bin/env python3
"""
Phase 4: Results Export and Visualization.
Creates 3 HNSW results plots based on combined results, following the style of _sample.plotting.py
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from config_loader import get_config


# ============================================================
# Global style (matching _sample.plotting.py)
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.linewidth": 0.6,
})


def load_results(
    hnsw_path: str | None = None
) -> pd.DataFrame:
    """Load HNSW results."""
    cfg = get_config()
    
    resolved_hnsw_path = hnsw_path or cfg.output.hnsw_results
    
    if not os.path.exists(resolved_hnsw_path):
        raise FileNotFoundError(f"No result file found: {resolved_hnsw_path}. Run Phase 3 first.")
    
    df_hnsw = pd.read_csv(resolved_hnsw_path)
    print(f"Loaded HNSW results: {len(df_hnsw)} rows")
    return df_hnsw


def print_results_table(df: pd.DataFrame) -> None:
    """Print formatted results table."""
    print("\n" + "=" * 100)
    print("HNSW RESULTS")
    print("=" * 100)
    
    display_cols = ['algorithm']
    
    for col in ['M', 'ef_construction', 'ef_search']:
        if col in df.columns:
            display_cols.append(col)
    
    display_cols += ['build_time_s', 'search_time_s', 'qps', 'recall@10', 'index_memory_mb']
    display_cols = [c for c in display_cols if c in df.columns]
    
    print(df[display_cols].to_string(index=False, float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x)))


# ============================================================
# Plot: Recall (x) vs QPS (y)
# Subplots per M, color by M, marker by ef_construction
# ============================================================
def plot_recall_vs_qps(df: pd.DataFrame, output_path: str | None = None) -> None:
    cfg = get_config()
    output_path = output_path or cfg.output.plot_recall_vs_qps
    
    hnsw = df[df['algorithm'] == 'HNSW']
    
    m_values = sorted(hnsw["M"].unique())
    efc_values = sorted(hnsw["ef_construction"].unique())
    
    fig, axes = plt.subplots(
        len(m_values),
        1,
        figsize=(10, 3.5 * len(m_values)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()
    
    # Markers for ef_construction
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    marker_map = {efc: markers[i % len(markers)] for i, efc in enumerate(efc_values)}
    
    # Colors for M
    colors = plt.get_cmap("Set2")(np.linspace(0, 1, len(m_values)))
    color_map = {m: c for m, c in zip(m_values, colors)}
    
    for ax, M in zip(axes, m_values):
        for efc in efc_values:
            subset = hnsw[(hnsw["M"] == M) & (hnsw["ef_construction"] == efc)]
            if subset.empty:
                continue
            subset = subset.sort_values("recall@10")
            ax.plot(
                subset["recall@10"],
                subset["qps"],
                marker=marker_map[efc],
                markersize=8,
                linewidth=1.5,
                alpha=0.85,
                color=color_map[M],
            )
        ax.set_ylabel(f"QPS\nM={M}")
        ax.grid(True, alpha=0.3)
    
    # Custom legend for ef_construction markers
    legend_efc = [
        Line2D([0], [0], color="gray", marker=marker_map[efc], linestyle="None", markersize=8, label=f"ef_c={efc}")
        for efc in efc_values
    ]
    
    axes[0].legend(handles=legend_efc, title="ef_construction (marker)", loc="upper right", frameon=True)
    axes[0].set_title("HNSW: QPS vs Recall@10")
    axes[-1].set_xlabel("Recall@10")
    axes[-1].set_xlim(0.83, 1.01)
    fig.tight_layout()
    
    plt.savefig(output_path, dpi=cfg.visualization.dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# Plot: 3D bar plot — sort by distance to viewpoint
# ============================================================
def plot_memory_3d_bar(df: pd.DataFrame, output_path: str | None = None) -> None:
    cfg = get_config()
    output_path = output_path or cfg.output.plot_memory_3d_bar
    
    hnsw = df[df['algorithm'] == 'HNSW']
    
    fig = plt.figure(figsize=(9.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    
    m_values = sorted(hnsw["M"].unique())
    colors = plt.get_cmap("Set2")(np.linspace(0, 1, len(m_values)))
    color_map_3d = {m: c for m, c in zip(m_values, colors)}
    
    efc_values = sorted(hnsw["ef_construction"].unique())
    efs_values = sorted(hnsw["ef_search"].unique())
    m_count = len(m_values)
    
    bar_width = 0.55 / m_count
    bar_depth = 0.55
    
    # ---------- visible z range ----------
    zmin, zmax = 0.60, 1.00
    
    # ---------- camera ----------
    elev = 22
    azim = 215 + 90
    ax.view_init(elev=elev, azim=azim)
    
    # approximate eye position in data coordinates
    R = 12.0   # camera distance
    elev_r = np.deg2rad(elev)
    azim_r = np.deg2rad(azim)
    
    eye = np.array([
        R * np.cos(elev_r) * np.cos(azim_r),
        R * np.cos(elev_r) * np.sin(azim_r),
        R * np.sin(elev_r)
    ])
    
    # ---------- collect bars ----------
    bars = []   # (x, y, z, dx, dy, dz, color, dist)
    
    for m_index, M in enumerate(m_values):
        group = hnsw[hnsw["M"] == M]
        x_idx = group["ef_construction"].map(
            {v: i for i, v in enumerate(efc_values)}
        ).to_numpy()
        y_idx = group["ef_search"].map(
            {v: i for i, v in enumerate(efs_values)}
        ).to_numpy()
        
        x = x_idx - 0.27 + m_index * bar_width
        y = y_idx - bar_depth / 2
        z = np.full(len(group), zmin)
        dz = group["recall@10"].to_numpy() - zmin
        
        for i in range(len(group)):
            # center of the bar (for distance)
            cx = x[i] + bar_width / 2
            cy = y[i] + bar_depth / 2
            cz = z[i] + dz[i] / 2
            
            # Euclidean distance to eye → larger = farther
            dist = np.linalg.norm(np.array([cx, cy, cz]) - eye)
            
            bars.append((
                x[i], y[i], z[i],
                bar_width, bar_depth, dz[i],
                color_map_3d[M],
                dist
            ))
    
    # far → near
    bars.sort(key=lambda b: b[7], reverse=True)
    
    # ---------- draw ----------
    for x, y, z, dx, dy, dz, color, _ in bars:
        ax.bar3d(
            x, y, z, dx, dy, dz,
            color=color,
            alpha=0.85,
            shade=True,
            edgecolor="k",
            linewidth=0.3,
        )
    
    # ---------- cosmetics ----------
    ax.set_xlabel("ef_construction", labelpad=8)
    ax.set_ylabel("ef_search", labelpad=8)
    ax.set_zlabel("Recall@10", labelpad=6)
    ax.set_title("HNSW Recall@10 by ef_construction & ef_search", pad=12)
    
    ax.set_xticks(range(len(efc_values)), efc_values)
    ax.set_yticks(range(len(efs_values)), efs_values)
    ax.set_zlim(zmin, zmax)
    
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    
    legend_handles = [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=color_map_3d[m], markersize=9,
               label=f"M = {m}")
        for m in m_values
    ]
    ax.legend(
        handles=legend_handles,
        title="M",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.95),
        frameon=True,
        edgecolor="black",
        title_fontsize=9,
    )
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=cfg.visualization.dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# Plot 3: recall@10 vs index_memory_mb
# one row per ef_construction
# lines connect same ef_search; markers differ by M
# independent x/y axes per row
# ============================================================
def plot_recall_vs_efsearch(df: pd.DataFrame, output_path: str | None = None) -> None:
    cfg = get_config()
    output_path = output_path or cfg.output.plot_recall_vs_efsearch
    
    hnsw = df[df['algorithm'] == 'HNSW']
    
    efc_values = sorted(hnsw["ef_construction"].unique())
    m_values   = sorted(hnsw["M"].unique())
    efs_values = sorted(hnsw["ef_search"].unique())
    
    # markers & colors
    markers = ["o", "s", "D", "^", "v", "P"]
    marker_map = {m: markers[i % len(markers)] for i, m in enumerate(m_values)}
    
    # soft color cycle for the ef_search lines
    efs_colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, len(efs_values)))
    efs_color_map = {efs: efs_colors[i] for i, efs in enumerate(efs_values)}
    
    n_rows = len(efc_values)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(8, 2.8 * n_rows),
        sharex=False,
        sharey=False,
    )
    
    if n_rows == 1:
        axes = [axes]
    
    for ax, efc in zip(axes, efc_values):
        sub = hnsw[hnsw["ef_construction"] == efc].copy()
        
        # --- lines: same ef_search ---
        for efs in efs_values:
            g = sub[sub["ef_search"] == efs].sort_values("index_memory_mb")
            if len(g) > 1:
                ax.plot(
                    g["index_memory_mb"],
                    g["recall@10"],
                    color=efs_color_map[efs],
                    linewidth=1.2,
                    alpha=0.75,
                    zorder=1,
                )
        
        # --- markers: different M ---
        for M in m_values:
            g = sub[sub["M"] == M]
            ax.scatter(
                g["index_memory_mb"],
                g["recall@10"],
                marker=marker_map[M],
                s=45,
                c=[efs_color_map[efs] for efs in g["ef_search"]],
                edgecolors="black",
                linewidths=0.6,
                zorder=3,
            )
        
        ax.set_ylabel("Recall@10")
        ax.set_title(f"ef_construction = {efc}", loc="left", fontsize=11)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        ax.set_axisbelow(True)
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
    
    # only bottom row gets x-label
    axes[-1].set_xlabel("Index memory (MB)")
    
    # M markers legend
    m_handles = [
        Line2D(
            [0], [0],
            marker=marker_map[m],
            color="none",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=7,
            label=f"M={m}",
        )
        for m in m_values
    ]
    marker_legend = axes[0].legend(
        handles=m_handles,
        title="M",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        edgecolor="black",
        fontsize=8,
    )
    axes[0].add_artist(marker_legend)
    
    # ef_search lines legend
    efs_handles = [
        Line2D([0], [0], color=efs_color_map[efs], lw=1.5, label=f"ef_search={efs}")
        for efs in efs_values
    ]
    axes[0].legend(
        handles=efs_handles,
        title="ef_search",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),
        frameon=True,
        edgecolor="black",
        fontsize=8,
    )
    
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
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
    print("Phase 4: Results Export & Visualization (3 Plots)")
    print("=" * 60)
    
    # Load results
    df = load_results()
    
    # Print table
    print_results_table(df)
    
    # Export combined CSV
    export_combined_csv(df)
    
    # Create 3 plots
    print("\nGenerating 3 plots...")
    plot_recall_vs_qps(df)
    plot_memory_3d_bar(df)
    plot_recall_vs_efsearch(df)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE - 3 Plots Generated")
    print("=" * 60)
    cfg = get_config()
    print("Generated files:")
    print(f"  - {cfg.output.combined_results}")
    print(f"  - {cfg.output.plot_recall_vs_qps}")
    print(f"  - {cfg.output.plot_memory_3d_bar}")
    print(f"  - {cfg.output.plot_recall_vs_efsearch}")


if __name__ == "__main__":
    main()