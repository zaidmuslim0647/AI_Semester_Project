"""Visualization for AeroNet Lite.

Owner: Zaid (zone map). Saad will extend with heatmap and anomaly views.
"""

from __future__ import annotations

import os
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from grid_model import Cell, GRID_SIZE, ZONES, make_sample_grid


ZONE_COLORS = {
    "Residential": "#FFE082",   # warm yellow
    "Commercial":  "#90CAF9",   # blue
    "Hospital":    "#EF9A9A",   # red-pink
    "School":      "#CE93D8",   # purple
    "Industrial":  "#A1887F",   # brown
    "OpenField":   "#C5E1A5",   # green
}

ZONE_TO_INDEX = {z: i for i, z in enumerate(ZONES)}


def _zone_image(grid: List[List[Cell]]) -> np.ndarray:
    n = len(grid)
    img = np.zeros((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            img[r, c] = ZONE_TO_INDEX[grid[r][c].zone]
    return img


def plot_zone_map(
    grid: List[List[Cell]],
    title: str = "AeroNet Lite — Zone Map",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    n = len(grid)
    cmap = ListedColormap([ZONE_COLORS[z] for z in ZONES])
    img = _zone_image(grid)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap=cmap, vmin=0, vmax=len(ZONES) - 1)

    # Grid lines
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)

    # Major ticks = cell coordinates
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_title(title)

    # Overlay markers
    hub_rc, charge_rc, med_rc, nofly_rc = [], [], [], []
    for r in range(n):
        for c in range(n):
            cell = grid[r][c]
            if cell.is_hub:        hub_rc.append((r, c))
            if cell.is_charging:   charge_rc.append((r, c))
            if cell.is_medical_pickup: med_rc.append((r, c))
            if cell.no_fly:        nofly_rc.append((r, c))

    def scatter(pts, marker, color, size, label, edge="black"):
        if not pts:
            return None
        ys, xs = zip(*pts)
        return ax.scatter(xs, ys, marker=marker, c=color, s=size,
                          edgecolors=edge, linewidths=1.2, label=label, zorder=3)

    scatter(hub_rc,    "*", "black",     220, "Hub")
    scatter(charge_rc, "^", "yellow",    140, "Charging")
    scatter(med_rc,    "P", "red",       140, "Medical Pickup")
    scatter(nofly_rc,  "X", "darkred",   180, "No-Fly")

    # Legend: zone patches + marker entries
    zone_patches = [Patch(facecolor=ZONE_COLORS[z], edgecolor="gray", label=z) for z in ZONES]
    marker_handles = [
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
                   markeredgecolor="black", markersize=14, label="Hub"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="yellow",
                   markeredgecolor="black", markersize=11, label="Charging"),
        plt.Line2D([0], [0], marker="P", color="w", markerfacecolor="red",
                   markeredgecolor="black", markersize=11, label="Medical Pickup"),
        plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="darkred",
                   markeredgecolor="black", markersize=12, label="No-Fly"),
    ]
    ax.legend(
        handles=zone_patches + marker_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
    )

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_demand_heatmap(
    grid: List[List[Cell]],
    title: str = "Predicted Demand Heatmap",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot demand forecast as a heatmap overlay on the grid.
    
    Owner: Saad
    
    Args:
        grid: 10x10 grid with demand values in cell.demand.
        title: Figure title.
        save_path: Optional path to save the figure.
        show: Whether to display the figure.
        
    Returns:
        matplotlib Figure object.
    """
    n = len(grid)
    demand_array = np.zeros((n, n), dtype=float)
    
    for r in range(n):
        for c in range(n):
            demand_array[r, c] = grid[r][c].demand
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot heatmap
    im = ax.imshow(demand_array, cmap="YlOrRd", interpolation="nearest")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Predicted Demand", rotation=270, labelpad=15)
    
    # Grid lines
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    
    # Major ticks
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_title(title)
    
    # Add text annotations with demand values
    for r in range(n):
        for c in range(n):
            val = demand_array[r, c]
            if val > 0:
                ax.text(c, r, f"{val:.1f}", ha="center", va="center",
                       color="black" if val < (demand_array.max() * 0.7) else "white",
                       fontsize=8)
    
    fig.tight_layout()
    
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    
    return fig


def plot_anomaly_timeline(
    anomalies: List[dict],
    title: str = "Anomaly Detection Timeline",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot anomalies detected during simulation as a timeline.
    
    Owner: Saad
    
    Args:
        anomalies: List of dicts with keys:
                   - drone_id (str)
                   - step (int)
                   - label (str): "Normal" | "Battery" | "Route" | "Sensor"
                   - confidence (float, optional)
        title: Figure title.
        save_path: Optional path to save the figure.
        show: Whether to display the figure.
        
    Returns:
        matplotlib Figure object.
    """
    # Filter out normal anomalies for cleaner visualization
    anomalies = [a for a in anomalies if a.get("label", "Normal") != "Normal"]
    
    if not anomalies:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No anomalies detected", ha="center", va="center",
               transform=ax.transAxes, fontsize=14)
        ax.axis("off")
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig
    
    # Extract data
    steps = [a["step"] for a in anomalies]
    drone_ids = [a["drone_id"] for a in anomalies]
    labels = [a["label"] for a in anomalies]
    
    # Color mapping for anomaly types
    anomaly_colors = {
        "Battery": "red",
        "Route": "orange",
        "Sensor": "purple",
        "Normal": "green",
    }
    colors = [anomaly_colors.get(label, "gray") for label in labels]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Scatter plot: x=step, y=drone_id, colored by anomaly type
    unique_drones = sorted(set(drone_ids))
    drone_to_y = {d: i for i, d in enumerate(unique_drones)}
    y_coords = [drone_to_y[d] for d in drone_ids]
    
    scatter = ax.scatter(steps, y_coords, c=colors, s=200, alpha=0.7,
                        edgecolors="black", linewidth=1.5)
    
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Drone ID")
    ax.set_yticks(range(len(unique_drones)))
    ax.set_yticklabels(unique_drones)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Legend for anomaly types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="red", edgecolor="black", label="Battery"),
        Patch(facecolor="orange", edgecolor="black", label="Route"),
        Patch(facecolor="purple", edgecolor="black", label="Sensor"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    
    return fig


def print_anomaly_log(
    anomalies: List[dict],
    title: str = "Anomaly Detection Log",
) -> None:
    """Print anomalies as a formatted log.
    
    Owner: Saad
    
    Args:
        anomalies: List of dicts with keys:
                   - drone_id (str)
                   - step (int)
                   - label (str)
                   - confidence (float, optional)
        title: Log title.
    """
    print(f"\n{'=' * 70}")
    print(f"{title:^70}")
    print(f"{'=' * 70}")
    
    # Filter anomalies
    anomalies = sorted(anomalies, key=lambda a: (a["step"], a["drone_id"]))
    
    if not anomalies or all(a.get("label") == "Normal" for a in anomalies):
        print("No anomalies detected during simulation.")
    else:
        print(f"{'Step':<8} {'Drone ID':<12} {'Label':<12} {'Confidence':<12}")
        print("-" * 70)
        for a in anomalies:
            if a.get("label", "Normal") != "Normal":
                confidence = a.get("confidence", 0.0)
                print(f"{a['step']:<8} {a['drone_id']:<12} {a['label']:<12} {confidence:>10.2%}")
    
    print(f"{'=' * 70}\n")



if __name__ == "__main__":
    g = make_sample_grid()
    out = os.path.join(os.path.dirname(__file__), "..", "report", "figures", "zone_map.png")
    plot_zone_map(g, save_path=os.path.abspath(out), show=False)
    print(f"Saved zone map to {os.path.abspath(out)}")
