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


if __name__ == "__main__":
    g = make_sample_grid()
    out = os.path.join(os.path.dirname(__file__), "..", "report", "figures", "zone_map.png")
    plot_zone_map(g, save_path=os.path.abspath(out), show=False)
    print(f"Saved zone map to {os.path.abspath(out)}")
