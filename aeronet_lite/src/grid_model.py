"""Shared grid model for AeroNet Lite.

This module is the Day-1 contract every other module reads from.
Owner: Zaid. Do not change field names without team agreement.
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


Coord = Tuple[int, int]

ZONES = ("Residential", "Commercial", "Hospital", "School", "Industrial", "OpenField")

ZONE_CODE = {
    "Residential": "R",
    "Commercial":  "C",
    "Hospital":    "H",
    "School":      "S",
    "Industrial":  "I",
    "OpenField":   "O",
}


@dataclass
class Cell:
    row: int
    col: int
    zone: str
    density: int = 0
    is_hub: bool = False
    is_charging: bool = False
    is_medical_pickup: bool = False
    no_fly: bool = False
    demand: float = 0.0
    cost: float = 1.0


@dataclass
class Drone:
    id: str
    type: str                       # "light" | "heavy"
    home_hub: Coord
    position: Coord
    battery: float = 100.0
    payload_kg: float = 0.0
    current_route: List[Coord] = field(default_factory=list)
    status: str = "idle"            # idle | delivering | returning | failed


Delivery = namedtuple(
    "Delivery",
    ["id", "hub", "pickup", "dropoff", "weight_kg", "priority"],
)


GRID_SIZE = 10


def make_empty_grid(size: int = GRID_SIZE) -> List[List[Cell]]:
    return [
        [Cell(row=r, col=c, zone="OpenField", density=0) for c in range(size)]
        for r in range(size)
    ]


def make_sample_grid() -> List[List[Cell]]:
    """A 10x10 reference grid used by every module during development.

    Designed to mostly satisfy the CSP rules (R1–R4) so other modules can
    run end-to-end before the validator is finished.
    """
    grid = make_empty_grid()

    # Two drone hubs
    grid[2][2].is_hub = True
    grid[7][7].is_hub = True

    # Charging pads within 2 cells of each hub (R3)
    grid[2][3].is_charging = True
    grid[7][6].is_charging = True

    # Hospital + medical pickup adjacent (R4)
    grid[1][8].zone = "Hospital"
    grid[1][8].density = 200
    grid[1][9].is_medical_pickup = True
    grid[1][9].zone = "OpenField"

    # A school placed away from industrial (R1)
    grid[8][1].zone = "School"
    grid[8][1].density = 500

    # Industrial in bottom-left corner, not adjacent to school/hospital (R1)
    grid[5][0].zone = "Industrial"
    grid[5][0].density = 100
    grid[6][0].zone = "Industrial"
    grid[6][0].density = 100

    # Commercial corridor across row 4 (cheaper cost path)
    for c in range(1, 9):
        grid[4][c].zone = "Commercial"
        grid[4][c].density = 1500
        grid[4][c].cost = 0.8

    # Residential cluster around hub (2,2) — within 3 cells (R2)
    residential_cells = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1),
        (3, 0), (3, 1), (3, 2), (3, 3),
        (5, 5), (5, 6), (5, 7),
        (6, 5), (6, 6), (6, 7), (6, 8),
        (7, 5), (7, 8),
        (8, 6), (8, 7), (8, 8),
    ]
    for (r, c) in residential_cells:
        grid[r][c].zone = "Residential"
        grid[r][c].density = 4500

    # A second commercial pocket (varied data)
    grid[0][7].zone = "Commercial"
    grid[0][7].density = 1200
    grid[0][7].cost = 0.8
    grid[0][8].zone = "Commercial"
    grid[0][8].density = 1200
    grid[0][8].cost = 0.8

    return grid


def cell_glyph(cell: Cell) -> str:
    """Single-character console representation, with overlay markers."""
    if cell.no_fly:
        return "X"
    if cell.is_hub:
        return "*"
    if cell.is_charging:
        return "^"
    if cell.is_medical_pickup:
        return "+"
    return ZONE_CODE[cell.zone]


def print_grid(grid: List[List[Cell]]) -> None:
    n = len(grid)
    header = "    " + " ".join(f"{c}" for c in range(n))
    print(header)
    print("   " + "--" * n)
    for r in range(n):
        row_str = " ".join(cell_glyph(grid[r][c]) for c in range(n))
        print(f"{r:2} | {row_str}")
    print()
    print("Legend: R=Residential C=Commercial H=Hospital S=School "
          "I=Industrial O=OpenField  *=Hub ^=Charging +=Medical X=NoFly")


def find_cells(grid: List[List[Cell]], predicate) -> List[Cell]:
    return [c for row in grid for c in row if predicate(c)]


def hubs(grid: List[List[Cell]]) -> List[Cell]:
    return find_cells(grid, lambda c: c.is_hub)


def charging_pads(grid: List[List[Cell]]) -> List[Cell]:
    return find_cells(grid, lambda c: c.is_charging)


def medical_pickups(grid: List[List[Cell]]) -> List[Cell]:
    return find_cells(grid, lambda c: c.is_medical_pickup)


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(row: int, col: int, size: int = GRID_SIZE) -> List[Coord]:
    candidates = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
    return [(r, c) for (r, c) in candidates if 0 <= r < size and 0 <= c < size]


if __name__ == "__main__":
    g = make_sample_grid()
    print_grid(g)
    print(f"\nHubs:           {[(c.row, c.col) for c in hubs(g)]}")
    print(f"Charging pads:  {[(c.row, c.col) for c in charging_pads(g)]}")
    print(f"Medical pickup: {[(c.row, c.col) for c in medical_pickups(g)]}")
