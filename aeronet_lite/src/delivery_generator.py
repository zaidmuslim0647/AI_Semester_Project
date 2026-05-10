"""Delivery generation and assignment for AeroNet Lite.

Owner: Rafay
Reads: grid_model.Grid + optional demand_forecast.csv (Saad)
Writes: Delivery list and delivery→drone assignments
"""

from __future__ import annotations

import csv
import os
import random
from typing import Dict, Iterable, List, Tuple

from grid_model import Cell, Delivery, Drone, hubs, manhattan


DEFAULT_DEMAND_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "demand_forecast.csv"
)


def _all_cells(grid: List[List[Cell]]) -> Iterable[Cell]:
    for row in grid:
        for cell in row:
            yield cell


def load_demand_forecast(
    grid: List[List[Cell]], demand_csv_path: str = DEFAULT_DEMAND_CSV
) -> bool:
    """Populate Cell.demand using Saad's CSV. Returns True if loaded."""
    if not os.path.exists(demand_csv_path):
        return False

    demands: Dict[Tuple[int, int], float] = {}
    with open(demand_csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                r = int(row["row"])
                c = int(row["col"])
                d = float(row["predicted_demand"])
                demands[(r, c)] = d
            except (KeyError, ValueError):
                continue

    if not demands:
        return False

    for cell in _all_cells(grid):
        cell.demand = float(demands.get((cell.row, cell.col), 0.0))

    return True


def apply_stub_demand(grid: List[List[Cell]], seed: int = 42) -> None:
    """Fallback: generate demand from density with a little randomness."""
    rng = random.Random(seed)
    for cell in _all_cells(grid):
        if cell.no_fly:
            cell.demand = 0.0
            continue
        density_factor = min(1.0, cell.density / 5000.0) if cell.density > 0 else 0.1
        jitter = rng.uniform(0.6, 1.4)
        cell.demand = round(density_factor * jitter, 3)


def _ensure_demand(
    grid: List[List[Cell]], demand_csv_path: str = DEFAULT_DEMAND_CSV, seed: int = 42
) -> None:
    loaded = load_demand_forecast(grid, demand_csv_path)
    if not loaded:
        apply_stub_demand(grid, seed=seed)


def _weighted_choice(cells: List[Cell], rng: random.Random) -> Cell:
    weights = [max(0.0, c.demand) for c in cells]
    if sum(weights) == 0:
        return rng.choice(cells)
    return rng.choices(cells, weights=weights, k=1)[0]


def _preferred_dropoff_cells(grid: List[List[Cell]]) -> List[Cell]:
    preferred_zones = {"Residential", "School", "Hospital", "Commercial"}
    cells = [
        c
        for c in _all_cells(grid)
        if (not c.no_fly) and (c.zone in preferred_zones)
    ]
    return cells


def generate_deliveries(
    grid: List[List[Cell]],
    n: int = 8,
    seed: int = 42,
    demand_csv_path: str = DEFAULT_DEMAND_CSV,
) -> List[Delivery]:
    """Generate a list of delivery requests.

    - Pickups are weighted by demand.
    - Dropoffs prefer residential/school/hospital/commercial.
    """
    rng = random.Random(seed)
    _ensure_demand(grid, demand_csv_path=demand_csv_path, seed=seed)

    hub_cells = hubs(grid)
    if not hub_cells:
        raise ValueError("No hubs found in grid; cannot generate deliveries.")

    pickup_candidates = [c for c in _all_cells(grid) if (not c.no_fly) and c.demand > 0]
    if not pickup_candidates:
        pickup_candidates = [c for c in _all_cells(grid) if not c.no_fly]

    dropoff_candidates = _preferred_dropoff_cells(grid)
    if not dropoff_candidates:
        dropoff_candidates = [c for c in _all_cells(grid) if not c.no_fly]

    deliveries: List[Delivery] = []
    for i in range(n):
        hub = rng.choice(hub_cells)
        pickup = _weighted_choice(pickup_candidates, rng)

        dropoff = rng.choice(dropoff_candidates)
        if dropoff.row == pickup.row and dropoff.col == pickup.col:
            dropoff = rng.choice(dropoff_candidates)

        is_medical = pickup.is_medical_pickup or pickup.zone == "Hospital"
        if is_medical:
            weight_kg = round(rng.uniform(0.5, 3.0), 2)
            priority = 1
        elif pickup.zone in {"Industrial", "Commercial"}:
            weight_kg = round(rng.uniform(1.5, 5.0), 2)
            priority = 2
        else:
            weight_kg = round(rng.uniform(0.5, 2.5), 2)
            priority = 3

        deliveries.append(
            Delivery(
                id=f"DEL-{i + 1}",
                hub=(hub.row, hub.col),
                pickup=(pickup.row, pickup.col),
                dropoff=(dropoff.row, dropoff.col),
                weight_kg=weight_kg,
                priority=priority,
            )
        )

    return deliveries


def _drone_payload_capacity(drone: Drone) -> float:
    if drone.type == "heavy":
        return 5.0
    return 2.0


def assign_deliveries(
    deliveries: List[Delivery], drones: List[Drone]
) -> Tuple[Dict[str, str], List[Delivery]]:
    """Assign each delivery to the nearest available drone.

    Returns (assignments, unassigned).
    """
    assignments: Dict[str, str] = {}
    unassigned: List[Delivery] = []

    for delivery in deliveries:
        candidates = [
            d
            for d in drones
            if d.status == "idle" and _drone_payload_capacity(d) >= delivery.weight_kg
        ]
        if not candidates:
            unassigned.append(delivery)
            continue

        best = min(
            candidates,
            key=lambda d: (
                manhattan(d.position, delivery.pickup),
                -_drone_payload_capacity(d),
            ),
        )
        assignments[delivery.id] = best.id
        best.status = "delivering"
        best.payload_kg = delivery.weight_kg

    return assignments, unassigned


def assign(deliveries: List[Delivery], drones: List[Drone]):
    """Compatibility wrapper matching the spec name."""
    return assign_deliveries(deliveries, drones)
