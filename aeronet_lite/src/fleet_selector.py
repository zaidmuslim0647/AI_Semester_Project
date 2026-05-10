"""Fleet selection module for AeroNet Lite.

Owner: Rafay
Selects a fleet under budget using brute force or a Genetic Algorithm.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from grid_model import Cell, Drone, Coord, hubs, manhattan
from delivery_generator import load_demand_forecast, apply_stub_demand, DEFAULT_DEMAND_CSV


LIGHT_COST = 1000
HEAVY_COST = 1800

LIGHT_PAYLOAD = 2.0
HEAVY_PAYLOAD = 5.0

LIGHT_RANGE = 12
HEAVY_RANGE = 20


@dataclass(frozen=True)
class FleetScore:
    light: int
    heavy: int
    cost: int
    coverage: float
    budget_used: float
    score: float


def _all_cells(grid: List[List[Cell]]) -> Iterable[Cell]:
    for row in grid:
        for cell in row:
            yield cell


def _ensure_demand(grid: List[List[Cell]], demand_csv_path: str, seed: int) -> None:
    loaded = load_demand_forecast(grid, demand_csv_path)
    if not loaded:
        apply_stub_demand(grid, seed=seed)


def _hub_coords(grid: List[List[Cell]]) -> List[Coord]:
    return [(c.row, c.col) for c in hubs(grid)]


def _fleet_cost(light_count: int, heavy_count: int) -> int:
    return light_count * LIGHT_COST + heavy_count * HEAVY_COST


def _fleet_payload_capacity(light_count: int, heavy_count: int) -> float:
    return light_count * LIGHT_PAYLOAD + heavy_count * HEAVY_PAYLOAD


def _fleet_max_range(light_count: int, heavy_count: int) -> int:
    max_range = 0
    if light_count > 0:
        max_range = max(max_range, LIGHT_RANGE)
    if heavy_count > 0:
        max_range = max(max_range, HEAVY_RANGE)
    return max_range


def _estimate_coverage(
    grid: List[List[Cell]],
    hub_coords: List[Coord],
    light_count: int,
    heavy_count: int,
) -> float:
    if not hub_coords or (light_count + heavy_count) == 0:
        return 0.0

    max_range = _fleet_max_range(light_count, heavy_count)
    total_demand = sum(c.demand for c in _all_cells(grid) if not c.no_fly)
    if total_demand <= 0:
        return 0.0

    demand_covered = 0.0
    for cell in _all_cells(grid):
        if cell.no_fly:
            continue
        closest_hub_dist = min(manhattan((cell.row, cell.col), h) for h in hub_coords)
        if closest_hub_dist <= max_range:
            demand_covered += cell.demand

    payload_capacity = _fleet_payload_capacity(light_count, heavy_count)
    payload_factor = min(1.0, payload_capacity / max(1.0, total_demand))

    coverage = (demand_covered / total_demand) * payload_factor
    return max(0.0, min(1.0, coverage))


def _score_fleet(
    grid: List[List[Cell]],
    hub_coords: List[Coord],
    budget: int,
    light_count: int,
    heavy_count: int,
) -> FleetScore:
    cost = _fleet_cost(light_count, heavy_count)
    budget_used = cost / budget if budget > 0 else 1.0
    if cost > budget:
        return FleetScore(light_count, heavy_count, cost, 0.0, budget_used, -1.0)

    coverage = _estimate_coverage(grid, hub_coords, light_count, heavy_count)
    score = 0.75 * coverage - 0.25 * budget_used
    return FleetScore(light_count, heavy_count, cost, coverage, budget_used, score)


def _brute_force_best(
    grid: List[List[Cell]],
    budget: int,
    hub_coords: List[Coord],
) -> FleetScore:
    best = FleetScore(0, 0, 0, 0.0, 0.0, -1.0)
    max_light = budget // LIGHT_COST if budget > 0 else 0
    max_heavy = budget // HEAVY_COST if budget > 0 else 0

    for light_count in range(max_light + 1):
        for heavy_count in range(max_heavy + 1):
            score = _score_fleet(grid, hub_coords, budget, light_count, heavy_count)
            if score.score > best.score:
                best = score

    return best


def _repair_chromosome(light_count: int, heavy_count: int, budget: int) -> Tuple[int, int]:
    light_count = max(0, light_count)
    heavy_count = max(0, heavy_count)
    while _fleet_cost(light_count, heavy_count) > budget and (light_count + heavy_count) > 0:
        if heavy_count > 0:
            heavy_count -= 1
        elif light_count > 0:
            light_count -= 1
    return light_count, heavy_count


def _random_chromosome(rng: random.Random, budget: int) -> Tuple[int, int]:
    max_light = budget // LIGHT_COST if budget > 0 else 0
    max_heavy = budget // HEAVY_COST if budget > 0 else 0
    light_count = rng.randint(0, max_light) if max_light > 0 else 0
    heavy_count = rng.randint(0, max_heavy) if max_heavy > 0 else 0
    return _repair_chromosome(light_count, heavy_count, budget)


def _tournament_select(
    rng: random.Random, population: List[Tuple[int, int]], scores: List[float], k: int = 3
) -> Tuple[int, int]:
    best_idx = None
    for _ in range(k):
        idx = rng.randrange(len(population))
        if best_idx is None or scores[idx] > scores[best_idx]:
            best_idx = idx
    return population[best_idx]


def _crossover(
    rng: random.Random, parent_a: Tuple[int, int], parent_b: Tuple[int, int]
) -> Tuple[int, int]:
    if rng.random() < 0.5:
        return parent_a[0], parent_b[1]
    return parent_b[0], parent_a[1]


def _mutate(
    rng: random.Random, chromo: Tuple[int, int], mutation_rate: float, budget: int
) -> Tuple[int, int]:
    light_count, heavy_count = chromo
    if rng.random() < mutation_rate:
        light_count += rng.choice([-1, 1])
    if rng.random() < mutation_rate:
        heavy_count += rng.choice([-1, 1])
    return _repair_chromosome(light_count, heavy_count, budget)


def _ga_best(
    grid: List[List[Cell]],
    budget: int,
    hub_coords: List[Coord],
    seed: int = 42,
    population_size: int = 30,
    generations: int = 40,
    mutation_rate: float = 0.2,
    elite_size: int = 2,
) -> FleetScore:
    rng = random.Random(seed)
    population = [_random_chromosome(rng, budget) for _ in range(population_size)]

    best_overall = FleetScore(0, 0, 0, 0.0, 0.0, -1.0)

    for _ in range(generations):
        scores = [
            _score_fleet(grid, hub_coords, budget, c[0], c[1]).score
            for c in population
        ]

        ranked = sorted(
            zip(population, scores), key=lambda x: x[1], reverse=True
        )
        population = [p for p, _ in ranked]
        scores = [s for _, s in ranked]

        top = _score_fleet(grid, hub_coords, budget, population[0][0], population[0][1])
        if top.score > best_overall.score:
            best_overall = top

        new_population = population[:elite_size]
        while len(new_population) < population_size:
            parent_a = _tournament_select(rng, population, scores)
            parent_b = _tournament_select(rng, population, scores)
            child = _crossover(rng, parent_a, parent_b)
            child = _mutate(rng, child, mutation_rate, budget)
            new_population.append(child)

        population = new_population

    return best_overall


def _build_fleet(light_count: int, heavy_count: int, hub_coords: List[Coord]) -> List[Drone]:
    drones: List[Drone] = []
    if not hub_coords:
        return drones

    hub_cycle = list(hub_coords)
    hub_index = 0

    def next_hub() -> Coord:
        nonlocal hub_index
        hub = hub_cycle[hub_index % len(hub_cycle)]
        hub_index += 1
        return hub

    for i in range(light_count):
        hub = next_hub()
        drones.append(
            Drone(
                id=f"L{i + 1}",
                type="light",
                home_hub=hub,
                position=hub,
                battery=100.0,
                payload_kg=0.0,
                current_route=[],
                status="idle",
            )
        )

    for i in range(heavy_count):
        hub = next_hub()
        drones.append(
            Drone(
                id=f"H{i + 1}",
                type="heavy",
                home_hub=hub,
                position=hub,
                battery=100.0,
                payload_kg=0.0,
                current_route=[],
                status="idle",
            )
        )

    return drones


def select_fleet(
    grid: List[List[Cell]],
    budget: int,
    mode: str = "ga",
    seed: int = 42,
    demand_csv_path: str = DEFAULT_DEMAND_CSV,
) -> List[Drone]:
    """Select a fleet under budget and return a list of Drone objects."""
    if budget <= 0:
        return []

    _ensure_demand(grid, demand_csv_path, seed=seed)
    hub_coords = _hub_coords(grid)
    if not hub_coords:
        return []

    if mode == "brute":
        best = _brute_force_best(grid, budget, hub_coords)
    else:
        best = _ga_best(grid, budget, hub_coords, seed=seed)

    return _build_fleet(best.light, best.heavy, hub_coords)
