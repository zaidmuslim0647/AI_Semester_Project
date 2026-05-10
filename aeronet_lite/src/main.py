"""AeroNet Lite — 20-step simulation orchestrator.

Owner: Zaid. Imports each teammate's module if present; otherwise falls back to
inline stubs so the simulation runs end-to-end on Zaid's branch alone. After
PRs from Hasaan, Rafay, and Saad are merged, the real implementations take over
automatically with no changes here.

Stub fallbacks (matching PROJECT_PLAN.md section 6 risk table):
  - Hasaan missing  -> straight-line mock A* path
  - Rafay missing   -> hardcoded fleet (3 light, 2 heavy) and 5 deliveries
  - Saad missing    -> uniform random demand and "Normal" anomaly labels
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from grid_model import (
    Cell,
    Coord,
    Delivery,
    Drone,
    GRID_SIZE,
    hubs,
    make_sample_grid,
    manhattan,
    print_grid,
)
from layout_validator import validate_layout


# ---------------------------------------------------------------------------- #
# Teammate modules — try real imports, fall back to stubs                       #
# ---------------------------------------------------------------------------- #

# --- Hasaan: A* planner ----------------------------------------------------- #
try:
    from astar_planner import astar as _real_astar  # type: ignore
    HAS_REAL_ASTAR = True
except Exception:
    HAS_REAL_ASTAR = False


def _stub_astar(start: Coord, goal: Coord, grid: List[List[Cell]]) -> Dict:
    """Straight-line mock: walk rows then cols, ignoring no-fly cells."""
    path = [start]
    r, c = start
    gr, gc = goal
    while r != gr:
        r += 1 if gr > r else -1
        path.append((r, c))
    while c != gc:
        c += 1 if gc > c else -1
        path.append((r, c))
    cost = float(len(path) - 1)
    return {"path": path, "cost": cost, "success": True, "reason": "stub-straight-line"}


def astar(start: Coord, goal: Coord, grid: List[List[Cell]]) -> Dict:
    return _real_astar(start, goal, grid) if HAS_REAL_ASTAR else _stub_astar(start, goal, grid)


# --- Rafay: fleet selector + delivery generator ----------------------------- #
try:
    from fleet_selector import select_fleet as _real_select_fleet  # type: ignore
    HAS_REAL_FLEET = True
except Exception:
    HAS_REAL_FLEET = False

try:
    from delivery_generator import generate_deliveries as _real_generate_deliveries  # type: ignore
    HAS_REAL_DELIVERY_GEN = True
except Exception:
    HAS_REAL_DELIVERY_GEN = False


def _stub_select_fleet(grid: List[List[Cell]], budget: int = 8000) -> List[Drone]:
    hub_list = hubs(grid)
    if not hub_list:
        return []
    fleet: List[Drone] = []
    counter = 1
    for i in range(3):
        h = hub_list[i % len(hub_list)]
        fleet.append(Drone(id=f"D{counter}", type="light",
                           home_hub=(h.row, h.col), position=(h.row, h.col)))
        counter += 1
    for i in range(2):
        h = hub_list[i % len(hub_list)]
        fleet.append(Drone(id=f"D{counter}", type="heavy",
                           home_hub=(h.row, h.col), position=(h.row, h.col)))
        counter += 1
    return fleet


def _stub_generate_deliveries(grid: List[List[Cell]], n: int = 5) -> List[Delivery]:
    hub_list = [(h.row, h.col) for h in hubs(grid)]
    if not hub_list:
        return []
    rng = random.Random(42)
    residential = [(c.row, c.col) for row in grid for c in row if c.zone == "Residential"]
    pickups = [(c.row, c.col) for row in grid for c in row
               if c.is_medical_pickup or c.zone == "Commercial"]
    deliveries: List[Delivery] = []
    for i in range(n):
        hub = rng.choice(hub_list)
        pickup = rng.choice(pickups) if pickups else hub
        dropoff = rng.choice(residential) if residential else hub
        deliveries.append(Delivery(
            id=f"DL{i+1}", hub=hub, pickup=pickup, dropoff=dropoff,
            weight_kg=round(rng.uniform(0.5, 4.5), 1),
            priority=rng.choice(["normal", "normal", "high"]),
        ))
    return deliveries


def select_fleet(grid: List[List[Cell]], budget: int = 8000) -> List[Drone]:
    return _real_select_fleet(grid, budget) if HAS_REAL_FLEET else _stub_select_fleet(grid, budget)


def generate_deliveries(grid: List[List[Cell]], n: int = 5) -> List[Delivery]:
    if HAS_REAL_DELIVERY_GEN:
        return _real_generate_deliveries(grid, n)
    return _stub_generate_deliveries(grid, n)


def assign_nearest(deliveries: List[Delivery], drones: List[Drone]) -> Dict[str, str]:
    """Match each delivery to the nearest idle drone (Manhattan to delivery hub)."""
    assignments: Dict[str, str] = {}
    available = list(drones)
    for d in deliveries:
        if not available:
            break
        available.sort(key=lambda dr: manhattan(dr.position, d.hub))
        chosen = available.pop(0)
        assignments[d.id] = chosen.id
    return assignments


# --- Saad: ML pipeline ------------------------------------------------------ #
try:
    from ml_pipeline import (  # type: ignore
        load_demand_forecast as _real_load_demand,
        predict_anomaly as _real_predict_anomaly,
    )
    HAS_REAL_ML = True
except Exception:
    HAS_REAL_ML = False


def _stub_load_demand(grid: List[List[Cell]]) -> Dict[Coord, float]:
    rng = random.Random(7)
    return {(r, c): round(rng.uniform(0, 10), 2)
            for r in range(len(grid)) for c in range(len(grid[0]))}


def _stub_predict_anomaly(drone: Drone, step: int) -> str:
    if step == 18 and drone.id == "D3":
        return "Battery"
    return "Normal"


def load_demand_forecast(grid: List[List[Cell]]) -> Dict[Coord, float]:
    return _real_load_demand(grid) if HAS_REAL_ML else _stub_load_demand(grid)


def predict_anomaly(drone: Drone, step: int) -> str:
    return _real_predict_anomaly(drone, step) if HAS_REAL_ML else _stub_predict_anomaly(drone, step)


# ---------------------------------------------------------------------------- #
# Event log                                                                    #
# ---------------------------------------------------------------------------- #

@dataclass
class EventLog:
    entries: List[Tuple[int, str]] = field(default_factory=list)

    def log(self, step: int, message: str) -> None:
        self.entries.append((step, message))
        print(f"  Step {step:>2}: {message}")

    def header(self, step: int) -> None:
        print(f"\n--- Step {step} ---")


# ---------------------------------------------------------------------------- #
# Simulation state                                                             #
# ---------------------------------------------------------------------------- #

@dataclass
class SimState:
    grid: List[List[Cell]]
    drones: List[Drone]
    deliveries: List[Delivery]
    assignments: Dict[str, str]                        # delivery_id -> drone_id
    routes: Dict[str, List[Coord]] = field(default_factory=dict)   # drone_id -> path
    drone_targets: Dict[str, Coord] = field(default_factory=dict)  # drone_id -> goal
    completed: List[str] = field(default_factory=list)
    delayed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)


def _drone_by_id(state: SimState, drone_id: str) -> Drone:
    return next(d for d in state.drones if d.id == drone_id)


def _plan_route_for_assignment(state: SimState, delivery: Delivery, drone: Drone, log: EventLog, step: int) -> None:
    """Plan a hub -> pickup -> dropoff -> hub route for a drone."""
    segments = [
        (drone.position, delivery.pickup),
        (delivery.pickup, delivery.dropoff),
        (delivery.dropoff, drone.home_hub),
    ]
    full_path: List[Coord] = []
    for seg_start, seg_goal in segments:
        result = astar(seg_start, seg_goal, state.grid)
        if not result["success"]:
            log.log(step, f"Route planning failed for {drone.id} on delivery {delivery.id}: {result['reason']}")
            state.failed.append(delivery.id)
            return
        # Skip the first node of each segment after the first to avoid duplicates
        full_path = full_path + (result["path"] if not full_path else result["path"][1:])
    state.routes[drone.id] = full_path
    state.drone_targets[drone.id] = delivery.dropoff
    drone.current_route = full_path
    drone.status = "delivering"


def _advance_drone(drone: Drone, route: List[Coord]) -> None:
    """Move drone one step along its route. Returns when route complete."""
    try:
        idx = route.index(drone.position)
    except ValueError:
        idx = 0
    if idx + 1 < len(route):
        drone.position = route[idx + 1]
        drone.battery = max(0.0, drone.battery - 1.5)
    else:
        drone.status = "idle"


def _route_crosses_no_fly(route: List[Coord], grid: List[List[Cell]]) -> bool:
    return any(grid[r][c].no_fly for (r, c) in route)


def _reroute(state: SimState, drone: Drone, log: EventLog, step: int) -> None:
    goal = state.drone_targets.get(drone.id)
    if goal is None:
        return
    result = astar(drone.position, goal, state.grid)
    if result["success"]:
        state.routes[drone.id] = result["path"]
        drone.current_route = result["path"]
        log.log(step, f"Drone {drone.id} rerouted using A* ({len(result['path'])} cells).")
    else:
        drone.status = "failed"
        log.log(step, f"Drone {drone.id} cannot reach destination safely: {result['reason']}.")


# ---------------------------------------------------------------------------- #
# 20-step simulation                                                           #
# ---------------------------------------------------------------------------- #

def run_simulation(verbose: bool = True) -> SimState:
    log = EventLog()
    print("=" * 64)
    print("AeroNet Lite — 20-Step Simulation")
    print(f"  Real modules: astar={HAS_REAL_ASTAR}  fleet={HAS_REAL_FLEET}  "
          f"deliveries={HAS_REAL_DELIVERY_GEN}  ml={HAS_REAL_ML}")
    print("=" * 64)

    # Step 1: Initialize grid and validate layout
    log.header(1)
    grid = make_sample_grid()
    if verbose:
        print_grid(grid)
    report = validate_layout(grid)
    if report.is_valid:
        log.log(1, "Layout validation passed.")
    else:
        log.log(1, f"Layout validation FAILED ({len(report.violations)} violations). Continuing for demo purposes.")
        report.print()

    # Step 2: Inject demand forecast
    log.header(2)
    demand = load_demand_forecast(grid)
    for (r, c), value in demand.items():
        grid[r][c].demand = value
    log.log(2, f"Demand forecast loaded ({'real' if HAS_REAL_ML else 'stub'}). "
               f"Mean predicted demand = {sum(demand.values())/len(demand):.2f}.")

    # Step 3: Select fleet
    log.header(3)
    fleet = select_fleet(grid)
    light_n = sum(1 for d in fleet if d.type == "light")
    heavy_n = sum(1 for d in fleet if d.type == "heavy")
    log.log(3, f"Fleet selected: {light_n} light drones, {heavy_n} heavy drones.")

    # Steps 4-5: Generate deliveries and assign
    log.header(4)
    deliveries = generate_deliveries(grid, n=5)
    for d in deliveries:
        log.log(4, f"Delivery {d.id}: hub {d.hub} -> pickup {d.pickup} -> drop {d.dropoff} ({d.weight_kg} kg)")

    log.header(5)
    assignments = assign_nearest(deliveries, fleet)
    for delivery_id, drone_id in assignments.items():
        log.log(5, f"Delivery {delivery_id} assigned to Drone {drone_id}.")

    state = SimState(grid=grid, drones=fleet, deliveries=deliveries, assignments=assignments)

    # Step 6: Plan routes
    log.header(6)
    delivery_by_id = {d.id: d for d in deliveries}
    for delivery_id, drone_id in assignments.items():
        drone = _drone_by_id(state, drone_id)
        _plan_route_for_assignment(state, delivery_by_id[delivery_id], drone, log, step=6)
        log.log(6, f"Route planned for Drone {drone_id} -> Delivery {delivery_id} "
                   f"(length {len(state.routes.get(drone_id, []))}).")

    # Steps 7-10: Move drones
    for step in range(7, 11):
        log.header(step)
        for drone in state.drones:
            if drone.status == "delivering":
                route = state.routes.get(drone.id, [])
                _advance_drone(drone, route)
                log.log(step, f"Drone {drone.id} now at {drone.position}, battery {drone.battery:.1f}%.")

    # Step 11: Activate a no-fly cell
    log.header(11)
    no_fly_cell = (4, 7)
    state.grid[no_fly_cell[0]][no_fly_cell[1]].no_fly = True
    log.log(11, f"No-fly cell activated at {no_fly_cell}.")

    # Steps 12-14: Detect & reroute
    for step in range(12, 15):
        log.header(step)
        for drone in state.drones:
            if drone.status != "delivering":
                continue
            route_remainder = state.routes.get(drone.id, [])
            if _route_crosses_no_fly(route_remainder, state.grid):
                log.log(step, f"Drone {drone.id} route crosses no-fly cell. Replanning.")
                _reroute(state, drone, log, step)
            else:
                _advance_drone(drone, state.routes.get(drone.id, []))
                log.log(step, f"Drone {drone.id} now at {drone.position}.")

    # Steps 15-17: Demand forecast nudge + optional extra delivery
    for step in range(15, 18):
        log.header(step)
        if step == 15:
            top_demand = sorted(demand.items(), key=lambda kv: kv[1], reverse=True)[:3]
            log.log(15, f"Top-3 demand cells from forecast: {top_demand}.")
        elif step == 16:
            extra_dropoff = sorted(demand.items(), key=lambda kv: kv[1], reverse=True)[0][0]
            primary_hub = (hubs(state.grid)[0].row, hubs(state.grid)[0].col)
            extra = Delivery(
                id="DL_extra", hub=primary_hub, pickup=primary_hub,
                dropoff=extra_dropoff, weight_kg=1.0, priority="normal",
            )
            state.deliveries.append(extra)
            log.log(16, f"Extra delivery {extra.id} added based on forecast (drop {extra_dropoff}).")
        else:
            log.log(17, "Forecast accepted; fleet capacity sufficient.")

    # Step 18: Anomaly check
    log.header(18)
    for drone in state.drones:
        label = predict_anomaly(drone, step=18)
        if label != "Normal":
            log.log(18, f"{label} anomaly detected for Drone {drone.id}.")

    # Step 19: React to anomaly (force return to hub for any failed drone)
    log.header(19)
    for drone in state.drones:
        label = predict_anomaly(drone, step=18)
        if label == "Battery":
            drone.status = "returning"
            result = astar(drone.position, drone.home_hub, state.grid)
            if result["success"]:
                state.routes[drone.id] = result["path"]
                drone.current_route = result["path"]
                log.log(19, f"Drone {drone.id} forced to return to hub {drone.home_hub}.")
            else:
                drone.status = "failed"
                log.log(19, f"Drone {drone.id} cannot return to hub: {result['reason']}.")

    # Step 20: Final summary
    log.header(20)
    completed = sum(1 for d in state.drones if d.status == "idle")
    delivering = sum(1 for d in state.drones if d.status == "delivering")
    failed = sum(1 for d in state.drones if d.status == "failed")
    returning = sum(1 for d in state.drones if d.status == "returning")
    log.log(20, f"Simulation complete. drones idle/done={completed} "
                f"delivering={delivering} returning={returning} failed={failed}.")

    print("\n" + "=" * 64)
    print(f"Total events logged: {len(log.entries)}")
    print("=" * 64)
    return state


if __name__ == "__main__":
    run_simulation(verbose=True)
