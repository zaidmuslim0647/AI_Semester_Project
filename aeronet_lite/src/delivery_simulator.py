"""Delivery Simulator for AeroNet Lite — Modules 3 & 4.

Owner: Hasaan.

This module provides the drone movement loop (Module 3 integration) and the
real-time disruption handler (Module 4).  It is designed so that Zaid's
``main.py`` only needs to call four functions:

    state = init_state(grid, drones, deliveries)
    state = assign_delivery(state, delivery, drone)
    state = step_simulation(state)
    state = activate_no_fly(state, row, col)
    print_summary(state)
    print_log(state)

State shape (plain dict — no extra imports needed by callers)
-------------------------------------------------------------
{
    "grid":               List[List[Cell]],
    "drones":             Dict[str, Drone],        # drone_id -> Drone
    "deliveries":         List[Delivery],
    "assignments":        Dict[str, Delivery],      # drone_id -> active Delivery
    "waypoints":          Dict[str, dict],          # drone_id -> phase tracker
    "step":               int,
    "log":                List[str],                # full chronological event log
    "completed":          List[str],                # delivery IDs finished
    "delayed":            List[str],                # delivery IDs that could not start
    "failed_deliveries":  List[str],                # delivery IDs that failed mid-route
}

Waypoints entry shape
---------------------
{
    "pickup":  Coord,
    "dropoff": Coord,
    "hub":     Coord,
    "phase":   str,   # "to_pickup" | "to_dropoff" | "to_hub"
}

Battery model (derived from spec: Light range=12 cells, Heavy range=20 cells)
-----------------------------------------------------------------------------
    Light drone: 100 / 12 ≈ 8.33 battery units per cell
    Heavy drone: 100 / 20 = 5.00 battery units per cell
"""

from __future__ import annotations

from typing import Dict, List, Optional

from grid_model import Cell, Coord, Delivery, Drone
from astar_planner import astar, plan_delivery_route


# --------------------------------------------------------------------------- #
# Battery drain constants
# --------------------------------------------------------------------------- #

BATTERY_DRAIN: Dict[str, float] = {
    "light": 100.0 / 12,   # ≈ 8.33 per cell
    "heavy": 100.0 / 20,   # = 5.00 per cell
}

_DEFAULT_DRAIN = 100.0 / 12   # fallback if drone type is unrecognised

# Convenience type alias for the phase string stored in waypoints
_Phase = str  # "to_pickup" | "to_dropoff" | "to_hub"


# --------------------------------------------------------------------------- #
# State initialisation
# --------------------------------------------------------------------------- #

def init_state(
    grid: List[List[Cell]],
    drones: List[Drone],
    deliveries: Optional[List[Delivery]] = None,
) -> dict:
    """Create and return a fresh simulator state dict.

    Parameters
    ----------
    grid:       10×10 Cell grid produced by ``make_sample_grid()``.
    drones:     List of Drone objects (must be idle and at their home_hub).
    deliveries: Optional initial list of Delivery namedtuples (informational
                only; use ``assign_delivery`` to actually dispatch them).
    """
    return {
        "grid":               grid,
        "drones":             {d.id: d for d in drones},
        "deliveries":         list(deliveries) if deliveries else [],
        "assignments":        {},   # drone_id -> Delivery
        "waypoints":          {},   # drone_id -> {pickup, dropoff, hub, phase}
        "step":               0,
        "log":                [],
        "completed":          [],
        "delayed":            [],
        "failed_deliveries":  [],
    }


# --------------------------------------------------------------------------- #
# Delivery assignment
# --------------------------------------------------------------------------- #

def assign_delivery(state: dict, delivery: Delivery, drone: Drone) -> dict:
    """Assign *delivery* to *drone* and pre-compute its full A* route.

    Expects the drone to be idle and located at ``delivery.hub``.
    If no valid route exists the delivery is added to ``state["delayed"]``
    and the drone remains idle.
    """
    step = state["step"]

    result = plan_delivery_route(delivery, state["grid"])
    if not result["success"]:
        _log(state,
             f"Step {step}: Cannot assign delivery {delivery.id} to "
             f"drone {drone.id} — {result['reason']}")
        state["delayed"].append(delivery.id)
        return state

    # result["path"][0] == delivery.hub (drone's current position); exclude it.
    drone.current_route = list(result["path"][1:])
    drone.status        = "delivering"
    drone.payload_kg    = delivery.weight_kg

    state["assignments"][drone.id] = delivery
    state["waypoints"][drone.id] = {
        "pickup":  delivery.pickup,
        "dropoff": delivery.dropoff,
        "hub":     delivery.hub,
        "phase":   "to_pickup",
    }
    _log(state,
         f"Step {step}: Delivery {delivery.id} assigned to drone {drone.id} "
         f"(route {len(result['path'])} cells, cost {result['cost']:.1f}).")
    return state


# --------------------------------------------------------------------------- #
# Simulation step — Module 3 core
# --------------------------------------------------------------------------- #

def step_simulation(state: dict) -> dict:
    """Advance every active drone by exactly one cell along its planned route.

    Per-drone logic
    ---------------
    1. If ``current_route`` is empty the drone has completed its journey and
       is marked idle; its delivery is logged as completed.
    2. If the next cell is no-fly (activated after the route was planned) an
       immediate reroute is attempted via A*.
    3. Otherwise the drone moves, battery is drained, and waypoint events
       (pickup / drop-off) are logged when reached.
    4. A drone whose battery hits 0 is marked ``failed``.

    Returns the mutated state dict (same object for in-place use).
    """
    state["step"] += 1
    step = state["step"]

    for drone_id, drone in state["drones"].items():
        if drone.status in ("idle", "failed"):
            continue

        # ------------------------------------------------------------------ #
        # Route exhausted → delivery complete, drone back at hub
        # ------------------------------------------------------------------ #
        if not drone.current_route:
            delivery = state["assignments"].pop(drone_id, None)
            if delivery:
                state["completed"].append(delivery.id)
                _log(state,
                     f"Step {step}: Drone {drone_id} completed delivery "
                     f"{delivery.id} — returned to hub.")
            drone.status      = "idle"
            drone.payload_kg  = 0.0
            state["waypoints"].pop(drone_id, None)
            continue

        # ------------------------------------------------------------------ #
        # Peek at next cell — handle newly activated no-fly before moving
        # ------------------------------------------------------------------ #
        next_coord = drone.current_route[0]
        nr, nc = next_coord
        if state["grid"][nr][nc].no_fly:
            _log(state,
                 f"Step {step}: Drone {drone_id} route blocked at "
                 f"{next_coord} (no-fly). Rerouting via A*.")
            if not _reroute(state, drone, step):
                _fail_drone(state, drone, step, reason="no safe path after block")
            continue

        # ------------------------------------------------------------------ #
        # Move one cell
        # ------------------------------------------------------------------ #
        drone.current_route.pop(0)
        drone.position = next_coord
        drain          = BATTERY_DRAIN.get(drone.type, _DEFAULT_DRAIN)
        drone.battery  = max(0.0, drone.battery - drain)

        if drone.battery <= 0.0:
            _log(state,
                 f"Step {step}: Drone {drone_id} battery depleted "
                 f"at {drone.position}.")
            _fail_drone(state, drone, step, reason="battery depleted")
            continue

        # ------------------------------------------------------------------ #
        # Waypoint events (pickup / drop-off)
        # ------------------------------------------------------------------ #
        waypts = state["waypoints"].get(drone_id)
        if waypts:
            phase: _Phase = waypts["phase"]

            if phase == "to_pickup" and drone.position == waypts["pickup"]:
                waypts["phase"] = "to_dropoff"
                _log(state,
                     f"Step {step}: Drone {drone_id} picked up package "
                     f"at {drone.position}.")

            elif phase == "to_dropoff" and drone.position == waypts["dropoff"]:
                waypts["phase"]  = "to_hub"
                drone.payload_kg = 0.0
                _log(state,
                     f"Step {step}: Drone {drone_id} dropped off package "
                     f"at {drone.position}.")

    return state


# --------------------------------------------------------------------------- #
# Disruption handler — Module 4
# --------------------------------------------------------------------------- #

def activate_no_fly(state: dict, row: int, col: int) -> dict:
    """Mark cell *(row, col)* as no-fly and handle every affected drone.

    Three cases are handled in priority order for each active drone:

    1. **Drone is sitting on the newly blocked cell.**
       The active delivery is delayed and the drone is rerouted home via A*.
       If that too fails, the drone is marked ``failed``.

    2. **Drone's remaining route passes through the blocked cell.**
       A* rereoutes from the drone's current position through its remaining
       waypoints (determined by phase). Delivery is marked ``failed`` only if
       no alternative path exists.

    3. **Drone's route does not cross the blocked cell.**
       No action needed.
    """
    step = state["step"]
    state["grid"][row][col].no_fly = True
    _log(state, f"Step {step}: No-fly cell activated at ({row}, {col}).")

    for drone_id, drone in state["drones"].items():
        if drone.status in ("idle", "failed"):
            continue

        # ------------------------------------------------------------------ #
        # Case 1: drone is currently ON the restricted cell
        # ------------------------------------------------------------------ #
        if drone.position == (row, col):
            _log(state,
                 f"Step {step}: Drone {drone_id} is ON the new no-fly cell. "
                 f"Forcing return to home hub.")
            delivery = state["assignments"].pop(drone_id, None)
            if delivery:
                state["delayed"].append(delivery.id)
            state["waypoints"].pop(drone_id, None)

            # The drone is physically at this cell so allow it to start moving
            # out: temporarily lift the flag just for the astar call.
            state["grid"][row][col].no_fly = False
            result = astar(drone.position, drone.home_hub, state["grid"])
            state["grid"][row][col].no_fly = True
            if result["success"]:
                drone.current_route = list(result["path"][1:])
                drone.status        = "returning"
                _log(state,
                     f"Step {step}: Drone {drone_id} rerouted to home hub via A*.")
            else:
                drone.status = "failed"
                _log(state,
                     f"Step {step}: Drone {drone_id} cannot escape no-fly cell. "
                     f"FAILED.")
            continue

        # ------------------------------------------------------------------ #
        # Case 2: planned route crosses the newly blocked cell
        # ------------------------------------------------------------------ #
        if (row, col) in drone.current_route:
            _log(state,
                 f"Step {step}: Drone {drone_id} route crosses new no-fly at "
                 f"({row},{col}). Rerouting via A*.")
            if not _reroute(state, drone, step):
                _fail_drone(state, drone, step,
                            reason=f"no path around no-fly at ({row},{col})")

        # Case 3: route unaffected — nothing to do

    return state


# --------------------------------------------------------------------------- #
# Summary and log printers (for Zaid's main.py / demo)
# --------------------------------------------------------------------------- #

def print_summary(state: dict) -> None:
    """Print the final simulation summary line (matches spec Step 20 format)."""
    n_done  = len(state["completed"])
    n_delay = len(state["delayed"])
    n_fail  = len(state["failed_deliveries"])
    print(f"\n{'=' * 52}")
    print(f"Step {state['step']}: Simulation complete. "
          f"{n_done} completed, {n_delay} delayed, {n_fail} failed.")
    print(f"{'=' * 52}")


def print_log(state: dict) -> None:
    """Print the full chronological event log to stdout."""
    for entry in state["log"]:
        print(entry)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _log(state: dict, message: str) -> None:
    state["log"].append(message)


def _fail_drone(
    state: dict,
    drone: Drone,
    step: int,
    reason: str = "",
) -> None:
    """Mark *drone* as failed and move its active delivery to the failed list."""
    drone.status = "failed"
    delivery     = state["assignments"].pop(drone.id, None)
    if delivery:
        state["failed_deliveries"].append(delivery.id)
    state["waypoints"].pop(drone.id, None)
    reason_str = f" ({reason})" if reason else ""
    _log(state, f"Step {step}: Drone {drone.id} marked FAILED{reason_str}.")


def _reroute(state: dict, drone: Drone, step: int) -> bool:
    """Re-plan *drone*'s route from its current position to its remaining targets.

    Uses the ``phase`` stored in ``state["waypoints"]`` to determine which
    waypoints are still ahead:

        "to_pickup"  → remaining targets: [pickup, dropoff, hub]
        "to_dropoff" → remaining targets: [dropoff, hub]
        "to_hub"     → remaining targets: [hub]

    If the drone has no active delivery (e.g. it was forced into "returning"
    status) it routes straight to its home hub.

    Returns True if a valid alternative route was found, False otherwise.
    """
    waypts = state["waypoints"].get(drone.id)

    # ---------------------------------------------------------------------- #
    # No active delivery — route back to home hub
    # ---------------------------------------------------------------------- #
    if waypts is None:
        result = astar(drone.position, drone.home_hub, state["grid"])
        if result["success"]:
            drone.current_route = list(result["path"][1:])
            _log(state, f"Step {step}: Drone {drone.id} rerouted to home hub.")
            return True
        return False

    # ---------------------------------------------------------------------- #
    # Active delivery — rebuild route through remaining waypoints
    # ---------------------------------------------------------------------- #
    phase:   _Phase = waypts["phase"]
    pickup:  Coord  = waypts["pickup"]
    dropoff: Coord  = waypts["dropoff"]
    hub:     Coord  = waypts["hub"]

    if phase == "to_pickup":
        targets = [pickup, dropoff, hub]
    elif phase == "to_dropoff":
        targets = [dropoff, hub]
    else:                          # "to_hub"
        targets = [hub]

    full_path: List[Coord] = []
    current = drone.position

    for target in targets:
        result = astar(current, target, state["grid"])
        if not result["success"]:
            _log(state,
                 f"Step {step}: Drone {drone.id} reroute failed — "
                 f"{result['reason']}")
            return False
        segment = result["path"] if not full_path else result["path"][1:]
        full_path.extend(segment)
        current = target

    # full_path[0] == drone.position (already there); exclude it
    drone.current_route = list(full_path[1:]) if len(full_path) > 1 else []
    _log(state,
         f"Step {step}: Drone {drone.id} rerouted via A* "
         f"({len(drone.current_route)} cells remaining).")
    return True


# --------------------------------------------------------------------------- #
# Quick self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from grid_model import make_sample_grid

    grid = make_sample_grid()

    drones = [
        Drone(id="D1", type="light", home_hub=(2, 2), position=(2, 2)),
        Drone(id="D2", type="heavy", home_hub=(7, 7), position=(7, 7)),
    ]
    deliveries = [
        Delivery(id="DEL1", hub=(2, 2), pickup=(0, 0), dropoff=(5, 5),
                 weight_kg=1.0, priority="normal"),
        Delivery(id="DEL2", hub=(7, 7), pickup=(9, 9), dropoff=(3, 3),
                 weight_kg=3.0, priority="high"),
    ]

    state = init_state(grid, drones, deliveries)

    print("=== Assigning deliveries ===")
    state = assign_delivery(state, deliveries[0], drones[0])
    state = assign_delivery(state, deliveries[1], drones[1])

    print("\n=== Running 20 steps (disruption at step 11) ===")
    for i in range(1, 21):
        if i == 11:
            print(f"\n--- Disruption event at step {i} ---")
            state = activate_no_fly(state, 4, 4)
        state = step_simulation(state)

    print("\n=== Event log ===")
    print_log(state)
    print_summary(state)
