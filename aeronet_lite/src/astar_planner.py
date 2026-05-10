"""A* Path Planner for AeroNet Lite — Module 3.

Owner: Hasaan.

Public API
----------
    astar(start, goal, grid)            -> RouteResult
    plan_delivery_route(delivery, grid) -> RouteResult

RouteResult (frozen contract, Section 3.3 of PROJECT_PLAN)
-----------------------------------------------------------
    {
        "path":    List[Coord],   # ordered list of (row, col) from start to goal
        "cost":    float,         # total g-cost along the chosen path
        "success": bool,
        "reason":  str,
    }
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional

from grid_model import Cell, Coord, Delivery, GRID_SIZE, get_neighbors, manhattan


# --------------------------------------------------------------------------- #
# Core A* search
# --------------------------------------------------------------------------- #

def astar(start: Coord, goal: Coord, grid: List[List[Cell]]) -> dict:
    """Find the cheapest safe path from *start* to *goal* on *grid*.

    Cost model
    ----------
    - Cells where ``no_fly=True`` are completely impassable.
    - Move cost = ``cell.cost`` of the *destination* cell
      (1.0 by default, 0.8 for Commercial corridors).
    - Heuristic = Manhattan distance — admissible for 4-direction movement,
      so A* is guaranteed to return the optimal path.

    Returns a RouteResult dict (see module docstring).
    """
    size = len(grid)
    sr, sc = start
    gr, gc = goal

    # ---------------------------------------------------------------------- #
    # Trivial and early-failure cases
    # ---------------------------------------------------------------------- #
    if start == goal:
        return {"path": [start], "cost": 0.0, "success": True,
                "reason": "Already at goal"}

    if not (0 <= sr < size and 0 <= sc < size):
        return {"path": [], "cost": 0.0, "success": False,
                "reason": f"Start {start} is out of bounds"}
    if not (0 <= gr < size and 0 <= gc < size):
        return {"path": [], "cost": 0.0, "success": False,
                "reason": f"Goal {goal} is out of bounds"}
    if grid[sr][sc].no_fly:
        return {"path": [], "cost": 0.0, "success": False,
                "reason": f"Start cell {start} is no-fly"}
    if grid[gr][gc].no_fly:
        return {"path": [], "cost": 0.0, "success": False,
                "reason": f"Goal cell {goal} is no-fly"}

    # ---------------------------------------------------------------------- #
    # Priority queue: (f_cost, tiebreak_counter, coord)
    # The counter prevents Python from comparing Coord tuples when f-costs tie.
    # ---------------------------------------------------------------------- #
    counter = 0
    open_heap: list = []
    heapq.heappush(open_heap, (manhattan(start, goal), counter, start))
    counter += 1

    g_cost: Dict[Coord, float] = {start: 0.0}
    parent: Dict[Coord, Optional[Coord]] = {start: None}

    while open_heap:
        _f, _, current = heapq.heappop(open_heap)

        if current == goal:
            return _reconstruct(parent, goal, g_cost[goal])

        cur_g = g_cost[current]
        cr, cc = current

        for nr, nc in get_neighbors(cr, cc, size):
            nbr = (nr, nc)
            cell = grid[nr][nc]
            if cell.no_fly:
                continue

            new_g = cur_g + cell.cost          # 0.8 for Commercial, 1.0 otherwise
            if nbr not in g_cost or new_g < g_cost[nbr]:
                g_cost[nbr] = new_g
                parent[nbr] = current
                f = new_g + manhattan(nbr, goal)
                heapq.heappush(open_heap, (f, counter, nbr))
                counter += 1

    return {"path": [], "cost": 0.0, "success": False,
            "reason": f"No safe path from {start} to {goal}"}


def _reconstruct(
    parent: Dict[Coord, Optional[Coord]],
    goal: Coord,
    cost: float,
) -> dict:
    """Walk the parent chain backwards and return the RouteResult."""
    path: List[Coord] = []
    node: Optional[Coord] = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return {"path": path, "cost": cost, "success": True, "reason": "Path found"}


# --------------------------------------------------------------------------- #
# Composite delivery route: hub -> pickup -> dropoff -> hub
# --------------------------------------------------------------------------- #

def plan_delivery_route(delivery: Delivery, grid: List[List[Cell]]) -> dict:
    """Chain three A* calls to plan the complete delivery route.

    Segments
    --------
        hub → pickup → dropoff → hub

    Each segment is solved independently. Junction nodes are not duplicated
    in the merged path. If any segment fails the whole route is unroutable.

    Returns a RouteResult with the merged path and total cost.
    """
    waypoints = [
        (delivery.hub,     delivery.pickup,  "hub→pickup"),
        (delivery.pickup,  delivery.dropoff, "pickup→dropoff"),
        (delivery.dropoff, delivery.hub,     "dropoff→hub"),
    ]

    full_path: List[Coord] = []
    total_cost = 0.0

    for start, goal, label in waypoints:
        result = astar(start, goal, grid)
        if not result["success"]:
            return {
                "path": [],
                "cost": 0.0,
                "success": False,
                "reason": f"Segment [{label}] blocked — {result['reason']}",
            }
        # For the first segment keep all nodes; for subsequent segments skip
        # the first node (it duplicates the previous segment's goal).
        segment = result["path"] if not full_path else result["path"][1:]
        full_path.extend(segment)
        total_cost += result["cost"]

    return {
        "path": full_path,
        "cost": total_cost,
        "success": True,
        "reason": "Full route planned",
    }


# --------------------------------------------------------------------------- #
# Quick self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from grid_model import make_sample_grid

    grid = make_sample_grid()

    print("=== Basic A* ===")
    r = astar((0, 0), (9, 9), grid)
    print(f"(0,0)→(9,9): success={r['success']}, cost={r['cost']:.1f}, "
          f"path_len={len(r['path'])}")
    print(f"  path: {r['path']}")

    print("\n=== No-fly wall (col 5, rows 0-8) ===")
    for row in range(9):
        grid[row][5].no_fly = True
    r2 = astar((0, 0), (0, 9), grid)
    print(f"(0,0)→(0,9): success={r2['success']}, cost={r2['cost']:.1f}, "
          f"path_len={len(r2['path'])}")

    print("\n=== Composite route ===")
    grid3 = make_sample_grid()
    delivery = Delivery(
        id="TEST1",
        hub=(2, 2),
        pickup=(0, 0),
        dropoff=(9, 9),
        weight_kg=1.5,
        priority="normal",
    )
    r3 = plan_delivery_route(delivery, grid3)
    print(f"hub(2,2)→pickup(0,0)→dropoff(9,9)→hub(2,2): "
          f"success={r3['success']}, cost={r3['cost']:.1f}, "
          f"total_cells={len(r3['path'])}")
