"""Unit tests for Module 3 (A* planner) and Module 4 (Disruption handler).

Owner: Hasaan

Run from the project root:
    python -m pytest aeronet_lite/tests/test_astar_simulator.py -v
  or run directly:
    python aeronet_lite/tests/test_astar_simulator.py
"""

from __future__ import annotations

import os
import sys
import unittest

# Allow imports from src/ regardless of where the test is run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grid_model import (
    Cell, Coord, Delivery, Drone,
    make_empty_grid, make_sample_grid,
)
from astar_planner import astar, plan_delivery_route
from delivery_simulator import (
    BATTERY_DRAIN,
    activate_no_fly,
    assign_delivery,
    init_state,
    step_simulation,
)


# =========================================================================== #
# Helpers
# =========================================================================== #

def _drone(did="D1", dtype="light", home=(0, 0), pos=None) -> Drone:
    return Drone(id=did, type=dtype, home_hub=home,
                 position=pos if pos else home)


def _delivery(did="DEL1", hub=(0, 0), pickup=(0, 2),
              dropoff=(0, 4), kg=1.0) -> Delivery:
    return Delivery(id=did, hub=hub, pickup=pickup,
                    dropoff=dropoff, weight_kg=kg, priority="normal")


def _wall(grid, coords) -> None:
    """Mark a list of (row, col) coordinates as no-fly."""
    for r, c in coords:
        grid[r][c].no_fly = True


# =========================================================================== #
# A* Planner Tests
# =========================================================================== #

class TestAstar(unittest.TestCase):

    # ----------------------------------------------------------------------- #
    # Basic correctness
    # ----------------------------------------------------------------------- #

    def test_finds_path_on_empty_grid(self):
        """A* must find a path on a completely open grid."""
        grid = make_empty_grid()
        result = astar((0, 0), (9, 9), grid)
        self.assertTrue(result["success"])
        self.assertEqual(result["path"][0], (0, 0))
        self.assertEqual(result["path"][-1], (9, 9))
        self.assertGreater(result["cost"], 0)

    def test_path_length_matches_manhattan_on_open_grid(self):
        """On an open grid the shortest path has manhattan+1 nodes."""
        grid = make_empty_grid()
        result = astar((0, 0), (0, 5), grid)
        self.assertTrue(result["success"])
        # 5 moves → 6 nodes
        self.assertEqual(len(result["path"]), 6)

    def test_start_equals_goal(self):
        """When start == goal the path is just [start] with cost 0."""
        grid = make_empty_grid()
        result = astar((3, 3), (3, 3), grid)
        self.assertTrue(result["success"])
        self.assertEqual(result["path"], [(3, 3)])
        self.assertEqual(result["cost"], 0.0)

    def test_path_nodes_are_adjacent(self):
        """Every consecutive pair of cells in the path must be 4-adjacent."""
        grid = make_sample_grid()
        result = astar((0, 0), (9, 9), grid)
        self.assertTrue(result["success"])
        for (r1, c1), (r2, c2) in zip(result["path"], result["path"][1:]):
            self.assertEqual(abs(r1 - r2) + abs(c1 - c2), 1,
                             msg=f"Non-adjacent step: ({r1},{c1})→({r2},{c2})")

    # ----------------------------------------------------------------------- #
    # No-fly constraint
    # ----------------------------------------------------------------------- #

    def test_avoids_single_no_fly_cell(self):
        """The path must not pass through any no-fly cell."""
        grid = make_empty_grid()
        grid[0][1].no_fly = True           # block the direct right neighbour
        result = astar((0, 0), (0, 3), grid)
        self.assertTrue(result["success"])
        for cell in result["path"]:
            self.assertFalse(grid[cell[0]][cell[1]].no_fly,
                             msg=f"Path entered no-fly cell {cell}")

    def test_detours_around_vertical_wall(self):
        """A full vertical no-fly wall in the middle must be circumnavigated."""
        grid = make_empty_grid()
        _wall(grid, [(r, 5) for r in range(9)])   # col 5, rows 0-8 blocked
        result = astar((0, 0), (0, 9), grid)       # must go under or over
        self.assertTrue(result["success"])
        for cell in result["path"]:
            self.assertFalse(grid[cell[0]][cell[1]].no_fly)

    def test_completely_blocked_goal(self):
        """Goal surrounded by no-fly cells must return success=False."""
        grid = make_empty_grid()
        # (9,9) is a corner — only 2 neighbours
        _wall(grid, [(8, 9), (9, 8)])
        result = astar((0, 0), (9, 9), grid)
        self.assertFalse(result["success"])
        self.assertEqual(result["path"], [])

    def test_start_is_no_fly_returns_failure(self):
        grid = make_empty_grid()
        grid[0][0].no_fly = True
        result = astar((0, 0), (9, 9), grid)
        self.assertFalse(result["success"])

    def test_goal_is_no_fly_returns_failure(self):
        grid = make_empty_grid()
        grid[9][9].no_fly = True
        result = astar((0, 0), (9, 9), grid)
        self.assertFalse(result["success"])

    def test_out_of_bounds_start(self):
        grid = make_empty_grid()
        result = astar((-1, 0), (9, 9), grid)
        self.assertFalse(result["success"])

    def test_out_of_bounds_goal(self):
        grid = make_empty_grid()
        result = astar((0, 0), (10, 10), grid)
        self.assertFalse(result["success"])

    # ----------------------------------------------------------------------- #
    # Cost model
    # ----------------------------------------------------------------------- #

    def test_commercial_corridor_costs_less(self):
        """A path through Commercial cells (cost=0.8) must be cheaper
        than the same-length path through normal cells (cost=1.0)."""
        grid = make_empty_grid()
        # Top row: normal cost (1.0 each)
        # Row 1: commercial corridor (0.8 each)
        for c in range(10):
            grid[1][c].zone = "Commercial"
            grid[1][c].cost = 0.8

        # Path forced to row 0 (top row only)
        grid_top_only = make_empty_grid()
        _wall(grid_top_only, [(1, c) for c in range(10)])
        r_top = astar((0, 0), (0, 5), grid_top_only)

        # Path using the commercial corridor
        r_comm = astar((0, 0), (0, 5), grid)

        self.assertTrue(r_top["success"])
        self.assertTrue(r_comm["success"])
        # Commercial corridor path must not cost more
        self.assertLessEqual(r_comm["cost"], r_top["cost"])

    def test_optimal_cost_straight_line(self):
        """Straight line of 5 moves on an open grid costs exactly 5.0."""
        grid = make_empty_grid()
        result = astar((0, 0), (0, 5), grid)
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["cost"], 5.0, places=5)

    # ----------------------------------------------------------------------- #
    # Composite route
    # ----------------------------------------------------------------------- #

    def test_plan_delivery_route_success(self):
        """hub→pickup→dropoff→hub composite route must succeed on open grid."""
        grid = make_empty_grid()
        delivery = _delivery(hub=(0, 0), pickup=(2, 0), dropoff=(2, 5))
        result = plan_delivery_route(delivery, grid)
        self.assertTrue(result["success"])
        self.assertEqual(result["path"][0], (0, 0))   # starts at hub
        self.assertEqual(result["path"][-1], (0, 0))  # ends at hub
        self.assertIn((2, 0), result["path"])          # visits pickup
        self.assertIn((2, 5), result["path"])          # visits dropoff

    def test_plan_delivery_route_blocked_segment(self):
        """If any segment is blocked the whole route must fail."""
        grid = make_empty_grid()
        # Seal off the pickup cell (2,2) completely
        _wall(grid, [(1, 2), (3, 2), (2, 1), (2, 3)])
        delivery = _delivery(hub=(0, 0), pickup=(2, 2), dropoff=(5, 5))
        result = plan_delivery_route(delivery, grid)
        self.assertFalse(result["success"])
        self.assertEqual(result["path"], [])

    def test_plan_delivery_route_visits_all_waypoints(self):
        """The merged path must visit hub, pickup, and dropoff.
        Note: a waypoint may appear more than once (e.g. pickup is visited
        on the outward leg AND the return-to-hub leg passes through it)."""
        grid = make_empty_grid()
        delivery = _delivery(hub=(0, 0), pickup=(2, 0), dropoff=(4, 0))
        result = plan_delivery_route(delivery, grid)
        self.assertTrue(result["success"])
        self.assertEqual(result["path"][0], (0, 0),  msg="Must start at hub")
        self.assertEqual(result["path"][-1], (0, 0), msg="Must end at hub")
        self.assertIn((2, 0), result["path"], msg="Must visit pickup")
        self.assertIn((4, 0), result["path"], msg="Must visit dropoff")


# =========================================================================== #
# Delivery Simulator Tests
# =========================================================================== #

class TestSimulator(unittest.TestCase):

    # ----------------------------------------------------------------------- #
    # Drone movement & battery
    # ----------------------------------------------------------------------- #

    def test_drone_moves_one_cell_per_step(self):
        """After one step the drone must advance exactly one cell."""
        grid = make_empty_grid()
        drone = _drone("D1", "light", home=(0, 0), pos=(0, 0))
        delivery = _delivery(hub=(0, 0), pickup=(0, 5), dropoff=(0, 9))
        state = init_state(grid, [drone], [delivery])
        state = assign_delivery(state, delivery, drone)

        pos_before = state["drones"]["D1"].position
        state = step_simulation(state)
        pos_after = state["drones"]["D1"].position

        dr = abs(pos_after[0] - pos_before[0])
        dc = abs(pos_after[1] - pos_before[1])
        self.assertEqual(dr + dc, 1, msg="Drone must move exactly one cell")

    def test_battery_drain_light_drone(self):
        """Light drone must lose ~8.33 battery per cell moved."""
        grid = make_empty_grid()
        drone = _drone("D1", "light", home=(0, 0), pos=(0, 0))
        delivery = _delivery(hub=(0, 0), pickup=(0, 5), dropoff=(0, 9))
        state = init_state(grid, [drone], [delivery])
        state = assign_delivery(state, delivery, drone)

        before = state["drones"]["D1"].battery
        state = step_simulation(state)
        after = state["drones"]["D1"].battery

        expected_drain = BATTERY_DRAIN["light"]
        self.assertAlmostEqual(before - after, expected_drain, places=5)

    def test_battery_drain_heavy_drone(self):
        """Heavy drone must lose exactly 5.0 battery per cell moved."""
        grid = make_empty_grid()
        drone = _drone("D1", "heavy", home=(0, 0), pos=(0, 0))
        delivery = _delivery(hub=(0, 0), pickup=(0, 5), dropoff=(0, 9))
        state = init_state(grid, [drone], [delivery])
        state = assign_delivery(state, delivery, drone)

        before = state["drones"]["D1"].battery
        state = step_simulation(state)
        after = state["drones"]["D1"].battery

        self.assertAlmostEqual(before - after, BATTERY_DRAIN["heavy"], places=5)

    def test_idle_drone_not_moved(self):
        """An idle drone must remain at its position across steps."""
        grid = make_empty_grid()
        drone = _drone("D1", pos=(3, 3))
        state = init_state(grid, [drone])
        state = step_simulation(state)
        self.assertEqual(state["drones"]["D1"].position, (3, 3))
        self.assertEqual(state["drones"]["D1"].status, "idle")

    # ----------------------------------------------------------------------- #
    # Delivery lifecycle
    # ----------------------------------------------------------------------- #

    def test_delivery_assigned_correctly(self):
        """After assign_delivery, drone status must be 'delivering'
        and it must have a non-empty route."""
        grid = make_empty_grid()
        drone = _drone("D1", pos=(0, 0))
        delivery = _delivery(hub=(0, 0), pickup=(0, 3), dropoff=(0, 6))
        state = init_state(grid, [drone], [delivery])
        state = assign_delivery(state, delivery, drone)
        d = state["drones"]["D1"]
        self.assertEqual(d.status, "delivering")
        self.assertGreater(len(d.current_route), 0)

    def test_delivery_completes_and_drone_goes_idle(self):
        """After the route is exhausted the delivery must be completed
        and the drone status must return to 'idle'."""
        grid = make_empty_grid()
        # Very short delivery: hub, pickup, dropoff all close together
        drone = _drone("D1", "heavy", home=(0, 0), pos=(0, 0))
        delivery = _delivery(hub=(0, 0), pickup=(0, 1), dropoff=(0, 2))
        state = init_state(grid, [drone], [delivery])
        state = assign_delivery(state, delivery, drone)

        # Run until idle or 50 steps (safety cap)
        for _ in range(50):
            state = step_simulation(state)
            if state["drones"]["D1"].status == "idle":
                break

        self.assertEqual(state["drones"]["D1"].status, "idle")
        self.assertIn("DEL1", state["completed"])

    def test_unroutable_delivery_is_delayed(self):
        """assign_delivery must delay a delivery that has no valid route."""
        grid = make_empty_grid()
        # Seal off pickup cell (5,5) completely
        _wall(grid, [(4, 5), (6, 5), (5, 4), (5, 6)])
        drone = _drone("D1", pos=(0, 0))
        delivery = _delivery(hub=(0, 0), pickup=(5, 5), dropoff=(9, 9))
        state = init_state(grid, [drone], [delivery])
        state = assign_delivery(state, delivery, drone)
        self.assertIn("DEL1", state["delayed"])
        self.assertEqual(state["drones"]["D1"].status, "idle")

    # ----------------------------------------------------------------------- #
    # Disruption handler
    # ----------------------------------------------------------------------- #

    def test_no_fly_cell_is_marked_on_grid(self):
        """activate_no_fly must update the grid cell flag."""
        grid = make_empty_grid()
        drone = _drone("D1", pos=(0, 0))
        state = init_state(grid, [drone])
        state = activate_no_fly(state, 4, 4)
        self.assertTrue(state["grid"][4][4].no_fly)

    def test_disruption_reroutes_drone_with_blocked_route(self):
        """When a no-fly cell is activated mid-route the drone must be
        rerouted and its original blocked path must no longer appear."""
        grid = make_empty_grid()
        # Long straight route along row 0: (0,0)→(0,9)
        drone = _drone("D1", "heavy", home=(0, 0), pos=(0, 0))
        delivery = _delivery(hub=(0, 0), pickup=(0, 5), dropoff=(0, 9))
        state = init_state(grid, [drone], [delivery])
        state = assign_delivery(state, delivery, drone)

        # Block a cell that is in the route
        blocked = (0, 3)
        state = activate_no_fly(state, *blocked)

        d = state["drones"]["D1"]
        self.assertNotIn(blocked, d.current_route,
                         msg="Blocked cell must not remain in route after reroute")
        self.assertNotEqual(d.status, "failed",
                            msg="Drone must be rerouted, not failed")

    def test_unaffected_drone_not_rerouted(self):
        """A drone whose route does not cross the blocked cell must keep
        its original route length unchanged."""
        grid = make_empty_grid()
        # Drone travels along col 0 downward; blocked cell is at (5,9) (far away)
        drone = _drone("D1", "heavy", home=(0, 0), pos=(0, 0))
        delivery = _delivery(hub=(0, 0), pickup=(3, 0), dropoff=(6, 0))
        state = init_state(grid, [drone], [delivery])
        state = assign_delivery(state, delivery, drone)

        route_before = list(state["drones"]["D1"].current_route)
        state = activate_no_fly(state, 5, 9)   # far away from the route
        route_after = state["drones"]["D1"].current_route

        self.assertEqual(route_before, route_after,
                         msg="Route must be unchanged when no-fly is irrelevant")

    def test_drone_on_nofly_cell_is_forced_home(self):
        """A drone sitting on the newly activated no-fly cell must have
        its delivery delayed and be rerouted toward home."""
        grid = make_empty_grid()
        # Place drone at (5,5), home at (0,0)
        drone = _drone("D1", "heavy", home=(0, 0), pos=(5, 5))
        delivery = _delivery(hub=(0, 0), pickup=(5, 7), dropoff=(9, 9))

        # Manually put the drone mid-delivery at (5,5)
        drone.status = "delivering"
        drone.current_route = [(5, 6), (5, 7), (5, 8), (5, 9)]

        state = init_state(grid, [drone])
        state["assignments"]["D1"] = delivery
        state["waypoints"]["D1"] = {
            "pickup": (5, 7), "dropoff": (9, 9),
            "hub": (0, 0), "phase": "to_pickup",
        }

        state = activate_no_fly(state, 5, 5)   # drone is ON this cell

        d = state["drones"]["D1"]
        # Delivery must be delayed (not completed, not in assignments)
        self.assertIn("DEL1", state["delayed"])
        self.assertNotIn("D1", state["assignments"])
        # Drone must be rerouting home (status = "returning"), not failed
        self.assertEqual(d.status, "returning",
                         msg="Drone should be returning to home hub, not failed")

    def test_no_path_after_disruption_fails_drone(self):
        """If rerouting after disruption is impossible the drone must be
        marked failed and the delivery must move to failed_deliveries."""
        grid = make_empty_grid()
        # Drone at (5,5), completely surrounded by no-fly on all 4 sides
        drone = _drone("D1", "heavy", home=(0, 0), pos=(5, 5))
        delivery = _delivery(hub=(0, 0), pickup=(5, 7), dropoff=(9, 9))

        drone.status = "delivering"
        drone.current_route = [(5, 6), (5, 7)]

        state = init_state(grid, [drone])
        state["assignments"]["D1"] = delivery
        state["waypoints"]["D1"] = {
            "pickup": (5, 7), "dropoff": (9, 9),
            "hub": (0, 0), "phase": "to_pickup",
        }

        # Seal all 4 neighbors of (5,5) — drone is trapped
        _wall(grid, [(4, 5), (6, 5), (5, 4), (5, 6)])

        # Activate no-fly on (5,6) which is already in the route
        state = activate_no_fly(state, 5, 6)

        self.assertEqual(state["drones"]["D1"].status, "failed")
        self.assertIn("DEL1", state["failed_deliveries"])

    def test_step_counter_increments(self):
        """state['step'] must increase by exactly 1 per step_simulation call."""
        grid = make_empty_grid()
        drone = _drone("D1", pos=(0, 0))
        state = init_state(grid, [drone])
        self.assertEqual(state["step"], 0)
        state = step_simulation(state)
        self.assertEqual(state["step"], 1)
        state = step_simulation(state)
        self.assertEqual(state["step"], 2)

    def test_event_log_records_disruption(self):
        """activate_no_fly must add an entry to state['log']."""
        grid = make_empty_grid()
        drone = _drone("D1", pos=(0, 0))
        state = init_state(grid, [drone])
        log_len_before = len(state["log"])
        state = activate_no_fly(state, 3, 7)
        self.assertGreater(len(state["log"]), log_len_before)
        # At least one log entry should mention the coordinates
        combined = " ".join(state["log"])
        self.assertIn("3", combined)
        self.assertIn("7", combined)


# =========================================================================== #
# Runner
# =========================================================================== #

if __name__ == "__main__":
    unittest.main(verbosity=2)
