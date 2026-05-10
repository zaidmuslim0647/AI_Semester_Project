# Viva Notes — Zaid

Modules owned: shared grid model, **Module 1 (CSP layout validator)**, base visualization, simulation orchestrator (`main.py`).

---

## Why CSP for Module 1

- A city layout is a set of variables (cell zones, hub locations, charging pads, medical pickups) with constraints between them.
- Rules R1–R4 are exactly **unary and binary constraints** over those variables.
- We're not searching for a satisfying assignment — students hand-author the layout — but the **constraint-checking phase** is the standard CSP consistency check. Each rule is implemented as a function that returns the cells violating it.
- Trade-off: we don't run AC-3 or backtracking because the layout is fixed by hand. The interesting AI work is in writing the constraints clearly and reporting violations with cell coordinates and suggested fixes.

## The four rules and their implementations

| Rule | Constraint | Type | Implementation |
| --- | --- | --- | --- |
| R1 | Industrial cells cannot be 4-neighbor-adjacent to Schools or Hospitals | Binary, local | For each Industrial cell, check the four orthogonal neighbors. |
| R2 | Every Residential cell must be within Manhattan 3 of a Drone Hub | Binary, global | For each Residential cell, take min Manhattan distance over all hubs. |
| R3 | Every Drone Hub must have a Charging Pad within Manhattan 2 | Binary, global | For each hub, take min Manhattan distance over all charging pads. |
| R4 | At least one Hospital must have a Medical Pickup within Manhattan 1 | Existential | Pass if any hospital satisfies; otherwise report against the first hospital. |

Why Manhattan and not Euclidean: drones move in 4 directions on a grid, so Manhattan is the natural admissible metric.

## Sample grid design

`make_sample_grid()` is a hand-crafted 10x10 grid that:

- Has two hubs at (2,2) and (7,7) so the residential clusters within Manhattan 3 cover most of the map.
- Has charging pads at (2,3) and (7,6), one per hub, both within distance 1.
- Puts the hospital at (1,8) with the medical pickup at (1,9), distance 1 — satisfies R4.
- Keeps the school at (8,1) far from the industrial cells at (5,0) and (6,0) so R1 holds.
- Has a commercial corridor along row 4 with `cost=0.8`, used by Hasaan's A* as a cheap thoroughfare.

The validator flagged corner cells the first time we ran it — that's what caught the original residential cluster being two cells too wide. Useful proof that the validator does what it claims.

## Likely viva questions

**Q: Why split rules into separate functions instead of one big checker?**
Each rule has its own complexity (R1 is local 4-neighbor; R4 is existential over hospitals). Separating them keeps each function small and lets the report show pass/fail per rule.

**Q: How would you scale this to a 100x100 grid?**
R1 is O(N) since each cell has 4 neighbors. R2/R3 are O(N × H) where H is the number of hubs/pads — fine when H is small. For very large grids, precompute a BFS distance map from each hub once, then look up in O(1) per residential cell.

**Q: What happens if the grid has no hubs at all?**
R2 reports every Residential cell as a violation with the suggestion to add a hub. R3 has nothing to check. R4 is independent of hubs.

**Q: What's a `Violation`?**
A dataclass holding `rule_id`, the offending cell coordinate, a human-readable message, and a suggested fix. The suggestion is what makes the report useful during viva — it's not just "fail," it tells the user what to do.

**Q: Is this really a CSP if you're not searching?**
The constraint **language and checking** are CSP. Solver-style search would apply if we were generating the layout automatically. The spec explicitly says students may hand-author the grid and run a validator that "behaves like a CSP constraint checker," which is what we did.

## Simulation orchestration (main.py)

- 20 steps mirror the spec's scenario table.
- `try/except` import pattern: if a teammate's module is missing, an inline stub takes over. Banner at the top tells the grader which modules are real vs stubbed.
- Inline stubs follow the fallback table in `PROJECT_PLAN.md`: straight-line A*, hardcoded fleet, random demand, "Normal" anomalies (with a deliberate Battery anomaly on D3 at step 18 so step 19 has something to react to).
- Routes are computed segment-by-segment (hub → pickup, pickup → dropoff, dropoff → hub) and concatenated.
- The disruption handler (steps 12–14) detects when the no-fly cell is on a remaining route and triggers replanning from the drone's current position.

## What I would extend with more time

- Generate the layout automatically from random seeds and use real CSP backtracking to repair violations.
- Add a proper drone-energy model that affects A* cost.
- Replace `try/except` import stubs with a registry pattern so swapping implementations is more explicit.
