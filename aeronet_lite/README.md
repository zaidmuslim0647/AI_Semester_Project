# AeroNet Lite

Autonomous drone delivery simulation on a 10x10 grid. BSDS AI semester project, Spring 2026, FAST.

## What it does

A 20-step simulation that:

1. Validates a city layout against four CSP rules.
2. Selects a small drone fleet under a budget.
3. Plans hub → pickup → dropoff → hub routes with A*.
4. Activates a no-fly cell mid-simulation and replans affected drones.
5. Runs a regression model to forecast demand and a classifier to detect flight anomalies.

## Folder structure

```
aeronet_lite/
  data/
    raw/                       # Kaggle datasets (Saad)
    processed/
      demand_forecast.csv      # produced by Saad, consumed by Rafay
      anomaly_predictions.csv  # produced by Saad, consumed by Hasaan
  src/
    grid_model.py              # shared contract: Cell, Drone, Delivery
    layout_validator.py        # Module 1: CSP rules R1-R4
    fleet_selector.py          # Module 2: heuristic / GA fleet selection
    delivery_generator.py      # delivery + drone-to-delivery assignment
    astar_planner.py           # Module 3: A* path planning
    delivery_simulator.py      # Module 4: drone movement + reroute on disruption
    ml_pipeline.py             # Module 5: demand regression + anomaly classifier
    visualization.py           # zone map, route overlay, demand heatmap
    main.py                    # 20-step simulation orchestrator
  notebooks/
    demand_forecasting.ipynb
    anomaly_classifier.ipynb
  report/
    figures/
    final_report.docx
  README.md
```

See `../PROJECT_PLAN.md` for the per-developer split, day-by-day timeline, and Day-1 contract.

## Running the project

Requirements: Python 3.10+, `matplotlib`, `numpy`. ML notebooks add `pandas` and `scikit-learn`.

```
cd src/

# print the sample grid in the console
python3 grid_model.py

# generate the zone-map figure (saved to report/figures/zone_map.png)
python3 visualization.py

# run the layout validator on a clean grid and a deliberately broken one
python3 layout_validator.py

# run the full 20-step simulation
python3 main.py
```

`main.py` imports each teammate's module if available and falls back to inline stubs otherwise, so it runs end-to-end on any single developer's branch.

## Module status (after merging all branches, this should all read REAL)

| Module | Owner | File | Status flag in main.py |
| --- | --- | --- | --- |
| Shared grid model | Zaid | `grid_model.py` | always real |
| CSP layout validator | Zaid | `layout_validator.py` | always real |
| Zone-map visualization | Zaid | `visualization.py` | always real |
| Simulation orchestrator | Zaid | `main.py` | always real |
| A* path planner | Hasaan | `astar_planner.py` | `HAS_REAL_ASTAR` |
| Disruption / movement | Hasaan | `delivery_simulator.py` | (called via main) |
| Fleet selector | Rafay | `fleet_selector.py` | `HAS_REAL_FLEET` |
| Delivery generator | Rafay | `delivery_generator.py` | `HAS_REAL_DELIVERY_GEN` |
| Demand regression | Saad | `ml_pipeline.py` | `HAS_REAL_ML` |
| Anomaly classifier | Saad | `ml_pipeline.py` | `HAS_REAL_ML` |

The first banner printed by `main.py` shows which real modules were loaded.

## Day-1 shared contract (locked)

Every developer reads from this contract. See `PROJECT_PLAN.md` section 3 for full details.

- `Cell` dataclass — row, col, zone, density, is_hub, is_charging, is_medical_pickup, no_fly, demand, cost.
- `Drone` dataclass — id, type, home_hub, position, battery, payload_kg, current_route, status.
- `Delivery` namedtuple — id, hub, pickup, dropoff, weight_kg, priority.
- `astar(start, goal, grid)` returns `{"path": [...], "cost": float, "success": bool, "reason": str}`.
- `select_fleet(grid, budget)` returns `list[Drone]`.
- `generate_deliveries(grid, n)` returns `list[Delivery]`.
- `load_demand_forecast(grid)` returns `dict[(r,c) -> float]`.
- `predict_anomaly(drone, step)` returns one of `Normal | Battery | Route | Sensor`.

CSV artifacts (Saad → others):

- `data/processed/demand_forecast.csv` — columns: `row, col, predicted_demand`.
- `data/processed/anomaly_predictions.csv` — columns: `drone_id, step, label`.

## Team

| Dev | Owns |
| --- | --- |
| Zaid    | Grid model, CSP validator, visualization, main orchestrator |
| Hasaan  | A* planner, drone movement and disruption handling |
| Rafay   | Fleet selection (brute force + GA), delivery generation, assignment |
| Saad    | ML pipeline (regression + classification), heatmap and anomaly views |
