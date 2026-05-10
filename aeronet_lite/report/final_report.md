# AeroNet Lite — Final Report

**Course:** BSDS Artificial Intelligence — Spring 2026, FAST
**Team:** Zaid, Hasaan, Rafay, Saad
**Project:** Autonomous drone delivery simulation on a 10x10 grid

---

## 1. Introduction

AeroNet Lite is a 20-step drone delivery simulator. A 10x10 grid models a small city. Each cell has a zone (Residential, Commercial, Hospital, School, Industrial, Open Field), a population density, and flags for hubs, charging pads, medical pickups, and no-fly status. The simulator validates the layout, selects a fleet under budget, plans hub → pickup → drop-off → hub routes, reacts to mid-simulation no-fly disruptions, forecasts demand, and detects drone anomalies.

The project demonstrates five AI techniques: Constraint Satisfaction, Genetic Algorithm, A* Search, real-time replanning, and supervised ML (regression + classification).

## 2. Architecture

A single shared grid model (`grid_model.py`) is the contract every module reads from. Each developer owned a vertical slice that reads from the grid and writes its own outputs through a defined interface, so the modules integrate with minimal coupling.

```
              ┌──────────────────┐
              │   grid_model.py  │  ← shared Cell, Drone, Delivery
              └────┬─────────────┘
                   │
   ┌───────────────┼───────────────┬────────────────┐
   │               │               │                │
┌──▼──────┐  ┌─────▼────┐   ┌──────▼────┐   ┌──────▼─────┐
│validator│  │ fleet/   │   │  astar /  │   │  ml_pipe + │
│  (CSP)  │  │delivery_ │   │delivery_  │   │ notebooks  │
│  Zaid   │  │generator │   │simulator  │   │  Saad      │
│         │  │  Rafay   │   │  Hasaan   │   │            │
└─────────┘  └──────────┘   └───────────┘   └──────┬─────┘
                                                   │
                                ┌──── demand_forecast.csv
                                └──── anomaly_predictions.csv
```

`main.py` (Zaid) orchestrates the 20-step simulation and integrates every module via the Day-1 contract.

## 3. Module 1 — CSP Layout Validation (Zaid)

Validates the layout against four constraints:

| Rule | Constraint |
| --- | --- |
| R1 | Industrial cells must not be 4-neighbor-adjacent to Schools or Hospitals |
| R2 | Every Residential cell must be within Manhattan 3 of a Drone Hub |
| R3 | Every Drone Hub must have a Charging Pad within Manhattan 2 |
| R4 | At least one Hospital must have a Medical Pickup within Manhattan 1 |

Each rule is implemented as a separate function returning a list of `Violation` objects with cell coordinates and a suggested fix. Violations are aggregated into a `ValidationReport` that prints pass/fail per rule and detailed messages.

The validator caught an early bug in the hand-authored sample grid where corner residential cells were 4 cells from the nearest hub — proof that the constraint checks do meaningful work.

## 4. Module 2 — Fleet Selection (Rafay)

Selects a fleet of light (cost 1000, payload 2 kg, range 12) and heavy (cost 1800, payload 5 kg, range 20) drones under a budget. Two algorithms are implemented:

- **Brute force**: enumerate every (light, heavy) combination within the budget and pick the highest score.
- **Genetic Algorithm**: chromosome `[light_count, heavy_count]`, tournament selection, single-point crossover, mutation rate 0.2, 30 individuals × 40 generations, elitism = 2.

Fitness combines geographic coverage (fraction of demand reachable from a hub within the fleet's range), a throughput factor (how many parallel drones the simulation needs), and the budget consumed:

```
score = 0.75 * coverage * throughput_factor − 0.25 * budget_used
```

For the sample grid with budget 8000, the GA converges to **4 drones**, matching the brute-force optimum.

## 5. Module 3 — A* Path Planning (Hasaan)

Each delivery requires three A* segments: hub → pickup, pickup → drop-off, drop-off → hub. The planner uses:

- **State**: grid coordinate `(row, col)`.
- **Actions**: 4-directional moves (up, down, left, right).
- **Cost**: 1.0 per move, 0.8 through Commercial corridor cells (favors using main thoroughfares).
- **Blocked nodes**: cells where `no_fly = True`.
- **Heuristic**: Manhattan distance — admissible for 4-direction grid movement.

The planner returns `{path, cost, success, reason}` per the Day-1 contract. A run from (0,0) to (9,9) finds a 19-cell path with cost 16.4, exploiting the cheap Commercial row 4.

## 6. Module 4 — Real-Time Disruption Handling (Hasaan)

When the simulation activates a no-fly cell mid-run (step 11 in the demo), every active drone's remaining route is checked. If the route passes through the new no-fly cell, A* is rerun from the drone's current position to the next target. Drones that cannot be rerouted are marked failed; deliveries are tagged delayed or failed accordingly.

The disruption handler is covered by 30 unit tests including edge cases such as a drone already standing on a newly-activated no-fly cell (forces escape to a safe neighbor).

## 7. Module 5 — Demand Forecasting & Anomaly Detection (Saad)

### 7.1 Demand forecasting (regression)
- Dataset: Bike Sharing Demand (Kaggle).
- Features: hour, day, temperature, weather, season.
- Models trained: Linear Regression and Random Forest Regressor.
- Metrics reported: MAE and RMSE.
- Output: `data/processed/demand_forecast.csv` with columns `row, col, predicted_demand` — consumed by both the fleet selector (drives coverage scoring) and the simulator (drives the step-15 forecast nudge).

### 7.2 Anomaly classification
- Dataset: synthetic, generated with rule-based labels per the spec — Battery (battery_drop > threshold), Route (route_deviation > threshold), Sensor (speed/altitude jump > threshold), Normal otherwise.
- Models trained: Decision Tree and Random Forest classifiers.
- Metrics reported: accuracy, confusion matrix, full classification report (precision/recall/F1 per class).
- Output: `data/processed/anomaly_predictions.csv` with columns `drone_id, step, label, confidence` — consumed by the simulator at step 18 to flag drones and at step 19 to force return-to-hub for any Battery anomalies.

## 8. Integration and 20-Step Scenario (Zaid)

`main.py` runs the spec scenario:

| Steps | Phase |
| --- | --- |
| 1–3 | Initialize grid, validate layout, load demand forecast, select fleet |
| 4–6 | Generate 5 deliveries, assign by nearest-drone, plan A* routes |
| 7–10 | Move drones along planned paths |
| 11 | Activate no-fly cell at (4, 7) |
| 12–14 | Detect routes crossing the no-fly cell and replan via A* |
| 15–17 | Demand forecast nudge, optionally inject an extra delivery for top-demand cell |
| 18 | Predict anomalies for every drone |
| 19 | React: force return-to-hub on Battery anomalies |
| 20 | Print summary: completed / delayed / failed counts |

`main.py` also handles graceful degradation: missing teammate modules fall back to inline stubs, so any single branch is runnable on its own. After full integration, the opening banner reads `astar=True fleet=True deliveries=True ml=True`.

## 9. Results from the Demo Run

A representative `python3 src/main.py` run on the sample grid:

- **Layout validation:** all 4 rules pass.
- **Demand forecast:** 100 cells loaded from real CSV, mean predicted demand ≈ 164.
- **Fleet selected:** 4 light drones (GA optimum, cost 4000 of 8000 budget).
- **Deliveries:** 5 generated, 4 assigned (one per drone).
- **Disruption:** no-fly cell at (4, 7) triggered 2 reroutes, all successful.
- **Anomalies:** 4 drones flagged at step 18 (3 Sensor, 1 Route).
- **Final summary:** 2 completed deliveries, 2 still in flight, 0 failed at step 20.

The remaining deliveries would complete with a longer simulation horizon — they are mid-route, not blocked.

## 10. Limitations and Future Work

- **Drone naming**: D1–D5 across all modules; multi-hub fleet rotation is round-robin rather than demand-aware.
- **Anomaly response**: only Battery anomalies trigger active recovery (return-to-hub). Route and Sensor anomalies are logged but not acted on.
- **Fleet optimization**: scoring uses a coarse throughput model; a more accurate model would simulate forward over the 20 steps to compute completed-deliveries-per-fleet.
- **Demand forecast**: regression is trained once and reused; a real deployment would retrain on rolling windows.
- **Layout authoring**: the grid is hand-authored; a proper CSP solver would generate satisfying layouts automatically.

## 11. Team Contributions

| Developer | Owned |
| --- | --- |
| **Zaid**   | Shared grid model, CSP layout validator (Module 1), zone-map visualization, simulation orchestrator (`main.py`), team plan, README, integration |
| **Hasaan** | A* path planner (Module 3), drone movement and disruption handling (Module 4), 30 unit tests |
| **Rafay**  | Fleet selection with brute force and Genetic Algorithm (Module 2), delivery generation and nearest-drone assignment |
| **Saad**   | Demand forecasting regression and anomaly classifier (Module 5), demand heatmap and anomaly timeline visualizations, both Jupyter notebooks |

## 12. Repository Layout

```
aeronet_lite/
├── data/processed/{demand_forecast.csv, anomaly_predictions.csv}
├── notebooks/{demand_forecasting.ipynb, anomaly_classifier.ipynb}
├── report/{figures/, final_report.md, viva_notes_zaid.md}
├── src/
│   ├── grid_model.py
│   ├── layout_validator.py
│   ├── fleet_selector.py
│   ├── delivery_generator.py
│   ├── astar_planner.py
│   ├── delivery_simulator.py
│   ├── ml_pipeline.py
│   ├── visualization.py
│   └── main.py
├── tests/test_astar_simulator.py
└── README.md
```

## 13. How to Run

```bash
cd aeronet_lite/src
python3 main.py            # full 20-step simulation
python3 layout_validator.py # CSP rules demo
python3 visualization.py   # save zone-map figure
```

ML notebooks are runnable from `aeronet_lite/notebooks/`; they regenerate the CSVs in `data/processed/`.
