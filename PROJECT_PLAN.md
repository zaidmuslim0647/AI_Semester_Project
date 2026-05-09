# AeroNet Lite — Team Plan

**Project:** Autonomous drone delivery simulation on a 10x10 grid (CSP layout validation, fleet selection, A* routing, real-time replanning, ML for demand & anomaly detection).
**Team:** Zaid, Hasaan, Rafay, Saad
**Deadline:** 14 days
**Source spec:** `BSDS-SemesterProject-AI-SP2026.docx.md`

---

## 1. Core Principle

**One shared contract. Four independent vertical slices.**
The only cross-cutting dependency is the grid data structure. We freeze it on **Day 1**. After that, every developer works inside their own files and reads/writes through clearly defined interfaces (function signatures + CSV artifacts). Nobody is blocked waiting for another teammate's code.

---

## 2. High-Level Split

| Dev | Modules owned | Files owned | Depends on |
| --- | --- | --- | --- |
| **Zaid** | Shared grid model + Module 1 (CSP Layout Validator) + base visualization + simulation skeleton | `src/grid_model.py`, `src/layout_validator.py`, `src/visualization.py` (zone map), `src/main.py` | Nothing. Others depend on him. |
| **Hasaan** | Module 3 (A* Path Planner) + Module 4 (Disruption Handler) + drone movement loop | `src/astar_planner.py`, `src/delivery_simulator.py` | `grid_model.py` (Zaid, Day 1) |
| **Rafay** | Module 2 (Fleet Selector) + delivery generation + drone-to-delivery assignment | `src/fleet_selector.py`, `src/delivery_generator.py` | `grid_model.py` (Zaid) + optional `demand_forecast.csv` (Saad) — stubs with random demand if Saad is late |
| **Saad** | Module 5 (ML Pipeline: regression + classification) + heatmap & anomaly visualization | `notebooks/demand_forecasting.ipynb`, `notebooks/anomaly_classifier.ipynb`, `src/ml_pipeline.py`, `src/visualization.py` (heatmap + anomaly view extensions) | Nothing. Pure notebooks against Kaggle data. Outputs CSVs. |

### Why this split has the lowest possible coupling
- **Modules 3 & 4 are paired under Hasaan** because the Disruption Handler calls A* repeatedly — splitting them would create the project's worst cross-dependency.
- **Saad is fully isolated** — notebooks run on Kaggle data and export CSV artifacts. Zero blocking on the rest of the team.
- **Rafay ↔ Saad** interaction is a one-way CSV file. Rafay can ship with random/stub demand on Day 6 even if Saad's regression isn't ready until Day 11.
- **Hasaan ↔ Rafay** never need to talk after Day 1 — they agree on the `delivery` tuple shape and never block each other.
- **Zaid is the only person others depend on**, and his Day 1–2 work (the grid contract) is the simplest deliverable. He then works in parallel on the validator and visualization.

---

## 3. Day-1 Shared Contract (Frozen by all four)

This must be locked in a 30-minute meeting on Day 1. Once frozen, no changes without team agreement.

### 3.1 Grid cell schema (`grid_model.py`)
Use a `dataclass` named `Cell`:

```python
@dataclass
class Cell:
    row: int
    col: int
    zone: str            # "Residential" | "Commercial" | "Hospital" | "School" | "Industrial" | "OpenField"
    density: int         # population proxy, used for demand
    is_hub: bool
    is_charging: bool
    is_medical_pickup: bool
    no_fly: bool
    demand: float        # set by Rafay/Saad later
    cost: float          # 1.0 default, 0.8 if Commercial corridor
```

The grid itself is `grid: list[list[Cell]]` (10x10).

### 3.2 Delivery tuple
```python
Delivery = namedtuple("Delivery", ["id", "hub", "pickup", "dropoff", "weight_kg", "priority"])
# hub, pickup, dropoff are (row, col) tuples
```

### 3.3 Route format
A* returns:
```python
{"path": [(r,c), (r,c), ...], "cost": float, "success": bool, "reason": str}
```

### 3.4 CSV artifacts (Saad → others)
- `data/processed/demand_forecast.csv` — columns: `row, col, predicted_demand`
- `data/processed/anomaly_predictions.csv` — columns: `drone_id, step, label` (label ∈ Normal | Battery | Route | Sensor)

### 3.5 Drone object
```python
@dataclass
class Drone:
    id: str            # "D1", "D2", ...
    type: str          # "light" | "heavy"
    home_hub: tuple
    position: tuple
    battery: float     # 0..100
    payload_kg: float
    current_route: list  # list of (r,c)
    status: str        # "idle" | "delivering" | "returning" | "failed"
```

---

## 4. Per-Developer Detailed Plan

### 4.1 Zaid — Foundation & Layout Validator
**Modules:** Shared grid model, Module 1 (CSP), base visualization, `main.py` orchestrator.

| Day | Task | Deliverable |
| --- | --- | --- |
| 1 | Lock the contract (Section 3) with the team. Implement `Cell`, `Drone`, sample 10x10 grid generator. | `grid_model.py` + sample grid printable to console |
| 2 | Build zone-map visualization (matplotlib colored grid). | `visualization.py` → `plot_zone_map(grid)` |
| 3 | Implement R1 (industrial adjacency) and R2 (residential within 3 of hub). Helpers: `get_neighbors`, `manhattan`. | `layout_validator.py` partial |
| 4 | Implement R3 (hub ↔ charging within 2) and R4 (hospital ↔ medical pickup within 1). Add suggestion text. | Full validator with report |
| 5–6 | Build `main.py` skeleton: loop over 20 simulation steps, event log printer, hooks for fleet/routing/ML modules (use stubs). | Runnable `main.py` printing 20-step log against stubs |
| 7 | **Sync point:** integrate Hasaan's A* + Rafay's fleet selector. | First end-to-end smoke run |
| 8–13 | Polish report formatting, write README, prepare viva talking points for CSP module. | README.md, viva notes |
| 14 | Final demo rehearsal. | — |

**Reads from teammates:** Hasaan's `astar_planner.astar()`, Rafay's `fleet_selector.select_fleet()`, Saad's CSV files.
**Writes for teammates:** the grid + sample data; visualization API.
**Definition of done:** validator catches all 4 rules with clear cell-coordinate error messages; `main.py` runs all 20 steps end-to-end.

---

### 4.2 Hasaan — Path Planning & Disruption
**Modules:** Module 3 (A*), Module 4 (Disruption), drone movement engine.

| Day | Task | Deliverable |
| --- | --- | --- |
| 1 | Attend contract meeting. Stub `astar()` returning a dummy straight-line path so Zaid's `main.py` can compile. | Function signature locked |
| 2–6 | Implement A*: priority queue, parent pointers, `cost` field per cell, `no_fly` blocking, Manhattan heuristic. Test on 10x10 with hand-crafted no-fly walls. | `astar_planner.py` |
| 7 | Hub → pickup → dropoff → hub composite routing. **Sync with Zaid** for integration. | Full route for any delivery |
| 8 | Drone movement loop: each step advances drones one cell along their `current_route`. | `delivery_simulator.py` skeleton |
| 9 | Implement disruption handler: `activate_no_fly(grid, r, c)` → detect affected drones → call A* from current position. Mark `failed` if no path. | Reroute event log |
| 10 | Edge cases: drone already on a no-fly cell (force return to hub), overlapping reroutes. | Robust simulator |
| 11–13 | Optimize, write unit tests on synthetic grids, prepare viva notes for A* and replanning. | Tests + notes |
| 14 | Demo rehearsal. | — |

**Reads:** `grid_model.py` (Zaid).
**Writes:** `astar(start, goal, grid)` and `step_simulation(state)` for Zaid's main loop.
**Definition of done:** A* works on grids with arbitrary no-fly patterns; disruption at step 11 produces a clean reroute log entry.

---

### 4.3 Rafay — Fleet Selection & Delivery Generation
**Modules:** Module 2 (Fleet Selector), delivery generation, drone-to-delivery assignment.

| Day | Task | Deliverable |
| --- | --- | --- |
| 1 | Attend contract meeting. Agree on `Delivery` and `Drone` shapes. | Shapes locked |
| 2 | Define drone catalogue (Light: 1000 cost / 2 kg / 12 cells; Heavy: 1800 / 5 kg / 20). Implement scoring: `0.75*coverage − 0.25*budget_used`. | `fleet_selector.py` skeleton |
| 3–5 | Implement brute-force search over light/heavy combinations under budget. **Then** implement Genetic Algorithm: `[light_count, heavy_count]` chromosome, mutation, crossover, tournament selection. | Two-mode selector with toggle |
| 6 | Demand stub (uniform random) so the selector runs without Saad. Generate 5–10 deliveries with hubs/pickups/dropoffs. | `delivery_generator.py` |
| 7 | Nearest-drone assignment for each delivery. **Sync point** with Zaid + Hasaan. | Full assignment table |
| 8–10 | Hook into Saad's `demand_forecast.csv` once available. Re-run selector with real predicted demand. | Improved fleet output |
| 11–13 | Tune GA hyperparameters, prepare viva notes. | — |
| 14 | Demo rehearsal. | — |

**Reads:** `grid_model.py` (Zaid) + `data/processed/demand_forecast.csv` (Saad, optional).
**Writes:** `select_fleet(grid, budget)` returning `list[Drone]`; `generate_deliveries(grid, n)` returning `list[Delivery]`; `assign(deliveries, drones)`.
**Definition of done:** under a budget cap, returns a fleet with score ≥ baseline; GA outperforms or matches brute force; deliveries assigned to nearest available drone.

---

### 4.4 Saad — ML Pipeline & Heatmap Visualization
**Modules:** Module 5 (Regression + Classification), demand heatmap and anomaly view in visualization.

| Day | Task | Deliverable |
| --- | --- | --- |
| 1 | Attend contract meeting. Agree on CSV column names. | CSV schema locked |
| 2–3 | Download Bike Sharing Demand dataset. EDA notebook. | `notebooks/demand_forecasting.ipynb` skeleton |
| 4–6 | Train Linear Regression and Random Forest Regressor. Compare MAE / RMSE. Pick best. Map predictions onto 10x10 cells (use hour, weather, density features). | Trained model + `demand_forecast.csv` |
| 7 | Build `plot_demand_heatmap(grid)` matplotlib function. | Heatmap view |
| 8–9 | Build synthetic anomaly dataset using rules from spec (battery_drop high → Battery; route_deviation high → Route; speed/altitude jump → Sensor). | `notebooks/anomaly_classifier.ipynb` |
| 10–11 | Train Decision Tree + Random Forest classifiers. Confusion matrix + accuracy. Optional: KNN comparison. | Classifier + `anomaly_predictions.csv` |
| 12 | Build anomaly view (table or printed log of drone events with predicted labels). | Anomaly visualization |
| 13 | Write up ML section of the report (assumptions, metrics, limitations). | Report draft section |
| 14 | Demo rehearsal. | — |

**Reads:** Kaggle datasets + (optional) flight log produced by Hasaan's simulator.
**Writes:** two CSVs + heatmap/anomaly visualization functions.
**Definition of done:** regression with reported MAE/RMSE; classifier with reported accuracy + confusion matrix; both CSVs produced and consumed by the simulator.

---

## 5. Integration Timeline & Sync Points

| Day | Event | Who attends |
| --- | --- | --- |
| **Day 1** | **Contract freeze meeting (30 min).** Lock Section 3 of this document. Everyone commits stubs the same day so the project compiles end-to-end from Day 2. | All 4 |
| **Day 7** | **Mid-integration smoke test (1 hour).** Run `main.py` end-to-end with real validator + real A* + real fleet selector + stub ML. Identify any contract violations and fix that day. | All 4 |
| **Day 12** | **ML hookup.** Saad's CSVs land in `data/processed/`. Rafay re-runs fleet selector with real demand. | Rafay + Saad |
| **Day 14** | **Final demo rehearsal.** Full 20-step simulation with all real modules. | All 4 |

---

## 6. Risk & Fallback Strategy

If a teammate's module slips, no one else gets blocked:

| If this is late | Fallback the others use |
| --- | --- |
| Zaid's grid model | Cannot fall back — this is Day 1, must ship. Mitigation: Zaid commits a minimal stub on Day 1 even if validator isn't done. |
| Hasaan's A* | Zaid's `main.py` uses straight-line mock routes (just the start→goal pair). Demo still runs but without obstacle avoidance. |
| Rafay's fleet selector | Zaid hardcodes 3 light + 2 heavy drones. Demo still runs. |
| Saad's regression | Rafay uses uniform-random demand. Selector still produces output. |
| Saad's classifier | Hasaan's simulator emits "Normal" for all drones. Step-18 anomaly is hardcoded for demo. |

---

## 7. Folder Structure (with owners)

```
aeronet_lite/
  data/
    raw/                           # Saad
    processed/
      demand_forecast.csv          # Saad → Rafay
      anomaly_predictions.csv      # Saad → Hasaan
  src/
    grid_model.py                  # Zaid
    layout_validator.py            # Zaid
    fleet_selector.py              # Rafay
    delivery_generator.py          # Rafay
    astar_planner.py               # Hasaan
    delivery_simulator.py          # Hasaan
    ml_pipeline.py                 # Saad (loaders/wrappers)
    visualization.py               # Zaid (zone map) + Saad (heatmap, anomaly view)
    main.py                        # Zaid
  notebooks/
    demand_forecasting.ipynb       # Saad
    anomaly_classifier.ipynb       # Saad
  report/
    figures/                       # all
    final_report.docx              # all (one section each)
  README.md                        # Zaid
```

---

## 8. Submission Checklist (with owners)

- [ ] Working Python project with clear folder structure — **Zaid**
- [ ] 10x10 grid visualization — **Zaid**
- [ ] CSP layout validator with failed-rule reporting — **Zaid**
- [ ] Fleet selection result under budget — **Rafay**
- [ ] A* route planner avoiding no-fly cells — **Hasaan**
- [ ] Rerouting demonstration after disruption — **Hasaan**
- [ ] Regression model with error metric (MAE/RMSE) — **Saad**
- [ ] Classifier with accuracy and confusion matrix — **Saad**
- [ ] 20-step simulation event log — **Zaid** (orchestration), all (modules)
- [ ] Short final report — all four, one section each

---

## 9. Report Section Ownership

| Report section | Owner |
| --- | --- |
| Introduction & scope | Zaid |
| Module 1 — CSP layout validation | Zaid |
| Module 2 — Fleet selection (heuristic + GA) | Rafay |
| Module 3 — A* path planning | Hasaan |
| Module 4 — Disruption handling | Hasaan |
| Module 5 — Demand forecasting & anomaly detection | Saad |
| Integration & 20-step scenario | Zaid |
| Conclusion & limitations | All (one paragraph each) |
