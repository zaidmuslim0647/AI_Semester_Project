"""CSP-style layout validator for AeroNet Lite (Module 1).

Owner: Zaid. Implements all four rules from the spec:
  R1 — Industrial adjacency
  R2 — Residential coverage by hubs
  R3 — Hub charging proximity
  R4 — Hospital medical pickup access

Each rule returns a list of Violation objects. Violations include a suggested fix
so the report is useful during viva.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

from grid_model import (
    Cell,
    Coord,
    GRID_SIZE,
    charging_pads,
    get_neighbors,
    hubs,
    make_sample_grid,
    manhattan,
    medical_pickups,
    print_grid,
)


@dataclass
class Violation:
    rule_id: str
    cell: Coord
    message: str
    suggestion: str = ""


@dataclass
class ValidationReport:
    rule_results: List[Tuple[str, bool, str]] = field(default_factory=list)
    violations: List[Violation] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return all(passed for _, passed, _ in self.rule_results)

    def add_rule(self, rule_id: str, description: str, rule_violations: List[Violation]) -> None:
        self.rule_results.append((rule_id, len(rule_violations) == 0, description))
        self.violations.extend(rule_violations)

    def print(self) -> None:
        print("=" * 64)
        print(f"AeroNet Lite — Layout Validation Report")
        print("=" * 64)
        print(f"Layout validity = {self.is_valid}\n")

        print("Rule summary:")
        for rule_id, passed, description in self.rule_results:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {rule_id}: {description}")

        if self.violations:
            print("\nViolations:")
            for v in self.violations:
                print(f"  - {v.rule_id} @ {v.cell}: {v.message}")
                if v.suggestion:
                    print(f"      -> Suggested fix: {v.suggestion}")
        else:
            print("\nNo violations found.")
        print("=" * 64)


# ---------- Rule R1: Industrial adjacency ---------------------------------- #

R1_DESCRIPTION = "Industrial cells must not be directly adjacent to Schools or Hospitals."


def check_industrial_safety(grid: List[List[Cell]]) -> List[Violation]:
    violations: List[Violation] = []
    n = len(grid)
    for r in range(n):
        for c in range(n):
            if grid[r][c].zone != "Industrial":
                continue
            for (nr, nc) in get_neighbors(r, c, size=n):
                neighbor = grid[nr][nc]
                if neighbor.zone in ("School", "Hospital"):
                    violations.append(
                        Violation(
                            rule_id="R1",
                            cell=(r, c),
                            message=(
                                f"Industrial cell ({r}, {c}) is adjacent to "
                                f"{neighbor.zone} at ({nr}, {nc})."
                            ),
                            suggestion=(
                                f"Move the {neighbor.zone.lower()} away from "
                                f"({nr}, {nc}) or convert ({r}, {c}) to Open Field."
                            ),
                        )
                    )
    return violations


# ---------- Rule R2: Residential coverage by hubs -------------------------- #

R2_DESCRIPTION = "Every Residential cell must be within 3 Manhattan cells of a Drone Hub."
R2_MAX_DISTANCE = 3


def check_residential_coverage(grid: List[List[Cell]]) -> List[Violation]:
    violations: List[Violation] = []
    hub_coords = [(h.row, h.col) for h in hubs(grid)]

    if not hub_coords:
        for row in grid:
            for cell in row:
                if cell.zone == "Residential":
                    violations.append(
                        Violation(
                            rule_id="R2",
                            cell=(cell.row, cell.col),
                            message=f"Residential cell ({cell.row}, {cell.col}) "
                                    f"has no hub on the grid at all.",
                            suggestion="Place at least one Drone Hub within 3 cells.",
                        )
                    )
        return violations

    for row in grid:
        for cell in row:
            if cell.zone != "Residential":
                continue
            origin = (cell.row, cell.col)
            distances = [manhattan(origin, h) for h in hub_coords]
            nearest = min(distances)
            if nearest > R2_MAX_DISTANCE:
                nearest_hub = hub_coords[distances.index(nearest)]
                violations.append(
                    Violation(
                        rule_id="R2",
                        cell=origin,
                        message=(
                            f"Residential cell {origin} is {nearest} cells from the "
                            f"nearest hub {nearest_hub} (limit is {R2_MAX_DISTANCE})."
                        ),
                        suggestion=(
                            f"Add a hub near {origin} or convert this cell to Open Field."
                        ),
                    )
                )
    return violations


# ---------- Rule R3: Hub-to-charging-pad proximity ------------------------- #

R3_DESCRIPTION = "Every Drone Hub must have a Charging Pad within 2 Manhattan cells."
R3_MAX_DISTANCE = 2


def check_hub_charging(grid: List[List[Cell]]) -> List[Violation]:
    violations: List[Violation] = []
    pad_coords = [(p.row, p.col) for p in charging_pads(grid)]

    for hub in hubs(grid):
        origin = (hub.row, hub.col)
        if not pad_coords:
            violations.append(
                Violation(
                    rule_id="R3",
                    cell=origin,
                    message=f"Hub {origin} has no charging pad anywhere on the grid.",
                    suggestion=f"Place a charging pad within {R3_MAX_DISTANCE} cells of {origin}.",
                )
            )
            continue

        distances = [manhattan(origin, p) for p in pad_coords]
        nearest = min(distances)
        if nearest > R3_MAX_DISTANCE:
            nearest_pad = pad_coords[distances.index(nearest)]
            violations.append(
                Violation(
                    rule_id="R3",
                    cell=origin,
                    message=(
                        f"Hub {origin} is {nearest} cells from the nearest charging pad "
                        f"{nearest_pad} (limit is {R3_MAX_DISTANCE})."
                    ),
                    suggestion=(
                        f"Add a charging pad within {R3_MAX_DISTANCE} cells of {origin}."
                    ),
                )
            )
    return violations


# ---------- Rule R4: Hospital medical-pickup access ------------------------ #

R4_DESCRIPTION = "At least one Hospital must have a Medical Pickup point within 1 cell."
R4_MAX_DISTANCE = 1


def check_medical_access(grid: List[List[Cell]]) -> List[Violation]:
    violations: List[Violation] = []
    hospital_cells = [c for row in grid for c in row if c.zone == "Hospital"]
    pickup_coords = [(p.row, p.col) for p in medical_pickups(grid)]

    if not hospital_cells:
        return violations  # No hospitals -> rule trivially satisfied

    if not pickup_coords:
        first = (hospital_cells[0].row, hospital_cells[0].col)
        violations.append(
            Violation(
                rule_id="R4",
                cell=first,
                message="Grid has hospitals but no Medical Pickup points anywhere.",
                suggestion=f"Place a Medical Pickup within {R4_MAX_DISTANCE} cell of any hospital.",
            )
        )
        return violations

    # Rule: at least ONE hospital must satisfy. Report only if all fail.
    any_satisfied = False
    for hosp in hospital_cells:
        origin = (hosp.row, hosp.col)
        nearest = min(manhattan(origin, p) for p in pickup_coords)
        if nearest <= R4_MAX_DISTANCE:
            any_satisfied = True
            break

    if not any_satisfied:
        # Report against the first hospital with a clear suggestion.
        hosp = hospital_cells[0]
        origin = (hosp.row, hosp.col)
        nearest = min(manhattan(origin, p) for p in pickup_coords)
        nearest_pickup = pickup_coords[
            [manhattan(origin, p) for p in pickup_coords].index(nearest)
        ]
        violations.append(
            Violation(
                rule_id="R4",
                cell=origin,
                message=(
                    f"No hospital has a Medical Pickup within {R4_MAX_DISTANCE} cell. "
                    f"Closest pair: hospital {origin} <-> pickup {nearest_pickup} ({nearest} cells)."
                ),
                suggestion=(
                    f"Add a Medical Pickup adjacent to a hospital (e.g. next to {origin})."
                ),
            )
        )
    return violations


# ---------- Aggregator ----------------------------------------------------- #

REGISTERED_RULES: List[Tuple[str, str, Callable[[List[List[Cell]]], List[Violation]]]] = [
    ("R1", R1_DESCRIPTION, check_industrial_safety),
    ("R2", R2_DESCRIPTION, check_residential_coverage),
    ("R3", R3_DESCRIPTION, check_hub_charging),
    ("R4", R4_DESCRIPTION, check_medical_access),
]


def validate_layout(grid: List[List[Cell]]) -> ValidationReport:
    report = ValidationReport()
    for rule_id, description, fn in REGISTERED_RULES:
        report.add_rule(rule_id, description, fn(grid))
    return report


# ---------- Demo ----------------------------------------------------------- #

def _demo_violation_grid() -> List[List[Cell]]:
    """Sample grid with intentional violations of every rule for demo output."""
    grid = make_sample_grid()
    # R1 violation: drop a School next to the industrial cell at (5, 0)
    grid[5][1].zone = "School"
    grid[5][1].density = 500
    # R2 violation: residential cell far from both hubs (2,2) and (7,7)
    grid[9][9].zone = "Residential"
    grid[9][9].density = 4500
    # R3 violation: remove the charging pad next to hub (7,7)
    grid[7][6].is_charging = False
    # R4 violation: remove the medical pickup next to the hospital
    grid[1][9].is_medical_pickup = False
    return grid


if __name__ == "__main__":
    print("== Clean sample grid ==")
    clean = make_sample_grid()
    print_grid(clean)
    validate_layout(clean).print()

    print("\n== Grid with intentional violations ==")
    bad = _demo_violation_grid()
    print_grid(bad)
    validate_layout(bad).print()
