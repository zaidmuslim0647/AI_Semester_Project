"""Module 5: ML Pipeline for demand forecasting and anomaly detection.

Owner: Saad

This module provides:
1. Demand forecasting regression (Linear Regression + Random Forest)
2. Anomaly classification (Decision Tree + Random Forest)
3. CSV loaders for integration with simulator

The actual model training happens in notebooks:
- notebooks/demand_forecasting.ipynb
- notebooks/anomaly_classifier.ipynb
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


@dataclass
class DemandForecast:
    """Stores demand predictions for grid cells."""
    demand_grid: np.ndarray  # 10x10 array of predicted demand values
    
    def get_demand(self, row: int, col: int) -> float:
        """Get predicted demand for a cell."""
        if 0 <= row < len(self.demand_grid) and 0 <= col < len(self.demand_grid[0]):
            return float(self.demand_grid[row, col])
        return 0.0


@dataclass
class AnomalyPrediction:
    """Stores anomaly prediction for a drone at a timestep."""
    drone_id: str
    step: int
    label: str  # "Normal" | "Battery" | "Route" | "Sensor"
    confidence: float = 0.0


class DemandForecastLoader:
    """Load demand forecast CSV and convert to grid format."""
    
    @staticmethod
    def load_from_csv(csv_path: Optional[str] = None) -> DemandForecast:
        """
        Load demand forecast from CSV.
        
        Expected CSV columns: row, col, predicted_demand
        
        Args:
            csv_path: Path to CSV file. If None, uses default location.
            
        Returns:
            DemandForecast object with 10x10 demand grid.
        """
        if csv_path is None:
            csv_path = DATA_DIR / "demand_forecast.csv"
        
        # Initialize 10x10 grid with zeros
        demand_grid = np.zeros((10, 10), dtype=float)
        
        if not os.path.exists(csv_path):
            # Return zero-demand grid if file doesn't exist yet
            return DemandForecast(demand_grid=demand_grid)
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    r = int(row['row'])
                    c = int(row['col'])
                    demand = float(row['predicted_demand'])
                    if 0 <= r < 10 and 0 <= c < 10:
                        demand_grid[r, c] = demand
        except (OSError, ValueError, KeyError) as e:
            print(f"Warning: Failed to load demand forecast from {csv_path}: {e}")
        
        return DemandForecast(demand_grid=demand_grid)
    
    @staticmethod
    def save_forecast_csv(
        demand_grid: np.ndarray,
        csv_path: Optional[str] = None,
    ) -> None:
        """
        Save demand forecast to CSV.
        
        Args:
            demand_grid: 10x10 array of predicted demand values.
            csv_path: Output path. If None, uses default location.
        """
        if csv_path is None:
            csv_path = DATA_DIR / "demand_forecast.csv"
        
        output_path = Path(csv_path)
        
        # Create the parent directory when the output path includes one.
        if output_path.parent != Path("."):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['row', 'col', 'predicted_demand'])
            writer.writeheader()
            for r in range(demand_grid.shape[0]):
                for c in range(demand_grid.shape[1]):
                    writer.writerow({
                        'row': r,
                        'col': c,
                        'predicted_demand': float(demand_grid[r, c]),
                    })


class AnomalyClassifierLoader:
    """Load anomaly predictions CSV."""
    
    @staticmethod
    def load_from_csv(csv_path: Optional[str] = None) -> Dict[str, List[AnomalyPrediction]]:
        """
        Load anomaly predictions from CSV.
        
        Expected CSV columns: drone_id, step, label, confidence (optional)
        
        Args:
            csv_path: Path to CSV file. If None, uses default location.
            
        Returns:
            Dict mapping drone_id to list of AnomalyPrediction objects.
        """
        if csv_path is None:
            csv_path = DATA_DIR / "anomaly_predictions.csv"
        
        predictions: Dict[str, List[AnomalyPrediction]] = {}
        
        if not os.path.exists(csv_path):
            return predictions
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    drone_id = row['drone_id']
                    step = int(row['step'])
                    label = row['label']
                    confidence = float(row.get('confidence', 0.0))
                    
                    if drone_id not in predictions:
                        predictions[drone_id] = []
                    
                    predictions[drone_id].append(
                        AnomalyPrediction(
                            drone_id=drone_id,
                            step=step,
                            label=label,
                            confidence=confidence,
                        )
                    )
        except (OSError, ValueError, KeyError) as e:
            print(f"Warning: Failed to load anomaly predictions from {csv_path}: {e}")
        
        return predictions
    
    @staticmethod
    def save_predictions_csv(
        predictions: Dict[str, List[AnomalyPrediction]],
        csv_path: Optional[str] = None,
    ) -> None:
        """
        Save anomaly predictions to CSV.
        
        Args:
            predictions: Dict mapping drone_id to list of AnomalyPrediction objects.
            csv_path: Output path. If None, uses default location.
        """
        if csv_path is None:
            csv_path = DATA_DIR / "anomaly_predictions.csv"
        
        csv_path = Path(csv_path)
        
        # Create directory if it doesn't exist
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['drone_id', 'step', 'label', 'confidence']
            )
            writer.writeheader()
            
            for drone_id, preds in predictions.items():
                for pred in preds:
                    writer.writerow({
                        'drone_id': pred.drone_id,
                        'step': pred.step,
                        'label': pred.label,
                        'confidence': pred.confidence,
                    })
    
    @staticmethod
    def get_label_for_drone_step(
        predictions: Dict[str, List[AnomalyPrediction]],
        drone_id: str,
        step: int,
    ) -> str:
        """
        Get anomaly label for a specific drone at a specific step.
        
        Args:
            predictions: Loaded predictions dict.
            drone_id: Drone identifier.
            step: Simulation step.
            
        Returns:
            Label string ("Normal" | "Battery" | "Route" | "Sensor"), or "Normal" if not found.
        """
        if drone_id not in predictions:
            return "Normal"
        
        for pred in predictions[drone_id]:
            if pred.step == step:
                return pred.label
        
        return "Normal"


class SyntheticAnomalyGenerator:
    """Generate synthetic anomaly data based on simulation metrics.
    
    Rules from spec:
    - Battery: battery_drop > threshold (high drain rates)
    - Route: route_deviation > threshold (unexpected path changes)
    - Sensor: speed/altitude jump > threshold (sensor errors)
    - Normal: all metrics within bounds
    """
    
    @staticmethod
    def generate_label(
        battery_drop: float,
        route_deviation: float,
        speed_jump: float,
        altitude_jump: float,
    ) -> str:
        """
        Generate anomaly label based on drone metrics.
        
        Args:
            battery_drop: Battery percentage drop in last step (0-100).
            route_deviation: Deviation from planned route in cells (0-10).
            speed_jump: Sudden change in speed (0-1, normalized).
            altitude_jump: Sudden change in altitude (0-1, normalized).
            
        Returns:
            Anomaly label: "Normal" | "Battery" | "Route" | "Sensor"
        """
        # Thresholds (to be tuned in notebooks)
        BATTERY_THRESHOLD = 8.0
        ROUTE_THRESHOLD = 2.0
        SPEED_JUMP_THRESHOLD = 0.6
        ALTITUDE_JUMP_THRESHOLD = 0.6
        
        # Priority: Battery > Route > Sensor > Normal
        if battery_drop > BATTERY_THRESHOLD:
            return "Battery"
        
        if route_deviation > ROUTE_THRESHOLD:
            return "Route"
        
        if speed_jump > SPEED_JUMP_THRESHOLD or altitude_jump > ALTITUDE_JUMP_THRESHOLD:
            return "Sensor"

        return "Normal"


# --------------------------------------------------------------------------- #
# Top-level adapter functions matching the Day-1 contract used by main.py.
# These wrap the class-based API above so the orchestrator can call them
# directly without needing to know about loaders.
# --------------------------------------------------------------------------- #

_ANOMALY_CACHE: Optional[Dict[str, List[AnomalyPrediction]]] = None


def load_demand_forecast(grid) -> Dict[Tuple[int, int], float]:
    """Return predicted demand keyed by (row, col), reading Saad's CSV.

    Falls back to a zero-demand grid if the CSV is missing.
    """
    forecast = DemandForecastLoader.load_from_csv()
    n_rows = forecast.demand_grid.shape[0]
    n_cols = forecast.demand_grid.shape[1]
    return {
        (r, c): float(forecast.demand_grid[r, c])
        for r in range(n_rows)
        for c in range(n_cols)
    }


def predict_anomaly(drone, step: int) -> str:
    """Look up the predicted anomaly label for a drone at a given step.

    Loads anomaly_predictions.csv on first call and caches the result.
    """
    global _ANOMALY_CACHE
    if _ANOMALY_CACHE is None:
        _ANOMALY_CACHE = AnomalyClassifierLoader.load_from_csv()
    return AnomalyClassifierLoader.get_label_for_drone_step(
        _ANOMALY_CACHE, drone.id, step
    )
