from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmulatorConfig:
    seed: int = 42
    start_timestamp: str = "2026-01-01 00:00:00"
    duration_hours: int = 120
    interval_minutes: int = 1
    nominal_hours_end: float = 75.0
    ood_hours_start: float = 76.0
    train_cutoff_hours: float = 80.0  # Chronological boundary to prevent scrambled line charts
    drift_rate_kpa_per_hour: float = -0.065  # Amplified drift to reward pressure reserve engineering
    cavitation_threshold_kpa: float = -7.5
    output_name: str = "pump_telemetry.csv"


class PumpTelemetryEmulator:
    def __init__(self, config: EmulatorConfig | None = None) -> None:
        self.config = config or EmulatorConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def _time_axis(self) -> np.ndarray:
        n_rows = int(self.config.duration_hours * 60 / self.config.interval_minutes)
        return np.arange(n_rows, dtype=float) * (self.config.interval_minutes / 60.0)

    def _timestamp_axis(self, time_hours: np.ndarray) -> pd.DatetimeIndex:
        start = pd.Timestamp(self.config.start_timestamp)
        return pd.DatetimeIndex(start + pd.to_timedelta(time_hours, unit="h"))

    @staticmethod
    def _rolling_average(values: np.ndarray, window: int) -> np.ndarray:
        series = pd.Series(values, dtype="float64")
        return series.rolling(window=window, min_periods=1, center=True).mean().bfill().ffill().to_numpy()

    @staticmethod
    def _rolling_rms(values: np.ndarray, window: int) -> np.ndarray:
        series = pd.Series(values, dtype="float64")
        return (
            series.rolling(window=window, min_periods=1, center=True)
            .apply(lambda arr: float(np.sqrt(np.mean(np.square(arr)))), raw=True)
            .bfill()
            .ffill()
            .to_numpy()
        )

    def _build_vibration(self, time_hours: np.ndarray, nominal_mask: np.ndarray, ood_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Healthy baseline component
        healthy = 0.07 + 0.02 * np.sin(2 * np.pi * time_hours / 19.0) + 0.01 * np.sin(2 * np.pi * time_hours / 7.0)
        healthy += self.rng.normal(0.0, 0.045, size=time_hours.size)

        # High-amplitude EMF sensor bursts (creates raw feature confusion)
        emf_bursts = np.zeros_like(time_hours)
        burst_centers = self.rng.choice(np.arange(time_hours.size), size=14, replace=False)
        for center in burst_centers:
            width = int(self.rng.integers(3, 10))
            start = max(0, center - width)
            stop = min(time_hours.size, center + width + 1)
            envelope = np.hanning(stop - start)
            emf_bursts[start:stop] += envelope * self.rng.normal(0.65, 0.15)

        raw_vibration = healthy + emf_bursts
        # Progressive cyclic structural imbalance during transition/OOD
        raw_vibration += np.where(nominal_mask, 0.0, 0.09 * np.sin(2 * np.pi * time_hours / 3.0))
        # Severe mechanical vibrations during cavitation phase
        raw_vibration += np.where(ood_mask, 0.58 * np.sin(2 * np.pi * time_hours * 10.0), 0.0)
        raw_vibration += np.where(ood_mask, self.rng.normal(0.0, 0.38, size=time_hours.size), 0.0)

        spike_probability = np.where(ood_mask, 0.025, 0.005)
        spike_mask = self.rng.random(time_hours.size) < spike_probability
        raw_vibration[spike_mask] += self.rng.normal(0.0, np.where(ood_mask[spike_mask], 2.4, 1.0))

        vibration_rms = self._rolling_rms(raw_vibration, window=31)
        return raw_vibration, vibration_rms

    def _build_pressure(self, time_hours: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        calibrated_pressure = (
            120.0
            + 0.6 * np.sin(2 * np.pi * time_hours / 24.0)
            + 0.25 * np.sin(2 * np.pi * time_hours / 8.0 + 0.4)
            + self.rng.normal(0.0, 0.13, size=time_hours.size)
        )

        ood_mask = time_hours >= self.config.ood_hours_start
        calibrated_pressure += np.where(ood_mask, -0.8 * np.sin(2 * np.pi * time_hours / 2.5), 0.0)

        # Localized transient cavitation drops
        cavitation_dips = np.zeros_like(time_hours)
        ood_indices = np.flatnonzero(ood_mask)
        for _ in range(18):
            if ood_indices.size == 0:
                break
            start = int(self.rng.choice(ood_indices))
            duration = int(self.rng.integers(4, 12))
            stop = min(time_hours.size, start + duration)
            segment = np.arange(start, stop)
            if segment.size == 0:
                continue
            center = segment.mean()
            sigma = max(1.0, duration / 2.8)
            amplitude = self.rng.uniform(7.5, 15.5)
            cavitation_dips[segment] -= amplitude * np.exp(-0.5 * ((segment - center) / sigma) ** 2)

        calibrated_pressure += cavitation_dips
        # Linear degradation/sensor drift over time
        drifted_pressure = calibrated_pressure + self.config.drift_rate_kpa_per_hour * time_hours
        pressure_reserve = calibrated_pressure - self._rolling_average(calibrated_pressure, window=31)
        return drifted_pressure, calibrated_pressure, pressure_reserve

    def _build_motor_temp(self, time_hours: np.ndarray, nominal_mask: np.ndarray, ood_mask: np.ndarray) -> np.ndarray:
        motor_temp = 60.0 + 2.1 * np.sin(2 * np.pi * time_hours / 12.0) + self.rng.normal(0.0, 0.45, size=time_hours.size)
        motor_temp = np.clip(motor_temp, 55.0, 65.0)

        ood_count = int(ood_mask.sum())
        if ood_count:
            ramp = np.linspace(66.0, 118.0, ood_count)
            motor_temp[ood_mask] = ramp + 1.6 * np.sin(2 * np.pi * time_hours[ood_mask] / 4.0)
            motor_temp[ood_mask] += self.rng.normal(0.0, 1.15, size=ood_count)

        motor_temp[nominal_mask] = np.clip(motor_temp[nominal_mask], 55.0, 65.0)
        return motor_temp

    def _build_fluid_temp(self, time_hours: np.ndarray) -> np.ndarray:
        return 20.0 + 0.28 * np.sin(2 * np.pi * time_hours / 18.0) + self.rng.normal(0.0, 0.08, size=time_hours.size)

    def _build_efficiency(
        self,
        time_hours: np.ndarray,
        vibration_rms: np.ndarray,
        pressure_reserve: np.ndarray,
        motor_temp: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates pump efficiency based on structural wear profiles and localized
        pressure imbalances, forcing models to rely on trend indicators over raw thresholds.
        """
        # Baseline hydraulic performance
        efficiency = 0.85 + 0.015 * np.sin(2 * np.pi * time_hours / 24.0)
        
        # 1. Structural Mechanical Wear (Accelerates rapidly during the final failure phase)
        wear_start_hour = 75.0
        wear_mask = time_hours >= wear_start_hour
        if wear_mask.any():
            # Exponential efficiency decay simulating internal hydraulic recalculation
            degradation = 0.50 * ((time_hours[wear_mask] - wear_start_hour) / (self.config.duration_hours - wear_start_hour)) ** 1.5
            efficiency[wear_mask] -= degradation

        # 2. Localized Transient Cavitation Penalty
        # Rewards high-frequency localized tracking (pressure reserve variance)
        cavitation_mask = pressure_reserve < -4.0
        efficiency[cavitation_mask] -= 0.12

        # 3. Suppress raw linear correlation with motor temperature
        # Adds dynamic thermal drag without letting it serve as a direct proxy for time
        efficiency -= 0.05 * np.clip((motor_temp - 55.0) / 30.0, 0.0, 1.0)

        # Stochastic noise layer
        efficiency += self.rng.normal(0.0, 0.008, size=time_hours.size)
        return np.clip(efficiency, 0.12, 0.95)
    
    def _inject_dropouts(self, telemetry: pd.DataFrame, ood_mask: np.ndarray) -> pd.DataFrame:
        telemetry = telemetry.copy()
        sensor_columns = ["raw_vibration", "drifted_pressure", "motor_temp", "fluid_temp"]
        ood_indices = np.flatnonzero(ood_mask)
        dropout_mask = np.zeros(len(telemetry), dtype=bool)

        for _ in range(25):
            if ood_indices.size == 0:
                break
            start = int(self.rng.choice(ood_indices))
            length = int(self.rng.integers(2, 7))
            stop = min(len(telemetry), start + length)
            dropout_mask[start:stop] = True

        telemetry.loc[dropout_mask, sensor_columns] = np.nan
        telemetry.attrs["dropout_count"] = int(dropout_mask.sum())
        return telemetry

    def generate(self, output_path: Path) -> pd.DataFrame:
        time_hours = self._time_axis()
        timestamp = self._timestamp_axis(time_hours)
        nominal_mask = time_hours <= self.config.nominal_hours_end
        ood_mask = time_hours >= self.config.ood_hours_start

        raw_vibration, vibration_rms = self._build_vibration(time_hours, nominal_mask, ood_mask)
        drifted_pressure, calibrated_pressure, pressure_reserve = self._build_pressure(time_hours)
        motor_temp = self._build_motor_temp(time_hours, nominal_mask, ood_mask)
        fluid_temp = self._build_fluid_temp(time_hours)
        
        # Compute clean target values from physical features
        pump_efficiency = self._build_efficiency(time_hours, vibration_rms, pressure_reserve, motor_temp)

        telemetry = pd.DataFrame(
            {
                "timestamp": timestamp,
                "time_hours": time_hours,
                "raw_vibration": raw_vibration,
                "drifted_pressure": drifted_pressure,
                "motor_temp": motor_temp,
                "fluid_temp": fluid_temp,
                "pump_efficiency": pump_efficiency,
                "calibrated_pressure": calibrated_pressure,
                "pressure_reserve": pressure_reserve,
            }
        )

        telemetry["operating_phase"] = np.select(
            [nominal_mask, ood_mask],
            ["nominal", "ood_cavitation"],
            default="transition",
        )

        # Inject chronological structural flag to explicitly prevent scrambled line chart ordering
        telemetry["suggested_split"] = np.where(time_hours <= self.config.train_cutoff_hours, "train", "test")

        # Inject real field dropout scenarios
        telemetry = self._inject_dropouts(telemetry, ood_mask)
        telemetry.to_csv(output_path, index=False)

        print(f"Successfully compiled: {output_path}")
        print(f"Generated {len(telemetry)} discrete records across {self.config.duration_hours} hours.")
        print(f"Total corrupted telemetry dropout frames: {telemetry.attrs.get('dropout_count', 0)}")
        return telemetry


def generate_pump_telemetry(output_path: Path) -> pd.DataFrame:
    return PumpTelemetryEmulator().generate(output_path)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / EmulatorConfig().output_name
    generate_pump_telemetry(csv_path)