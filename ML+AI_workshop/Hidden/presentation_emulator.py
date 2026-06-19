from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmulatorConfig:
    seed: int = 42
    start_timestamp: str = "2026-01-01 00:00:00"
    duration_hours: int = 500
    interval_minutes: int = 1
    nominal_hours_end: float = 320.0
    ood_hours_start: float = 360.0
    train_cutoff_hours: float = 380.0
    drift_rate_kpa_per_hour: float = -0.065
    output_name: str = "presentation_pump_telemetry.csv"


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
    def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
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

    @staticmethod
    def _rolling_trend(values: np.ndarray, window: int) -> np.ndarray:
        series = pd.Series(values, dtype="float64")
        return (
            series.rolling(window=window, min_periods=2, center=True)
            .apply(lambda arr: float((arr[-1] - arr[0]) / max(1, len(arr) - 1)), raw=True)
            .bfill()
            .ffill()
            .to_numpy()
        )

    def _build_vibration(self, time_hours: np.ndarray, nominal_mask: np.ndarray, ood_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        latent_bearing_load = 0.18 + 0.03 * np.sin(2 * np.pi * time_hours / 48.0) + 0.02 * np.sin(2 * np.pi * time_hours / 19.0)
        latent_bearing_load += np.where(nominal_mask, 0.0, 0.045 * np.sin(2 * np.pi * time_hours / 4.2))
        latent_bearing_load += np.where(ood_mask, 0.09 + 0.05 * np.sin(2 * np.pi * time_hours / 2.4), 0.0)

        base_noise = self.rng.normal(0.0, 0.17, size=time_hours.size)
        electrical_hash = 1.15 * np.sin(2 * np.pi * time_hours / 0.45) + 0.85 * np.sin(2 * np.pi * time_hours / 0.19)
        electrical_hash += 0.55 * np.sin(2 * np.pi * time_hours / 0.11)

        bursts = np.zeros_like(time_hours)
        burst_centers = self.rng.choice(np.arange(time_hours.size), size=96, replace=False)
        for center in burst_centers:
            width = int(self.rng.integers(4, 16))
            start = max(0, center - width)
            stop = min(time_hours.size, center + width + 1)
            envelope = np.hanning(stop - start)
            bursts[start:stop] += envelope * self.rng.normal(1.2, 0.35)

        modulation = 0.34 + latent_bearing_load
        raw_vibration = modulation * electrical_hash + base_noise + bursts
        raw_vibration += np.where(ood_mask, self.rng.normal(0.0, 0.55, size=time_hours.size), 0.0)
        raw_vibration += np.where(ood_mask, 0.35 * np.sin(2 * np.pi * time_hours * 8.5), 0.0)

        spike_probability = np.where(ood_mask, 0.045, 0.01)
        spike_mask = self.rng.random(time_hours.size) < spike_probability
        raw_vibration[spike_mask] += self.rng.normal(0.0, np.where(ood_mask[spike_mask], 4.0, 1.8))

        vibration_rms = self._rolling_rms(raw_vibration, window=61)
        vibration_trend = self._rolling_trend(raw_vibration, window=61)
        return raw_vibration, vibration_rms, vibration_trend

    def _build_pressure(self, time_hours: np.ndarray, failure_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        calibrated_pressure = 121.0 + 0.35 * np.sin(2 * np.pi * time_hours / 28.0) + 0.12 * np.sin(2 * np.pi * time_hours / 7.0)
        calibrated_pressure += self.rng.normal(0.0, 0.28, size=time_hours.size)
        calibrated_pressure += 0.45 * np.sin(2 * np.pi * time_hours / 4.5)

        cavitation_dips = np.zeros_like(time_hours)
        failure_indices = np.flatnonzero(failure_mask)
        for _ in range(46):
            if failure_indices.size == 0:
                break
            start = int(self.rng.choice(failure_indices))
            duration = int(self.rng.integers(8, 28))
            stop = min(time_hours.size, start + duration)
            segment = np.arange(start, stop)
            if segment.size == 0:
                continue
            center = float(segment.mean())
            sigma = max(1.0, duration / 2.8)
            amplitude = self.rng.uniform(10.0, 26.0)
            cavitation_dips[segment] -= amplitude * np.exp(-0.5 * ((segment - center) / sigma) ** 2)

        calibrated_pressure += cavitation_dips
        drifted_pressure = calibrated_pressure + self.config.drift_rate_kpa_per_hour * time_hours
        pressure_reserve = calibrated_pressure - self._rolling_mean(calibrated_pressure, window=121)
        return drifted_pressure, calibrated_pressure, pressure_reserve

    def _build_motor_temp(self, time_hours: np.ndarray, nominal_mask: np.ndarray, ood_mask: np.ndarray) -> np.ndarray:
        motor_temp = 60.5 + 2.2 * np.sin(2 * np.pi * time_hours / 20.0) + 0.9 * np.sin(2 * np.pi * time_hours / 3.6)
        motor_temp += self.rng.normal(0.0, 1.15, size=time_hours.size)
        motor_temp += np.where(nominal_mask, 0.0, 1.8 * np.sin(2 * np.pi * time_hours / 6.5))
        motor_temp = np.clip(motor_temp, 53.5, 71.5)

        ood_count = int(ood_mask.sum())
        if ood_count:
            hotspots = 67.5 + 4.5 * np.sin(2 * np.pi * time_hours[ood_mask] / 4.0)
            hotspots += 2.3 * np.sin(2 * np.pi * time_hours[ood_mask] / 0.8)
            motor_temp[ood_mask] = hotspots + self.rng.normal(0.0, 1.9, size=ood_count)

        motor_temp[nominal_mask] = np.clip(motor_temp[nominal_mask], 54.5, 66.5)
        return np.clip(motor_temp, 54.0, 102.0)

    def _build_fluid_temp(self, time_hours: np.ndarray) -> np.ndarray:
        return 20.0 + 0.22 * np.sin(2 * np.pi * time_hours / 18.0) + self.rng.normal(0.0, 0.06, size=time_hours.size)

    def _build_efficiency(
        self,
        time_hours: np.ndarray,
        vibration_rms: np.ndarray,
        vibration_trend: np.ndarray,
        calibrated_pressure: np.ndarray,
        pressure_reserve: np.ndarray,
        motor_temp: np.ndarray,
    ) -> np.ndarray:
        efficiency = 0.96 + 0.004 * np.sin(2 * np.pi * time_hours / 96.0)

        late_life = np.clip((time_hours - 310.0) / 95.0, 0.0, 1.0)
        efficiency -= 0.42 * late_life**2.0

        rolling_vibration = self._rolling_mean(vibration_rms, window=121)
        vibration_energy = np.clip((rolling_vibration - 0.08) / 0.08, 0.0, 1.0)
        efficiency -= 0.54 * vibration_energy**1.55

        directional_wear = np.clip(np.maximum(self._rolling_mean(vibration_trend, window=61), 0.0) / 0.028, 0.0, 1.0)
        efficiency -= 0.18 * directional_wear**1.2

        rolling_pressure = self._rolling_mean(calibrated_pressure, window=121)
        cavitation_severity = np.clip((119.0 - rolling_pressure) / 10.0, 0.0, 1.0)
        efficiency -= 0.66 * cavitation_severity**1.55

        thermal_ramp = np.clip(self._rolling_trend(motor_temp, window=61) / 0.18, 0.0, 1.0)
        efficiency -= 0.02 * thermal_ramp
        efficiency -= 0.01 * np.clip(-pressure_reserve / 12.0, 0.0, 1.0)

        efficiency += self.rng.normal(0.0, 0.0045, size=time_hours.size)
        return np.clip(efficiency, 0.04, 0.97)

    def _inject_dropouts(self, telemetry: pd.DataFrame, ood_mask: np.ndarray) -> pd.DataFrame:
        telemetry = telemetry.copy()
        sensor_columns = ["raw_vibration", "drifted_pressure", "motor_temp", "fluid_temp"]
        ood_indices = np.flatnonzero(ood_mask)
        dropout_mask = np.zeros(len(telemetry), dtype=bool)

        for _ in range(110):
            if ood_indices.size == 0:
                break
            start = int(self.rng.choice(ood_indices))
            length = int(self.rng.integers(2, 10))
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

        raw_vibration, vibration_rms, vibration_trend = self._build_vibration(time_hours, nominal_mask, ood_mask)
        drifted_pressure, calibrated_pressure, pressure_reserve = self._build_pressure(time_hours, ood_mask)
        motor_temp = self._build_motor_temp(time_hours, nominal_mask, ood_mask)
        fluid_temp = self._build_fluid_temp(time_hours)
        pump_efficiency = self._build_efficiency(
            time_hours,
            vibration_rms,
            vibration_trend,
            calibrated_pressure,
            pressure_reserve,
            motor_temp,
        )

        # Bearing seizure at hour 420 — creates a sharp step in efficiency and a vibration spike
        # that is easy to point at during the live demo
        seizure_h = 420.0
        seizure_ramp = np.clip((time_hours - seizure_h) / 4.0, 0.0, 1.0)
        pump_efficiency = np.clip(pump_efficiency - 0.20 * seizure_ramp, 0.04, 0.97)
        raw_vibration = raw_vibration.copy()
        spike_window = np.abs(time_hours - seizure_h) < 1.5
        raw_vibration[spike_window] += 9.0 * np.exp(-((time_hours[spike_window] - seizure_h) / 0.6) ** 2)

        telemetry = pd.DataFrame(
            {
                "timestamp": timestamp,
                "time_hours": time_hours,
                "raw_vibration": raw_vibration,
                "drifted_pressure": drifted_pressure,
                "motor_temp": motor_temp,
                "fluid_temp": fluid_temp,
                "pump_efficiency": pump_efficiency,
            }
        )

        telemetry["suggested_split"] = np.where(time_hours <= self.config.train_cutoff_hours, "train", "test")

        telemetry = self._inject_dropouts(telemetry, ood_mask)
        telemetry.to_csv(output_path, index=False)

        print(f"Successfully compiled: {output_path}")
        print(f"Generated {len(telemetry)} records across {self.config.duration_hours} hours.")
        print(f"Total corrupted telemetry dropout frames: {telemetry.attrs.get('dropout_count', 0)}")
        print()
        print(telemetry.head().to_string(index=False))
        return telemetry


def generate_presentation_telemetry(output_path: Path) -> pd.DataFrame:
    return PumpTelemetryEmulator().generate(output_path)


PresentationEmulatorConfig = EmulatorConfig
PresentationPumpTelemetryEmulator = PumpTelemetryEmulator


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / EmulatorConfig().output_name
    generate_presentation_telemetry(csv_path)