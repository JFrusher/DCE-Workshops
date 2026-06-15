from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _rolling_window_average(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    return series.rolling(window=window, min_periods=1, center=True).mean().bfill().ffill().to_numpy()


def generate_pump_telemetry(output_path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_rows = 30_000
    row_index = np.arange(n_rows)
    time_hours = row_index / 60.0
    timestamp = pd.Timestamp("2026-01-01 00:00:00") + pd.to_timedelta(row_index, unit="m")

    wear = np.clip((time_hours - 120.0) / 260.0, 0.0, 1.0)
    operating_load = (
        0.58
        + 0.18 * np.sin(2 * np.pi * time_hours / 24.0)
        + 0.08 * np.sin(2 * np.pi * time_hours / 7.0 + 0.5)
        + 0.02 * rng.normal(size=n_rows)
    )
    operating_load = np.clip(operating_load, 0.15, 0.98)

    true_frequency = 0.075 + 0.022 * wear + 0.003 * np.sin(2 * np.pi * time_hours / 96.0)
    phase = np.cumsum(2 * np.pi * true_frequency)
    vibration_envelope = 0.30 + 0.95 * wear + 0.18 * operating_load
    clean_vibration = vibration_envelope * np.sin(phase) + 0.16 * np.sin(2 * phase)
    clean_vibration += 0.03 * np.sin(2 * np.pi * time_hours / 11.0)

    raw_vibration = clean_vibration + rng.normal(0.0, 1.05, n_rows)
    spike_mask = rng.random(n_rows) < 0.01
    raw_vibration[spike_mask] += rng.normal(0.0, 4.0, spike_mask.sum())

    true_pressure = 120.0 + 1.8 * np.sin(2 * np.pi * time_hours / 180.0) - 0.08 * wear * time_hours / 10.0
    drifted_pressure = true_pressure - 0.05 * time_hours + rng.normal(0.0, 0.55, n_rows)

    motor_temp = 64.0 + 4.5 * operating_load + 2.5 * wear + 0.9 * np.sin(2 * np.pi * time_hours / 9.0)
    motor_temp += rng.normal(0.0, 0.55, n_rows)
    fluid_temp = 20.0 + 0.25 * np.sin(2 * np.pi * time_hours / 18.0) + rng.normal(0.0, 0.14, n_rows)

    pressure_reserve = true_pressure - _rolling_window_average(true_pressure, window=91)
    hidden_rms = np.sqrt(_rolling_window_average(clean_vibration**2, window=61))
    hidden_trend = _rolling_window_average(clean_vibration, window=31) - clean_vibration

    efficiency_raw = (
        0.98
        - 0.62 * np.clip((hidden_rms - 0.32) / 0.75, 0.0, 1.0)
        - 0.26 * np.clip((-pressure_reserve - 1.8) / 4.5, 0.0, 1.0)
        - 0.10 * np.clip((wear - 0.4) / 0.6, 0.0, 1.0)
        - 0.04 * np.clip((motor_temp - 69.0) / 8.0, 0.0, 1.0)
        - 0.03 * np.abs(hidden_trend)
        + rng.normal(0.0, 0.012, n_rows)
    )
    pump_efficiency = np.clip(efficiency_raw, 0.0, 1.0)

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

    dropout_count = int(len(telemetry) * 0.05)
    dropout_rows = rng.choice(telemetry.index[20:], size=dropout_count, replace=False)
    telemetry.loc[dropout_rows, ["raw_vibration", "drifted_pressure", "motor_temp", "fluid_temp"]] = np.nan

    telemetry.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    print(f"Injected {dropout_count} sparse telemetry dropouts into the emulator output.")
    print(telemetry.head())
    return telemetry


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "pump_telemetry.csv"
    generate_pump_telemetry(csv_path)