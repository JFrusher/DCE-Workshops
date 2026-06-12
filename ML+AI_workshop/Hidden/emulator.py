from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_pump_telemetry(output_path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_rows = 30_000
    row_index = np.arange(n_rows)
    time_hours = row_index / 60.0
    timestamp = pd.Timestamp("2026-01-01 00:00:00") + pd.to_timedelta(row_index, unit="m")

    wear = np.clip((time_hours - 140.0) / 260.0, 0.0, 1.0)
    load = (
        0.5
        + 0.2 * np.sin(2 * np.pi * time_hours / 24.0)
        + 0.1 * np.sin(2 * np.pi * time_hours / 8.0)
        + 0.03 * rng.normal(size=n_rows)
    )
    load = np.clip(load, 0.1, 1.0)

    true_frequency = 0.075 + 0.025 * wear + 0.003 * np.sin(2 * np.pi * time_hours / 100.0)
    phase = np.cumsum(2 * np.pi * true_frequency)
    clean_vibration = (0.75 + 1.6 * wear + 0.1 * load) * np.sin(phase) + 0.22 * np.sin(2 * phase)

    raw_vibration = clean_vibration + rng.normal(0.0, 1.0, n_rows)
    spike_mask = rng.random(n_rows) < 0.008
    raw_vibration[spike_mask] += rng.normal(0.0, 4.5, spike_mask.sum())

    true_pressure = 118.0 + 0.2 * np.sin(2 * np.pi * time_hours / 180.0)
    drifted_pressure = true_pressure - 0.05 * time_hours + rng.normal(0.0, 0.6, n_rows)
    motor_temp = 65.0 + rng.normal(0.0, 1.0, n_rows)
    fluid_temp = 20.0 + rng.normal(0.0, 0.12, n_rows)

    hidden_rms = np.sqrt(np.convolve(clean_vibration**2, np.ones(61) / 61.0, mode="same"))
    pump_efficiency = np.clip(
        1.0 - 1.0 * np.clip((hidden_rms - 0.5) / 1.0, 0.0, 1.0) + rng.normal(0.0, 0.005, n_rows),
        0.0,
        1.0,
    )

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