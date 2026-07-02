"""Synthetic pump-telemetry generator for the ML+AI presentation notebook.

Single responsibility: produce the raw-sensor CSV the presentation consumes.
No model fitting, no plotting. Deterministic for a given config.

One latent machine state (bearing load, pressure state, thermal state) drives
both the sensor readings and pump efficiency — from hour 0, not only after
failure onset — so that:
  * a model on the single raw vibration reading gets weak-but-real signal,
  * a model on rolling-window engineered features gets strong signal,
  * a chronological split exposes the failure regime the models never saw.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE_EFFICIENCY = 0.92


@dataclass(frozen=True)
class EmulatorConfig:
    # ── timeline ──────────────────────────────────────────────────────────
    seed: int = 42
    start_timestamp: str = "2026-01-01 00:00:00"
    duration_hours: float = 500.0
    interval_minutes: float = 1.0
    nominal_end_h: float = 320.0
    failure_onset_h: float = 360.0
    train_cutoff_h: float = 380.0
    seizure_h: float = 490.0

    # ── story knobs (effect sizes) ────────────────────────────────────────
    healthy_efficiency_swing: float = 0.06   # in-train efficiency variation from latent state
    raw_signal_fraction: float = 0.30        # how much of raw vibration is signal vs hash
    feature_effect_gain: float = 1.0         # latent state → efficiency coupling strength
    degradation_depth: float = 0.25          # efficiency loss across the failure regime
    degradation_tau_h: float = 6.0           # time constant of post-onset wear ramp
    cavitation_effect: float = 0.18          # efficiency loss per unit cavitation penalty
    seizure_drop: float = 0.15               # extra step drop at seizure_h
    efficiency_noise_std: float = 0.015      # irreducible target noise

    # ── realism knobs ─────────────────────────────────────────────────────
    drift_rate_kpa_per_hour: float = -0.065
    n_dropout_events: int = 110              # NaN windows, failure regime only
    n_cavitation_events: int = 46            # pressure-dip events, failure regime only

    def __post_init__(self) -> None:
        if self.interval_minutes <= 0:
            raise ValueError(f"interval_minutes must be > 0, got {self.interval_minutes}")
        previous_name, previous_value = "0", 0.0
        for name in ("nominal_end_h", "failure_onset_h", "train_cutoff_h", "seizure_h", "duration_hours"):
            value = getattr(self, name)
            if value <= previous_value:
                raise ValueError(f"{name} ({value}) must be greater than {previous_name} ({previous_value})")
            previous_name, previous_value = name, value
        if not 0.0 < self.raw_signal_fraction < 1.0:
            raise ValueError(f"raw_signal_fraction must be in (0, 1), got {self.raw_signal_fraction}")
        for name in ("healthy_efficiency_swing", "feature_effect_gain", "degradation_tau_h"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        for name in ("degradation_depth", "cavitation_effect", "seizure_drop", "efficiency_noise_std"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        for name in ("n_dropout_events", "n_cavitation_events"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        # Efficiency budget: everything subtracted from BASE_EFFICIENCY must leave
        # headroom, otherwise the target clips flat — the failure mode this
        # generator exists to avoid.
        floor = (
            BASE_EFFICIENCY
            - 1.5 * self.feature_effect_gain * self.healthy_efficiency_swing
            - self.degradation_depth
            - 1.5 * self.cavitation_effect
            - self.seizure_drop
        )
        if floor < 0.05:
            raise ValueError(
                "efficiency budget exhausted: base 0.92 minus swing/degradation/cavitation/"
                f"seizure leaves floor {floor:.3f} < 0.05 — reduce healthy_efficiency_swing, "
                "feature_effect_gain, degradation_depth, cavitation_effect or seizure_drop"
            )


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    return series.rolling(window=window, min_periods=1, center=True).mean().bfill().ffill().to_numpy()


class PumpTelemetryEmulator:
    def __init__(self, config: EmulatorConfig = EmulatorConfig()) -> None:
        self.config = config

    def generate(self, output_path: Path | None = None) -> pd.DataFrame:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        step_h = cfg.interval_minutes / 60.0
        n = int(round(cfg.duration_hours * 60.0 / cfg.interval_minutes))
        t = np.arange(n, dtype=float) * step_h
        timestamp = pd.Timestamp(cfg.start_timestamp) + pd.to_timedelta(t, unit="h")
        failure_mask = t >= cfg.failure_onset_h

        # ── latent machine state (slow: recoverable by ~1 h rolling windows) ──
        wear = 1.0 - np.exp(-np.clip(t - cfg.failure_onset_h, 0.0, None) / cfg.degradation_tau_h)
        seizure_ramp = np.clip((t - cfg.seizure_h) / 2.0, 0.0, 1.0)
        slow_noise = _rolling_mean(rng.normal(0.0, 1.0, n), window=181)
        bearing_load = (
            0.5
            + 0.30 * np.sin(2 * np.pi * t / 45.0)
            + 0.18 * np.sin(2 * np.pi * t / 13.7)
            + 2.2 * slow_noise
            + 1.1 * wear
            + 1.0 * seizure_ramp  # seized bearing: vibration stays elevated after the event
        )
        thermal_state = (
            0.5
            + 0.30 * np.sin(2 * np.pi * t / 20.0)
            + 0.15 * np.sin(2 * np.pi * t / 6.3)
            + 1.4 * wear
        )

        # ── pressure: waves + failure-regime cavitation dips ──
        pressure_wave = (
            1.1 * np.sin(2 * np.pi * t / 2.6)
            + 0.6 * np.sin(2 * np.pi * t / 7.0)
            + rng.normal(0.0, 0.28, n)
        )
        cavitation_dips = np.zeros(n)
        failure_indices = np.flatnonzero(failure_mask)
        for _ in range(cfg.n_cavitation_events):
            if failure_indices.size == 0:
                break
            start = int(rng.choice(failure_indices))
            duration = int(rng.integers(8, 28))
            segment = np.arange(start, min(n, start + duration))
            center = float(segment.mean())
            sigma = max(1.0, duration / 2.8)
            amplitude = rng.uniform(10.0, 26.0)
            cavitation_dips[segment] -= amplitude * np.exp(-0.5 * ((segment - center) / sigma) ** 2)
        calibrated_pressure = 121.0 + pressure_wave + cavitation_dips
        drifted_pressure = calibrated_pressure + cfg.drift_rate_kpa_per_hour * t

        # ── pump efficiency: driven by smoothed latent state, not by time ──
        load_smooth = _rolling_mean(bearing_load, window=61)
        thermal_smooth = _rolling_mean(thermal_state, window=61)
        pressure_penalty = np.clip((119.0 - _rolling_mean(calibrated_pressure, window=61)) / 6.0, 0.0, 1.5)

        healthy_loss = cfg.healthy_efficiency_swing * (
            1.0 * (load_smooth - 0.5)
            + 0.35 * (thermal_smooth - 0.5)
        )
        # wear is observable only through the same latents the sensors see:
        # degradation loss keys off elevated smoothed bearing load
        wear_proxy = np.clip((load_smooth - 1.0) / 0.55, 0.0, 1.0)

        pump_efficiency = (
            BASE_EFFICIENCY
            - cfg.feature_effect_gain * healthy_loss
            - cfg.degradation_depth * wear_proxy
            - cfg.cavitation_effect * pressure_penalty
            - cfg.seizure_drop * seizure_ramp
            + rng.normal(0.0, cfg.efficiency_noise_std, n)
        )
        pump_efficiency = np.clip(pump_efficiency, 0.05, 0.97)

        # ── raw vibration: latent load amplitude-modulates an electrical-hash carrier ──
        hash_carrier = (
            1.15 * np.sin(2 * np.pi * t / 0.45)
            + 0.85 * np.sin(2 * np.pi * t / 0.19)
            + 0.55 * np.sin(2 * np.pi * t / 0.11)
        )
        amplitude = np.clip(0.45 * (1.0 + 4.0 * cfg.raw_signal_fraction * (bearing_load - 0.5)), 0.06, None)
        raw_vibration = (
            amplitude * hash_carrier
            + 3.6 * cfg.raw_signal_fraction * (bearing_load - 0.5)
            + rng.normal(0.0, 0.04 + 0.24 * (1.0 - cfg.raw_signal_fraction), n)
        )

        burst_centers = rng.choice(n, size=96, replace=False)
        for center in burst_centers:
            width = int(rng.integers(4, 16))
            start, stop = max(0, center - width), min(n, center + width + 1)
            raw_vibration[start:stop] += np.hanning(stop - start) * rng.normal(1.2, 0.35)

        spike_mask = rng.random(n) < np.where(failure_mask, 0.045, 0.01)
        raw_vibration[spike_mask] += rng.normal(0.0, np.where(failure_mask[spike_mask], 4.0, 1.8))

        seizure_window = np.abs(t - cfg.seizure_h) < 1.5
        raw_vibration[seizure_window] += 9.0 * np.exp(-((t[seizure_window] - cfg.seizure_h) / 0.6) ** 2)

        # ── remaining sensors ──
        motor_temp = np.clip(53.5 + 12.0 * thermal_state + rng.normal(0.0, 0.8, n), 50.0, 105.0)
        fluid_temp = 20.0 + 0.22 * np.sin(2 * np.pi * t / 18.0) + rng.normal(0.0, 0.06, n)

        telemetry = pd.DataFrame(
            {
                "timestamp": timestamp,
                "time_hours": t,
                "raw_vibration": raw_vibration,
                "drifted_pressure": drifted_pressure,
                "motor_temp": motor_temp,
                "fluid_temp": fluid_temp,
                "pump_efficiency": pump_efficiency,
                "suggested_split": np.where(t <= cfg.train_cutoff_h, "train", "test"),
            }
        )

        # ── sensor dropouts (NaN windows), failure regime only ──
        sensor_columns = ["raw_vibration", "drifted_pressure", "motor_temp", "fluid_temp"]
        dropout_mask = np.zeros(n, dtype=bool)
        for _ in range(cfg.n_dropout_events):
            if failure_indices.size == 0:
                break
            start = int(rng.choice(failure_indices))
            dropout_mask[start:min(n, start + int(rng.integers(2, 10)))] = True
        telemetry.loc[dropout_mask, sensor_columns] = np.nan
        telemetry.attrs["dropout_count"] = int(dropout_mask.sum())

        if output_path is not None:
            telemetry.to_csv(output_path, index=False)
        return telemetry


if __name__ == "__main__":
    # Self-check: replicate the presentation notebook's cleaning + feature
    # pipeline and assert the R² landscape matches the story the notebook tells.
    from scipy.signal import savgol_filter
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    df = PumpTelemetryEmulator().generate()

    sensor_columns = ["raw_vibration", "drifted_pressure", "motor_temp", "fluid_temp"]
    df[sensor_columns] = df[sensor_columns].ffill().bfill()
    cleaned = savgol_filter(df["raw_vibration"].to_numpy(), window_length=15, polyorder=2)
    cleaned_series = pd.Series(cleaned)
    df["vibration_rms"] = (
        cleaned_series.rolling(61, min_periods=1, center=True)
        .apply(lambda a: float(np.sqrt(np.mean(np.square(a)))), raw=True)
        .to_numpy()
    )
    df["vibration_trend"] = (
        cleaned_series.rolling(31, min_periods=2, center=True)
        .apply(lambda a: float((a[-1] - a[0]) / max(1, len(a) - 1)), raw=True)
        .bfill()
        .to_numpy()
    )
    calibrated = df["drifted_pressure"] - EmulatorConfig().drift_rate_kpa_per_hour * df["time_hours"]
    df["pressure_reserve"] = calibrated - calibrated.rolling(91, min_periods=1, center=True).mean()

    raw_features = ["raw_vibration"]
    engineered_features = ["vibration_rms", "vibration_trend", "pressure_reserve", "motor_temp"]

    def fit_score(train, test, features, n_estimators):
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        model.fit(train[features], train["pump_efficiency"])
        return r2_score(test["pump_efficiency"], model.predict(test[features]))

    train_shuf, test_shuf = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
    train_chrono = df[df["suggested_split"] == "train"]
    test_chrono = df[df["suggested_split"] == "test"]

    shuf_raw = fit_score(train_shuf, test_shuf, raw_features, 120)
    shuf_eng = fit_score(train_shuf, test_shuf, engineered_features, 160)
    chrono_raw = fit_score(train_chrono, test_chrono, raw_features, 120)
    chrono_eng = fit_score(train_chrono, test_chrono, engineered_features, 160)

    print(f"shuffled  raw R²: {shuf_raw:+.3f}   engineered R²: {shuf_eng:+.3f}")
    print(f"chrono    raw R²: {chrono_raw:+.3f}   engineered R²: {chrono_eng:+.3f}")

    assert 0.10 <= shuf_raw <= 0.45, f"shuffled raw R² {shuf_raw:.3f} outside [0.10, 0.45] — trap has no bait"
    assert shuf_eng >= 0.70, f"shuffled engineered R² {shuf_eng:.3f} < 0.70 — features not dominant"
    assert chrono_raw <= 0.05, f"chronological raw R² {chrono_raw:.3f} > 0.05 — raw model should collapse"
    assert 0.10 <= chrono_eng < shuf_eng, f"chronological engineered R² {chrono_eng:.3f} should be in [0.10, {shuf_eng:.3f})"

    output = Path(__file__).resolve().parents[1] / "presentation_pump_telemetry.csv"
    PumpTelemetryEmulator().generate(output)
    print(f"self-check passed — wrote {output}")
