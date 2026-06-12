from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


np.random.seed(42)
RNG = np.random.default_rng(42)

OUTPUT_DIR = Path(__file__).resolve().parent
CLEAN_FILE = OUTPUT_DIR / 'ground_truth_clean.csv'
MESSY_FILE = OUTPUT_DIR / 'messy_turbine_telemetry.csv'
LEGACY_FILE = OUTPUT_DIR / 'wind_turbine_telemetry.csv'

START_TIMESTAMP = pd.Timestamp('2026-01-01 00:00:00')
INTERVALS_PER_DAY = 24 * 6
PERIODS = 90 * INTERVALS_PER_DAY
TIMESTAMPS = pd.date_range(start=START_TIMESTAMP, periods=PERIODS, freq='10min')

RATED_POWER_KW = 2500.0
RATED_RPM = 15.0
CUT_IN_WIND_MPS = 3.0

WT02_DEGRADATION_START = 30 * INTERVALS_PER_DAY
WT02_SEIZURE_INDEX = 45 * INTERVALS_PER_DAY
WT03_DRIFT_START = 60 * INTERVALS_PER_DAY


@dataclass(frozen=True)
class TurbineProfile:
    turbine_id: str
    gust_lag: int
    wake_bias: float
    wake_scale: float
    temp_micro_sigma: float
    rpm_response: float
    bearing_temp_alpha: float
    bearing_temp_beta: float
    bearing_cooling: float
    vibration_scale: float
    degradation_scale: float
    power_scale: float


TURBINE_PROFILES = {
    'WT-01': TurbineProfile('WT-01', gust_lag=0, wake_bias=0.00, wake_scale=0.10, temp_micro_sigma=0.20, rpm_response=0.22, bearing_temp_alpha=0.58, bearing_temp_beta=2.6, bearing_cooling=0.18, vibration_scale=0.026, degradation_scale=0.00, power_scale=1.00),
    'WT-02': TurbineProfile('WT-02', gust_lag=1, wake_bias=-0.05, wake_scale=0.13, temp_micro_sigma=0.22, rpm_response=0.20, bearing_temp_alpha=0.60, bearing_temp_beta=3.2, bearing_cooling=0.16, vibration_scale=0.028, degradation_scale=1.00, power_scale=0.995),
    'WT-03': TurbineProfile('WT-03', gust_lag=1, wake_bias=-0.03, wake_scale=0.12, temp_micro_sigma=0.21, rpm_response=0.21, bearing_temp_alpha=0.57, bearing_temp_beta=2.1, bearing_cooling=0.17, vibration_scale=0.027, degradation_scale=0.20, power_scale=0.998),
}


def shift_series(values: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0:
        return values.copy()
    shifted = np.empty_like(values)
    shifted[:lag] = values[0]
    shifted[lag:] = values[:-lag]
    return shifted


def build_shared_environment() -> pd.DataFrame:
    steps = np.arange(PERIODS)
    hours = steps / 6.0
    days = hours / 24.0
    hour_of_day = TIMESTAMPS.hour.to_numpy() + (TIMESTAMPS.minute.to_numpy() / 60.0)

    seasonal_downward = 15.5 - (3.3 * (days / days.max()))
    daily_cycle = 3.1 * np.sin(2 * np.pi * (hour_of_day - 14.0) / 24.0)
    weather_front = np.cumsum(RNG.normal(0.0, 0.06, PERIODS))
    ambient_shared = np.clip(seasonal_downward + daily_cycle + 0.5 * weather_front, -5.0, 28.0)

    weibull_base = RNG.weibull(2.15, PERIODS) * 6.4
    macro_wind = np.empty(PERIODS, dtype=float)
    macro_wind[0] = weibull_base[0]
    for idx in range(1, PERIODS):
        drift = 0.075 * (weibull_base[idx] - macro_wind[idx - 1])
        macro_wind[idx] = np.clip(macro_wind[idx - 1] + drift + RNG.normal(0.0, 0.18), 0.5, 24.0)

    gust_impulses = np.where(RNG.random(PERIODS) < 0.055, RNG.gamma(shape=2.0, scale=1.3, size=PERIODS), 0.0)
    gust_base = np.zeros(PERIODS, dtype=float)
    gust_base[0] = gust_impulses[0]
    for idx in range(1, PERIODS):
        gust_base[idx] = np.clip(0.55 * gust_base[idx - 1] + gust_impulses[idx], 0.0, 10.0)

    turbulence_intensity = np.clip(0.05 + 0.013 * macro_wind + 0.018 * gust_base + RNG.normal(0.0, 0.008, PERIODS), 0.03, 0.45)
    atmospheric_pressure = np.round(1015.8 - 0.82 * macro_wind + 0.18 * weather_front + RNG.normal(0.0, 0.45, PERIODS), 2)
    relative_humidity = np.clip(76.0 - 1.55 * (ambient_shared - 11.0) + 0.65 * np.maximum(weather_front, 0.0) + RNG.normal(0.0, 2.4, PERIODS), 24.0, 99.0)
    wind_direction = (182.0 + np.cumsum(RNG.normal(0.0, 1.9, PERIODS)) + 26.0 * np.sin(2 * np.pi * steps / (6 * 24 * 11))) % 360.0

    return pd.DataFrame(
        {
            'Timestamp': TIMESTAMPS,
            'Hour_Of_Day': np.round(hour_of_day, 2),
            'Day_Index': np.round(days, 4),
            'Ambient_Base_Temperature_C': np.round(ambient_shared, 6),
            'Macro_Wind_Speed_mps': np.round(macro_wind, 6),
            'Wind_Gust_Base_mps': np.round(gust_base, 6),
            'Turbulence_Intensity': np.round(turbulence_intensity, 4),
            'Atmospheric_Pressure_hPa': atmospheric_pressure,
            'Relative_Humidity_pct': np.round(relative_humidity, 2),
            'Wind_Direction_deg': np.round(wind_direction, 2),
        }
    )


def build_degradation_series(turbine_id: str) -> np.ndarray:
    days = np.arange(PERIODS) / INTERVALS_PER_DAY
    degradation = np.zeros(PERIODS, dtype=float)

    if turbine_id == 'WT-01':
        return degradation

    if turbine_id == 'WT-02':
        ramp = np.clip(days - 30.0, 0.0, None)
        degradation = np.where(days < 30.0, 0.0, np.expm1(ramp / 4.2))
        return np.clip(degradation, 0.0, 18.0)

    if turbine_id == 'WT-03':
        drift = np.clip(days - 60.0, 0.0, None)
        degradation = 0.032 * drift
        return np.clip(degradation, 0.0, 1.2)

    raise ValueError(f'Unknown turbine_id: {turbine_id}')


def simulate_turbine(environment: pd.DataFrame, profile: TurbineProfile) -> pd.DataFrame:
    n_rows = len(environment)
    ambient = environment['Ambient_Base_Temperature_C'].to_numpy(dtype=float) + RNG.normal(0.0, profile.temp_micro_sigma, n_rows)
    macro_wind = environment['Macro_Wind_Speed_mps'].to_numpy(dtype=float)
    gust = shift_series(environment['Wind_Gust_Base_mps'].to_numpy(dtype=float), profile.gust_lag)
    turbulence = environment['Turbulence_Intensity'].to_numpy(dtype=float)
    pressure = environment['Atmospheric_Pressure_hPa'].to_numpy(dtype=float)
    humidity = environment['Relative_Humidity_pct'].to_numpy(dtype=float)
    wind_direction = environment['Wind_Direction_deg'].to_numpy(dtype=float)

    local_wake = np.clip(
        profile.wake_bias + profile.wake_scale * np.tanh((macro_wind - 8.0) / 4.0) + 0.08 * gust + RNG.normal(0.0, 0.05, n_rows),
        -0.8,
        1.8,
    )
    effective_wind = np.clip(macro_wind + gust + local_wake, 0.0, None)

    degradation = build_degradation_series(profile.turbine_id)
    rpm = np.zeros(n_rows, dtype=float)
    power_kw = np.zeros(n_rows, dtype=float)
    vibration = np.zeros(n_rows, dtype=float)
    bearing_temp = np.zeros(n_rows, dtype=float)
    nacelle_temp = np.zeros(n_rows, dtype=float)
    gearbox_temp = np.zeros(n_rows, dtype=float)
    oil_pressure = np.zeros(n_rows, dtype=float)
    blade_pitch = np.zeros(n_rows, dtype=float)
    generator_current = np.zeros(n_rows, dtype=float)
    yaw_error = np.zeros(n_rows, dtype=float)
    operational_state = np.full(n_rows, 'healthy', dtype=object)
    maintenance_flag = np.full(n_rows, 'none', dtype=object)

    for idx in range(n_rows):
        if profile.turbine_id == 'WT-02' and idx >= WT02_SEIZURE_INDEX:
            hours_since_seizure = (idx - WT02_SEIZURE_INDEX) / 6.0
            ambient_now = ambient[idx]
            rpm[idx] = 0.0
            power_kw[idx] = 0.0
            vibration[idx] = np.clip(0.35 + 0.15 * np.exp(-hours_since_seizure / 10.0) + RNG.normal(0.0, 0.03), 0.0, None)
            bearing_temp[idx] = ambient_now + 120.0 * np.exp(-hours_since_seizure / 18.0)
            nacelle_temp[idx] = ambient_now + 55.0 * np.exp(-hours_since_seizure / 14.0)
            gearbox_temp[idx] = ambient_now + 70.0 * np.exp(-hours_since_seizure / 16.0)
            oil_pressure[idx] = np.clip(0.55 + 0.08 * np.exp(-hours_since_seizure / 10.0), 0.15, 1.2)
            blade_pitch[idx] = 90.0
            generator_current[idx] = 0.0
            yaw_error[idx] = np.clip(11.0 + 2.0 * np.exp(-hours_since_seizure / 24.0), 0.0, 45.0)
            operational_state[idx] = 'offline'
            maintenance_flag[idx] = 'catastrophic seizure'
            continue

        effective_wind_now = effective_wind[idx]
        ambient_now = ambient[idx]
        degradation_now = degradation[idx]

        if effective_wind_now < CUT_IN_WIND_MPS:
            target_rpm = 0.0
            rpm[idx] = 0.0
        else:
            raw_target = 1.95 * (effective_wind_now - CUT_IN_WIND_MPS)
            raw_target *= max(0.0, 1.0 - 0.02 * degradation_now)
            target_rpm = np.clip(raw_target, 0.0, RATED_RPM)
            if idx == 0:
                rpm[idx] = target_rpm
            else:
                response = profile.rpm_response * max(0.6, 1.0 - 0.015 * degradation_now)
                rpm[idx] = rpm[idx - 1] + response * (target_rpm - rpm[idx - 1])
                if rpm[idx] < 0.0:
                    rpm[idx] = 0.0
                if rpm[idx] > RATED_RPM:
                    rpm[idx] = RATED_RPM

        power_kw[idx] = RATED_POWER_KW * profile.power_scale * np.minimum((rpm[idx] / RATED_RPM) ** 3, 1.0)
        if profile.turbine_id == 'WT-02' and idx >= WT02_DEGRADATION_START:
            power_kw[idx] *= max(0.5, 1.0 - 0.02 * degradation_now)
        if profile.turbine_id == 'WT-03' and idx >= WT03_DRIFT_START:
            power_kw[idx] *= max(0.7, 1.0 - 0.012 * degradation_now)

        vibration[idx] = np.clip(
            0.22 + profile.vibration_scale * (rpm[idx] ** 2) + 0.40 * degradation_now + RNG.normal(0.0, 0.045),
            0.0,
            None,
        )

        cooling_rate = profile.bearing_cooling * np.maximum(effective_wind_now, 0.0)
        bearing_temp[idx] = ambient_now + profile.bearing_temp_alpha * rpm[idx] + profile.bearing_temp_beta * degradation_now - cooling_rate + RNG.normal(0.0, 0.12)
        nacelle_temp[idx] = ambient_now + 4.8 + 0.16 * rpm[idx] + 0.55 * degradation_now + RNG.normal(0.0, 0.18)
        gearbox_temp[idx] = nacelle_temp[idx] + 5.6 + 0.08 * rpm[idx] + 0.9 * degradation_now + RNG.normal(0.0, 0.20)
        oil_pressure[idx] = np.clip(3.35 - 0.025 * rpm[idx] - 0.08 * degradation_now + RNG.normal(0.0, 0.03), 0.35, 4.2)
        blade_pitch[idx] = np.clip(2.0 + 0.85 * np.maximum(effective_wind_now - 7.0, 0.0) + 1.25 * degradation_now + RNG.normal(0.0, 0.18), 0.0, 90.0)
        generator_current[idx] = np.clip(power_kw[idx] / 7.9 + RNG.normal(0.0, 2.4), 0.0, 180.0)
        yaw_error[idx] = np.clip(np.abs(RNG.normal(0.7 + 0.22 * degradation_now + 0.03 * gust[idx], 0.35)), 0.0, 35.0)

        if profile.turbine_id == 'WT-02':
            if idx >= WT02_DEGRADATION_START:
                operational_state[idx] = 'degrading'
                maintenance_flag[idx] = 'bearing degradation'
            if idx >= WT02_SEIZURE_INDEX:
                operational_state[idx] = 'offline'
                maintenance_flag[idx] = 'catastrophic seizure'
        elif profile.turbine_id == 'WT-03' and idx >= WT03_DRIFT_START:
            operational_state[idx] = 'drifting'
            maintenance_flag[idx] = 'early drift'

    if profile.turbine_id == 'WT-02':
        seizure_idx = WT02_SEIZURE_INDEX
        bearing_temp[seizure_idx] = ambient[seizure_idx] + 165.0
        vibration[seizure_idx] = max(vibration[seizure_idx], 8.5)
        maintenance_flag[seizure_idx] = 'catastrophic seizure'
        operational_state[seizure_idx] = 'offline'

    if profile.turbine_id == 'WT-03':
        vibration = np.clip(vibration, 0.0, 7.5)
        bearing_temp = np.clip(bearing_temp, -10.0, 105.0)
    elif profile.turbine_id == 'WT-01':
        vibration = np.clip(vibration, 0.0, 6.0)
        bearing_temp = np.clip(bearing_temp, -10.0, 90.0)

    turbine_df = pd.DataFrame(
        {
            'Timestamp': environment['Timestamp'],
            'Turbine_ID': profile.turbine_id,
            'Ambient_Temperature_C': np.round(ambient, 6),
            'Macro_Wind_Speed_mps': np.round(macro_wind, 6),
            'Wind_Speed_mps': np.round(macro_wind, 6),
            'Wind_Gust_mps': np.round(gust, 6),
            'Local_Wake_Turbulence_mps': np.round(local_wake, 6),
            'Effective_Wind_Speed_mps': np.round(effective_wind, 6),
            'Turbulence_Intensity': np.round(turbulence, 4),
            'Atmospheric_Pressure_hPa': pressure,
            'Relative_Humidity_pct': np.round(humidity, 2),
            'Wind_Direction_deg': np.round(wind_direction, 2),
            'Degradation_Factor': np.round(degradation, 6),
            'Bearing_Temperature_C': np.round(bearing_temp, 6),
            'Vibration_mm_s': np.round(vibration, 6),
            'RPM': np.round(rpm, 6),
            'Power_kW': np.round(power_kw, 6),
            'Turbine_Load_Percent': np.round(np.clip(100.0 * power_kw / RATED_POWER_KW, 0.0, 100.0), 4),
            'Yaw_Error_deg': np.round(yaw_error, 6),
            'Blade_Pitch_Deg': np.round(blade_pitch, 6),
            'Generator_Current_A': np.round(generator_current, 6),
            'Nacelle_Temperature_C': np.round(nacelle_temp, 6),
            'Oil_Pressure_bar': np.round(oil_pressure, 6),
            'Gearbox_Temperature_C': np.round(gearbox_temp, 6),
            'Operational_State': operational_state,
            'Maintenance_Flag': maintenance_flag,
            'Grid_Priority': 'Standard',
        }
    )

    return turbine_df


def build_clean_dataset() -> pd.DataFrame:
    environment = build_shared_environment()
    frames = [simulate_turbine(environment, profile) for profile in TURBINE_PROFILES.values()]
    clean_df = pd.concat(frames, ignore_index=True)
    clean_df = clean_df.sort_values(['Turbine_ID', 'Timestamp']).reset_index(drop=True)
    return clean_df


def inject_messy_data(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    messy_df = clean_df.copy(deep=True)
    injected = {
        'flatline_start': None,
        'flatline_window': None,
        'rpm_nulls': 0,
        'power_nulls': 0,
        'temp_spikes': [],
        'power_spikes': [],
    }

    wt01_indices = np.flatnonzero((messy_df['Turbine_ID'] == 'WT-01').to_numpy())
    flatline_window = 24 * 6
    flatline_start = int(RNG.integers(0, len(wt01_indices) - flatline_window + 1))
    flatline_rows = wt01_indices[flatline_start:flatline_start + flatline_window]
    flatline_value = messy_df.iloc[flatline_rows[0]]['Vibration_mm_s']
    messy_df.loc[flatline_rows, 'Vibration_mm_s'] = flatline_value
    injected['flatline_start'] = messy_df.iloc[flatline_rows[0]]['Timestamp']
    injected['flatline_window'] = flatline_window

    for column in ['RPM', 'Power_kW']:
        mask = RNG.random(len(messy_df)) < 0.03
        messy_df.loc[mask, column] = np.nan
        if column == 'RPM':
            injected['rpm_nulls'] = int(mask.sum())
        else:
            injected['power_nulls'] = int(mask.sum())

    temp_spike_indices = RNG.choice(len(messy_df), size=3, replace=False)
    power_spike_indices = RNG.choice(len(messy_df), size=3, replace=False)
    messy_df.loc[temp_spike_indices, 'Bearing_Temperature_C'] = 999.0
    messy_df.loc[power_spike_indices, 'Power_kW'] = -500.0
    injected['temp_spikes'] = messy_df.loc[temp_spike_indices, ['Timestamp', 'Turbine_ID']].to_dict('records')
    injected['power_spikes'] = messy_df.loc[power_spike_indices, ['Timestamp', 'Turbine_ID']].to_dict('records')

    return messy_df, injected


def print_correlation_summary(df: pd.DataFrame) -> None:
    relevant = ['RPM', 'Power_kW', 'Vibration_mm_s', 'Bearing_Temperature_C']
    print('\nCorrelation matrices by turbine (rows with missing values removed):')
    for turbine_id, group in df.groupby('Turbine_ID', sort=True):
        analysis_frame = group.dropna(subset=relevant).copy()
        analysis_frame = analysis_frame.assign(RPM_Cubed=analysis_frame['RPM'] ** 3)
        matrix = analysis_frame[['RPM', 'RPM_Cubed', 'Power_kW', 'Vibration_mm_s', 'Bearing_Temperature_C']].corr().round(3)
        print(f'\n{turbine_id}')
        print(matrix.to_string())


def main() -> None:
    clean_df = build_clean_dataset()
    messy_df, injected = inject_messy_data(clean_df)

    clean_df.to_csv(CLEAN_FILE, index=False)
    messy_df.to_csv(MESSY_FILE, index=False)
    legacy_written = False
    try:
        messy_df.to_csv(LEGACY_FILE, index=False)
        legacy_written = True
    except PermissionError:
        pass

    print(f'Wrote clean dataset to {CLEAN_FILE.resolve()}')
    print(f'Wrote student-facing dataset to {MESSY_FILE.resolve()}')
    if legacy_written:
        print(f'Wrote compatibility copy to {LEGACY_FILE.resolve()}')
    else:
        print(f'Skipped compatibility copy at {LEGACY_FILE.resolve()} because the file is locked')
    print(f'Rows per turbine: {len(clean_df) // 3} | Total rows: {len(clean_df)}')
    print(
        'Injected anomalies: '
        f"WT-01 flatline starting {injected['flatline_start']} for {injected['flatline_window']} rows, "
        f"RPM nulls={injected['rpm_nulls']}, Power nulls={injected['power_nulls']}, "
        f"temperature spikes={len(injected['temp_spikes'])}, power spikes={len(injected['power_spikes'])}"
    )
    print_correlation_summary(messy_df)


if __name__ == '__main__':
    main()