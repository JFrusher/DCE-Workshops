from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_FILE = Path(__file__).resolve().parent / 'wind_turbine_telemetry.csv'
RNG = np.random.default_rng(42)
START_TIMESTAMP = pd.Timestamp('2026-01-01 00:00:00')
PERIODS = 24 * 90
TIMESTAMPS = pd.date_range(start=START_TIMESTAMP, periods=PERIODS, freq='h')


MONTH1_HOURS = 31 * 24
MONTH2_HOURS = 28 * 24
MONTH3_HOURS = 31 * 24
MONTH1_END = MONTH1_HOURS
MONTH2_END = MONTH1_HOURS + MONTH2_HOURS


TURBINES = {
    'WT-01': {
        'Grid_Priority': 'Standard',
        'Wind_Capture_Adjustment': -0.1,
        'Load_Sensitivity': 0.85,
        'Vibration_Baseline': 1.75,
        'Temp_Baseline': 47.5,
        'RPM_Baseline': 13.4,
    },
    'WT-02': {
        'Grid_Priority': 'Standard',
        'Wind_Capture_Adjustment': 0.05,
        'Load_Sensitivity': 1.0,
        'Vibration_Baseline': 1.9,
        'Temp_Baseline': 48.5,
        'RPM_Baseline': 13.6,
    },
    'WT-03': {
        'Grid_Priority': 'Critical Infrastructure',
        'Wind_Capture_Adjustment': 0.08,
        'Load_Sensitivity': 0.92,
        'Vibration_Baseline': 1.8,
        'Temp_Baseline': 47.8,
        'RPM_Baseline': 13.5,
    },
}


def build_environment(timestamp_index: pd.DatetimeIndex) -> pd.DataFrame:
    hour_of_day = timestamp_index.hour.to_numpy()
    day_of_year = timestamp_index.dayofyear.to_numpy()

    seasonal_temp = 12 + 3.2 * np.sin(2 * np.pi * (day_of_year - 15) / 365)
    daily_temp = 2.2 * np.sin(2 * np.pi * (hour_of_day - 14) / 24)
    weather_noise = RNG.normal(0, 0.8, len(timestamp_index))
    ambient_temp = np.clip(seasonal_temp + daily_temp + weather_noise, -2, 22)

    base_wind = 7.2 + 1.3 * np.sin(2 * np.pi * (hour_of_day - 4) / 24)
    seasonal_wind = 0.8 * np.sin(2 * np.pi * (day_of_year + 40) / 365)
    gust_noise = RNG.normal(0, 0.9, len(timestamp_index))
    autocorrelated = np.zeros(len(timestamp_index), dtype=float)
    for idx in range(1, len(timestamp_index)):
        autocorrelated[idx] = 0.72 * autocorrelated[idx - 1] + gust_noise[idx]

    wind_speed = np.clip(base_wind + seasonal_wind + autocorrelated, 1.2, 18.5)

    storm_centres = [320, 980, 1540, 1910]
    for centre in storm_centres:
        window = slice(max(0, centre - 10), min(len(wind_speed), centre + 14))
        wind_speed[window] = np.clip(wind_speed[window] + RNG.uniform(1.2, 3.5), 1.2, 20)

    return pd.DataFrame(
        {
            'Timestamp': timestamp_index,
            'Hour_Of_Day': hour_of_day,
            'Day_Of_Year': day_of_year,
            'Ambient_Temperature_C': np.round(ambient_temp, 2),
            'Wind_Speed_mps': np.round(wind_speed, 2),
        }
    )


def simulate_shared_operating_load(wind_speed: np.ndarray, adjustment: float) -> np.ndarray:
    load = 38 + (wind_speed * 5.4) + adjustment * 2.5
    load += 3.0 * np.sin(2 * np.pi * (np.arange(len(wind_speed)) % 24) / 24)
    return np.clip(load, 20, 100)


def add_benign_dropout(values: np.ndarray, dropout_probability: float) -> np.ndarray:
    result = values.copy()
    dropout_mask = RNG.random(len(result)) < dropout_probability
    result[dropout_mask] = np.nan
    return result


def build_wt01(environment: pd.DataFrame) -> pd.DataFrame:
    config = TURBINES['WT-01']
    wind_speed = environment['Wind_Speed_mps'].to_numpy()
    ambient_temp = environment['Ambient_Temperature_C'].to_numpy()
    load = simulate_shared_operating_load(wind_speed, config['Wind_Capture_Adjustment'])
    load_factor = load / 100.0

    temperature = ambient_temp + 34.4 + (5.5 * load_factor) + RNG.normal(0, 0.55, len(environment))
    vibration = config['Vibration_Baseline'] + (0.55 * load_factor) + RNG.normal(0, 0.08, len(environment))
    rpm = config['RPM_Baseline'] + (0.7 * (load_factor - 0.45)) + RNG.normal(0, 0.11, len(environment))
    power_kw = 980 + (10.5 * wind_speed**2) + RNG.normal(0, 18, len(environment))
    yaw_error = np.abs(RNG.normal(0.9, 0.45, len(environment)))
    gearbox_temp = temperature + 9.0 + RNG.normal(0, 0.7, len(environment))

    maintenance_flag = np.where(environment['Timestamp'].dt.month.eq(2) & environment['Timestamp'].dt.day.eq(14), 'inspection', 'none')
    status = np.full(len(environment), 'normal', dtype=object)

    temperature = np.clip(temperature, 45.0, 55.0)
    vibration = np.clip(vibration, 1.5, 2.5)
    rpm = np.clip(rpm, 12.0, 15.0)
    power_kw = np.clip(power_kw, 800, 2350)

    temperature = add_benign_dropout(temperature, 0.004)
    vibration = add_benign_dropout(vibration, 0.003)

    return pd.DataFrame(
        {
            'Timestamp': environment['Timestamp'],
            'Turbine_ID': 'WT-01',
            'Ambient_Temperature_C': np.round(ambient_temp, 2),
            'Wind_Speed_mps': np.round(wind_speed, 2),
            'Bearing_Temperature_C': np.round(temperature, 2),
            'Vibration_mm_s': np.round(vibration, 2),
            'RPM': np.round(rpm, 2),
            'Power_kW': np.round(power_kw, 1),
            'Yaw_Error_deg': np.round(yaw_error, 2),
            'Gearbox_Temperature_C': np.round(gearbox_temp, 2),
            'Turbine_Load_Percent': np.round(load, 1),
            'Operational_State': status,
            'Maintenance_Flag': maintenance_flag,
            'Grid_Priority': config['Grid_Priority'],
        }
    )


def build_wt02(environment: pd.DataFrame) -> pd.DataFrame:
    config = TURBINES['WT-02']
    wind_speed = environment['Wind_Speed_mps'].to_numpy()
    ambient_temp = environment['Ambient_Temperature_C'].to_numpy()
    load = simulate_shared_operating_load(wind_speed, config['Wind_Capture_Adjustment'])
    load_factor = load / 100.0

    temperature = np.empty(len(environment), dtype=float)
    vibration = np.empty(len(environment), dtype=float)
    rpm = np.empty(len(environment), dtype=float)
    power_kw = np.empty(len(environment), dtype=float)
    yaw_error = np.empty(len(environment), dtype=float)
    gearbox_temp = np.empty(len(environment), dtype=float)
    operational_state = np.full(len(environment), 'normal', dtype=object)

    month1_index = np.arange(MONTH1_END)
    month1_progress = month1_index / max(MONTH1_END - 1, 1)
    degradation_curve = np.clip((month1_progress - 0.35) / 0.65, 0, 1)

    temperature[:MONTH1_END] = ambient_temp[:MONTH1_END] + 36 + (46 * degradation_curve) + RNG.normal(0, 1.0, MONTH1_END)
    vibration[:MONTH1_END] = 1.95 + (7.05 * degradation_curve) + RNG.normal(0, 0.16, MONTH1_END)
    rpm[:MONTH1_END] = 14.2 - (11.8 * degradation_curve) + RNG.normal(0, 0.28, MONTH1_END)
    power_kw[:MONTH1_END] = np.clip(950 + (wind_speed[:MONTH1_END] ** 2.15) * 10 - (260 * degradation_curve), 0, 2400)
    yaw_error[:MONTH1_END] = np.abs(RNG.normal(1.2 + 2.3 * degradation_curve, 0.65, MONTH1_END))
    gearbox_temp[:MONTH1_END] = temperature[:MONTH1_END] + 11.0 + (4.0 * degradation_curve) + RNG.normal(0, 0.85, MONTH1_END)

    spike_mask = (month1_progress > 0.78) & (month1_progress < 0.92)
    temperature[:MONTH1_END][spike_mask] += RNG.uniform(4.0, 10.0, spike_mask.sum())
    vibration[:MONTH1_END][spike_mask] += RNG.uniform(0.4, 1.4, spike_mask.sum())
    temperature[:MONTH1_END] = np.clip(temperature[:MONTH1_END], 44.0, 95.0)
    vibration[:MONTH1_END] = np.clip(vibration[:MONTH1_END], 1.4, 9.0)

    failure_start = MONTH1_END - 5
    temperature[failure_start:MONTH1_END] = np.linspace(88, 95, MONTH1_END - failure_start) + RNG.normal(0, 0.4, MONTH1_END - failure_start)
    vibration[failure_start:MONTH1_END] = np.linspace(7.6, 9.0, MONTH1_END - failure_start) + RNG.normal(0, 0.08, MONTH1_END - failure_start)
    rpm[failure_start:MONTH1_END] = np.linspace(6.0, 0.0, MONTH1_END - failure_start) + RNG.normal(0, 0.05, MONTH1_END - failure_start)
    power_kw[failure_start:MONTH1_END] = np.linspace(320, 0, MONTH1_END - failure_start)
    yaw_error[failure_start:MONTH1_END] = np.linspace(8.0, 18.0, MONTH1_END - failure_start)
    gearbox_temp[failure_start:MONTH1_END] = temperature[failure_start:MONTH1_END] + 14.0
    operational_state[failure_start:MONTH1_END] = 'critical failure onset'
    temperature[MONTH1_END - 1] = 95.0
    vibration[MONTH1_END - 1] = 9.0

    temperature[MONTH1_END:] = np.nan
    vibration[MONTH1_END:] = np.nan
    rpm[MONTH1_END:] = 0.0
    power_kw[MONTH1_END:] = 0.0
    yaw_error[MONTH1_END:] = np.nan
    gearbox_temp[MONTH1_END:] = np.nan
    operational_state[MONTH1_END:] = 'offline after bearing seizure'

    sensor_dropout_mask = (RNG.random(len(environment)) < 0.008) & (np.arange(len(environment)) < MONTH1_END)
    vibration[sensor_dropout_mask] = np.nan

    return pd.DataFrame(
        {
            'Timestamp': environment['Timestamp'],
            'Turbine_ID': 'WT-02',
            'Ambient_Temperature_C': np.round(ambient_temp, 2),
            'Wind_Speed_mps': np.round(wind_speed, 2),
            'Bearing_Temperature_C': np.round(temperature, 2),
            'Vibration_mm_s': np.round(vibration, 2),
            'RPM': np.round(rpm, 2),
            'Power_kW': np.round(power_kw, 1),
            'Yaw_Error_deg': np.round(yaw_error, 2),
            'Gearbox_Temperature_C': np.round(gearbox_temp, 2),
            'Turbine_Load_Percent': np.round(load, 1),
            'Operational_State': operational_state,
            'Maintenance_Flag': np.where(np.arange(len(environment)) == failure_start, 'catastrophic failure', 'none'),
            'Grid_Priority': config['Grid_Priority'],
        }
    )


def build_wt03(environment: pd.DataFrame) -> pd.DataFrame:
    config = TURBINES['WT-03']
    wind_speed = environment['Wind_Speed_mps'].to_numpy()
    ambient_temp = environment['Ambient_Temperature_C'].to_numpy()
    load = simulate_shared_operating_load(wind_speed, config['Wind_Capture_Adjustment'])
    load_factor = load / 100.0

    temperature = np.empty(len(environment), dtype=float)
    vibration = np.empty(len(environment), dtype=float)
    rpm = np.empty(len(environment), dtype=float)
    power_kw = np.empty(len(environment), dtype=float)
    yaw_error = np.empty(len(environment), dtype=float)
    gearbox_temp = np.empty(len(environment), dtype=float)
    operational_state = np.full(len(environment), 'normal', dtype=object)

    healthy_span = MONTH2_END
    healthy_noise_temp = RNG.normal(0, 0.6, healthy_span)
    healthy_noise_vibration = RNG.normal(0, 0.08, healthy_span)
    healthy_noise_rpm = RNG.normal(0, 0.12, healthy_span)

    temperature[:healthy_span] = ambient_temp[:healthy_span] + 35.0 + (4.7 * load_factor[:healthy_span]) + healthy_noise_temp
    vibration[:healthy_span] = 1.8 + (0.45 * load_factor[:healthy_span]) + healthy_noise_vibration
    rpm[:healthy_span] = 13.45 + (0.5 * (load_factor[:healthy_span] - 0.45)) + healthy_noise_rpm
    power_kw[:healthy_span] = np.clip(960 + (wind_speed[:healthy_span] ** 2.1) * 10.5 + RNG.normal(0, 15, healthy_span), 0, 2400)
    yaw_error[:healthy_span] = np.abs(RNG.normal(0.95, 0.4, healthy_span))
    gearbox_temp[:healthy_span] = temperature[:healthy_span] + 9.5 + RNG.normal(0, 0.75, healthy_span)

    month3_start = MONTH2_END
    month3_hours = MONTH3_HOURS
    month3_progress = np.linspace(0, 1, month3_hours)
    temperature[month3_start:] = ambient_temp[month3_start:] + 33.0 + (29.0 * month3_progress) + RNG.normal(0, 0.9, month3_hours)
    vibration[month3_start:] = 1.85 + (4.15 * month3_progress) + RNG.normal(0, 0.12, month3_hours)
    rpm[month3_start:] = 13.9 - (2.0 * month3_progress) + RNG.normal(0, 0.18, month3_hours)
    power_kw[month3_start:] = np.clip(980 + (wind_speed[month3_start:] ** 2.1) * 9.6 - (160 * month3_progress), 0, 2400)
    yaw_error[month3_start:] = np.abs(RNG.normal(1.1 + 4.0 * month3_progress, 0.45, month3_hours))
    gearbox_temp[month3_start:] = temperature[month3_start:] + 10.0 + (2.8 * month3_progress) + RNG.normal(0, 0.8, month3_hours)

    warning_window = slice(month3_start + 96, month3_start + 180)
    temperature[warning_window] += np.linspace(0, 6.0, warning_window.stop - warning_window.start)
    vibration[warning_window] += np.linspace(0, 0.9, warning_window.stop - warning_window.start)

    temperature = np.clip(temperature, 45.0, 82.0)
    vibration = np.clip(vibration, 1.5, 6.0)
    rpm = np.clip(rpm, 11.4, 15.0)
    power_kw = np.clip(power_kw, 750, 2400)
    temperature[-1] = 82.0
    vibration[-1] = 6.0

    return pd.DataFrame(
        {
            'Timestamp': environment['Timestamp'],
            'Turbine_ID': 'WT-03',
            'Ambient_Temperature_C': np.round(ambient_temp, 2),
            'Wind_Speed_mps': np.round(wind_speed, 2),
            'Bearing_Temperature_C': np.round(temperature, 2),
            'Vibration_mm_s': np.round(vibration, 2),
            'RPM': np.round(rpm, 2),
            'Power_kW': np.round(power_kw, 1),
            'Yaw_Error_deg': np.round(yaw_error, 2),
            'Gearbox_Temperature_C': np.round(gearbox_temp, 2),
            'Turbine_Load_Percent': np.round(load, 1),
            'Operational_State': operational_state,
            'Maintenance_Flag': np.where(np.arange(len(environment)) == month3_start + 96, 'pre-failure anomaly', 'none'),
            'Grid_Priority': config['Grid_Priority'],
        }
    )


def build_telemetry_dataset() -> pd.DataFrame:
    environment = build_environment(TIMESTAMPS)
    wt01 = build_wt01(environment)
    wt02 = build_wt02(environment)
    wt03 = build_wt03(environment)

    telemetry_df = pd.concat([wt01, wt02, wt03], ignore_index=True)
    telemetry_df = telemetry_df.sort_values(['Turbine_ID', 'Timestamp']).reset_index(drop=True)
    return telemetry_df


def main() -> None:
    telemetry_df = build_telemetry_dataset()
    telemetry_df.to_csv(OUTPUT_FILE, index=False)

    print(f'Wrote {len(telemetry_df)} rows to {OUTPUT_FILE.resolve()}')
    summary_columns = ['Ambient_Temperature_C', 'Wind_Speed_mps', 'Bearing_Temperature_C', 'Vibration_mm_s', 'RPM', 'Power_kW']
    print(telemetry_df.groupby('Turbine_ID')[summary_columns].agg(['min', 'max', 'mean']).round(2))
    print('\nMissing values by turbine:')
    print(telemetry_df.groupby('Turbine_ID')[['Bearing_Temperature_C', 'Vibration_mm_s', 'Yaw_Error_deg', 'Gearbox_Temperature_C']].apply(lambda frame: frame.isna().sum()).to_string())


if __name__ == '__main__':
    main()
