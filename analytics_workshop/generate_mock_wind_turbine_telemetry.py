from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_FILE = Path(__file__).resolve().parent / 'wind_turbine_telemetry.csv'
RNG = np.random.default_rng(42)
START_TIMESTAMP = pd.Timestamp('2026-01-01 00:00:00')
PERIODS = 24 * 90
TIMESTAMPS = pd.date_range(start=START_TIMESTAMP, periods=PERIODS, freq='h')


TURBINES = {
    'WT-01': {
        'Grid_Priority': 'Standard',
    },
    'WT-02': {
        'Grid_Priority': 'Standard',
    },
    'WT-03': {
        'Grid_Priority': 'Critical Infrastructure',
    },
}

MONTH1_HOURS = 31 * 24
MONTH2_HOURS = 28 * 24
MONTH3_HOURS = 31 * 24
MONTH1_END = MONTH1_HOURS
MONTH2_END = MONTH1_HOURS + MONTH2_HOURS


def make_healthy_signals(base_temp_low, base_temp_high, base_vib_low, base_vib_high, base_rpm_low, base_rpm_high, size):
    return (
        RNG.uniform(base_temp_low, base_temp_high, size=size),
        RNG.uniform(base_vib_low, base_vib_high, size=size),
        RNG.uniform(base_rpm_low, base_rpm_high, size=size),
    )


def build_wt01(timestamp_index):
    size = len(timestamp_index)
    temperature, vibration, rpm = make_healthy_signals(45, 55, 1.5, 2.5, 12, 15, size)

    return pd.DataFrame(
        {
            'Timestamp': timestamp_index,
            'Turbine_ID': 'WT-01',
            'Bearing_Temperature_C': np.round(temperature, 2),
            'Vibration_mm_s': np.round(vibration, 2),
            'RPM': np.round(rpm, 2),
            'Grid_Priority': 'Standard',
        }
    )


def build_wt02(timestamp_index):
    temperature = np.empty(len(timestamp_index), dtype=float)
    vibration = np.empty(len(timestamp_index), dtype=float)
    rpm = np.empty(len(timestamp_index), dtype=float)

    month1_progress = np.linspace(0, 1, MONTH1_END)
    temperature[:MONTH1_END] = 50 + (45 * month1_progress) + RNG.normal(0, 1.5, MONTH1_END)
    vibration[:MONTH1_END] = 2 + (7 * month1_progress) + RNG.normal(0, 0.2, MONTH1_END)
    rpm[:MONTH1_END] = 14 - (14 * month1_progress) + RNG.normal(0, 0.25, MONTH1_END)

    temperature[:MONTH1_END] = np.clip(temperature[:MONTH1_END], 44, 95)
    vibration[:MONTH1_END] = np.clip(vibration[:MONTH1_END], 1.4, 9.0)
    rpm[:MONTH1_END] = np.clip(rpm[:MONTH1_END], 0, 15)

    temperature[MONTH1_END:] = 22.0
    vibration[MONTH1_END:] = np.nan
    rpm[MONTH1_END:] = 0.0

    return pd.DataFrame(
        {
            'Timestamp': timestamp_index,
            'Turbine_ID': 'WT-02',
            'Bearing_Temperature_C': np.round(temperature, 2),
            'Vibration_mm_s': np.round(vibration, 2),
            'RPM': np.round(rpm, 2),
            'Grid_Priority': 'Standard',
        }
    )


def build_wt03(timestamp_index):
    size = len(timestamp_index)
    temperature = np.empty(size, dtype=float)
    vibration = np.empty(size, dtype=float)
    rpm = np.empty(size, dtype=float)

    healthy_temp, healthy_vib, healthy_rpm = make_healthy_signals(45, 55, 1.5, 2.5, 12, 15, MONTH2_END)
    temperature[:MONTH2_END] = healthy_temp
    vibration[:MONTH2_END] = healthy_vib
    rpm[:MONTH2_END] = healthy_rpm

    month3_progress = np.linspace(0, 1, MONTH3_HOURS)
    month3_start = MONTH2_END
    temperature[month3_start:] = 50 + (32 * month3_progress) + RNG.normal(0, 1.0, MONTH3_HOURS)
    vibration[month3_start:] = 2 + (4 * month3_progress) + RNG.normal(0, 0.15, MONTH3_HOURS)
    rpm[month3_start:] = 14 - (2 * month3_progress) + RNG.normal(0, 0.2, MONTH3_HOURS)

    temperature[month3_start:] = np.clip(temperature[month3_start:], 50, 82)
    vibration[month3_start:] = np.clip(vibration[month3_start:], 2, 6.0)
    rpm[month3_start:] = np.clip(rpm[month3_start:], 10.5, 15)

    return pd.DataFrame(
        {
            'Timestamp': timestamp_index,
            'Turbine_ID': 'WT-03',
            'Bearing_Temperature_C': np.round(temperature, 2),
            'Vibration_mm_s': np.round(vibration, 2),
            'RPM': np.round(rpm, 2),
            'Grid_Priority': 'Critical Infrastructure',
        }
    )


def build_telemetry_dataset() -> pd.DataFrame:
    wt01 = build_wt01(TIMESTAMPS)
    wt02 = build_wt02(TIMESTAMPS)
    wt03 = build_wt03(TIMESTAMPS)

    telemetry_df = pd.concat([wt01, wt02, wt03], ignore_index=True)
    telemetry_df = telemetry_df.sort_values(['Turbine_ID', 'Timestamp']).reset_index(drop=True)
    return telemetry_df


def main() -> None:
    telemetry_df = build_telemetry_dataset()
    telemetry_df.to_csv(OUTPUT_FILE, index=False)

    print(f'Wrote {len(telemetry_df)} rows to {OUTPUT_FILE.resolve()}')
    print(telemetry_df.groupby('Turbine_ID')[['Bearing_Temperature_C', 'Vibration_mm_s', 'RPM']].agg(['min', 'max', 'mean']).round(2))


if __name__ == '__main__':
    main()
