from pathlib import Path
import sys, pandas as pd, numpy as np
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
root = Path(r'c:\Users\hp\OneDrive - University of Southampton\Year 3\2 DCE\DCE Code\ML+AI_workshop')
sys.path.insert(0, str(root / 'Hidden'))
import presentation_emulator as pe
cfg = pe.EmulatorConfig()
emu = pe.PresentationPumpTelemetryEmulator(cfg)
df = emu.generate(root / cfg.output_name)
work = df.copy()
work[['raw_vibration','drifted_pressure','motor_temp','fluid_temp']] = work[['raw_vibration','drifted_pressure','motor_temp','fluid_temp']].ffill().bfill()
work['cleaned_vibration'] = savgol_filter(work['raw_vibration'].to_numpy(), 15, 2)
work['calibrated_pressure'] = work['drifted_pressure'].to_numpy() - (cfg.drift_rate_kpa_per_hour * work['time_hours'].to_numpy())
work['vibration_rms'] = pd.Series(work['cleaned_vibration']).rolling(61, min_periods=1, center=True).apply(lambda a: float(np.sqrt(np.mean(np.square(a)))), raw=True).bfill().ffill().to_numpy()
work['vibration_trend'] = pd.Series(work['cleaned_vibration']).rolling(31, min_periods=2, center=True).apply(lambda a: float((a[-1]-a[0])/max(1, len(a)-1)), raw=True).bfill().ffill().to_numpy()
work['pressure_reserve'] = work['calibrated_pressure'].to_numpy() - pd.Series(work['calibrated_pressure']).rolling(91, min_periods=1, center=True).mean().bfill().ffill().to_numpy()
model_df = work.dropna(subset=['pump_efficiency'])
train_df, test_df = train_test_split(model_df, test_size=0.3, random_state=42, shuffle=True)
raw_features = ['raw_vibration','drifted_pressure','motor_temp']
engineered_features = ['vibration_rms','vibration_trend','pressure_reserve','motor_temp']
baseline = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1).fit(train_df[raw_features], train_df['pump_efficiency'])
eng = RandomForestRegressor(n_estimators=160, random_state=42, n_jobs=-1).fit(train_df[engineered_features], train_df['pump_efficiency'])
raw = r2_score(test_df['pump_efficiency'], baseline.predict(test_df[raw_features]))
eng_r2 = r2_score(test_df['pump_efficiency'], eng.predict(test_df[engineered_features]))
Path(r'c:\Users\hp\OneDrive - University of Southampton\Year 3\2 DCE\DCE Code\mlai_scores.txt').write_text(f'raw={raw:.6f}\neng={eng_r2:.6f}\n', encoding='utf-8')
print('done')
