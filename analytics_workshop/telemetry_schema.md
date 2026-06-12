# Wind Turbine Telemetry Schema

## Dataset Grain

- One row per turbine per 10-minute timestamp.
- Three turbines are simulated: `WT-01`, `WT-02`, and `WT-03`.

## Column Dictionary

| Column | Type | Description | How it is used |
|---|---|---|---|
| `Timestamp` | datetime | 10-minute observation time for the record. | Used for time-series plots, trend fitting, rolling averages, and failure timing. |
| `Turbine_ID` | categorical | Turbine identifier, typically `WT-01`, `WT-02`, or `WT-03`. | Used to separate the turbine data |
| `Ambient_Temperature_C` | float | Local ambient temperature around the turbine nacelle area. | Environmental Variable |
| `Macro_Wind_Speed_mps` | float | Shared regional wind speed before local wake and gust effects. | The broader wind environment. |
| `Wind_Speed_mps` | float | Working wind speed experienced by the turbine after local effects. | Environmental Variable |
| `Wind_Gust_mps` | float | Short-term gust component added to the wind field. | Used to explore links with RPM and transient loading. |
| `Local_Wake_Turbulence_mps` | float | Local wake-induced turbulence component around each turbine. | Helps explain turbine-to-turbine differences in effective wind loading. |
| `Effective_Wind_Speed_mps` | float | Combined wind speed after macro wind, gust, and local wake effects. | Operating wind variable. |
| `Turbulence_Intensity` | float | Dimensionless turbulence metric derived from wind and gust variability. | Operating wind variable |
| `Atmospheric_Pressure_hPa` | float | Local atmospheric pressure. | Weather context variable |
| `Relative_Humidity_pct` | float | Relative humidity percentage. | Weather context variable |
| `Wind_Direction_deg` | float | Wind direction in degrees from 0 to 360. | Used in yaw error plots and directiona
| `Bearing_Temperature_C` | float | Bearing temperature in degrees Celsius. | Bearing health signal. |
| `Vibration_mm_s` | float | Vibration magnitude in mm/s. | Key condition-monitoring signal |
| `RPM` | float | Rotor rotational speed. | Sensor variable. |
| `Power_kW` | float | Electrical power output in kilowatts. | Sensor variable. |
| `Turbine_Load_Percent` | float | Power output scaled as a percentage of rated power. | Normalised load view for interpretation and comparison. |
| `Yaw_Error_deg` | float | Difference between wind direction and turbine yaw alignment. | Used in yaw misalignment exploration. |
| `Blade_Pitch_Deg` | float | Blade pitch angle in degrees. | Used in pitch-vs-power relationships and curtailment checks. |
| `Generator_Current_A` | float | Generator current in amps. | Electrical response variable that follows output changes. |
| `Nacelle_Temperature_C` | float | Nacelle temperature in degrees Celsius. | Secondary thermal indicator. |
| `Oil_Pressure_bar` | float | Hydraulic or lubrication oil pressure in bar. | Mechanical health indicator for drivetrain condition. |
| `Gearbox_Temperature_C` | float | Gearbox temperature in degrees Celsius. | Secondary drivetrain thermal health indicator. |
| `Operational_State` | categorical | Simulated operating mode such as `healthy`, `degrading`, `drifting`, or `offline`. | Useful for labels, filtering, and interpretation. |
| `Maintenance_Flag` | categorical | Human-readable maintenance condition such as `none`, `bearing degradation`, `catastrophic seizure`, or `early drift`. | Connects telemetry patterns to maintenance meaning. |
| `Grid_Priority` | categorical | Simplified grid dispatch label. | Included as a contextual field in exploratory charts. |

## Derived Columns Used in the Notebooks

These are created during analysis rather than written by the generator.

| Column | Type | Description |
|---|---|---|
| `Failure_State` | categorical | Notebook-created label used to mark rows with missing bearing temperature or vibration as sensor dropout / failure. |
| `Vibration_per_RPM` | float | Vibration normalised by RPM, used to compare roughness at different operating speeds. |
| `Vibration_per_RPM_Rolling` | float | Rolling mean of vibration per RPM for trend inspection. |
| `Day_Index` | float | Day offset from the start of the time series, used for linear trend fitting. |
| `Trend_Fit` | float | Fitted line for the WT-03 vibration-per-RPM trend. |
