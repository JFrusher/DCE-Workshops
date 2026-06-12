# Modelled Interactions and Relationships

This file explains the main relationships that are deliberately built into the wind turbine telemetry. It is meant to help students notice the patterns themselves rather than hand them the answer immediately.

## What the data is designed to show

The telemetry is not random. It has several linked behaviours:

- Weather drives wind speed, gusts, turbulence, pressure, and direction together.
- Wind loading drives rotor speed, power, blade pitch, and electrical current.
- Mechanical degradation shows up first as subtle drift in vibration, temperature, pressure, and alignment.
- When a turbine fails, the output variables can flatline or drop to zero while the thermal and vibration signals remain abnormal for a while.

## Core relationships to nudge students toward

| Relationship | Expected pattern | Why it matters | Good visual to try |
|---|---|---|---|
| Wind speed vs power | Non-linear power curve that rises sharply after cut-in and then flattens near rated output. | This is the main operating relationship for a healthy turbine. | Scatter plot with a fitted curve. |
| Wind gust vs RPM | Positive relationship, but less smooth than power vs wind speed. | Gusts can drive short-term mechanical response. | Scatter plot with colour by turbine and a trend line. |
| Turbulence vs vibration | Positive relationship, especially when the machine is under stress. | Higher turbulence should increase structural loading and vibration. | Scatter plot and subgroup comparison by turbine. |
| Blade pitch vs power | Low to moderate pitch changes should track power output; failed or curtailed behaviour breaks the pattern. | Pitch is a strong clue when the turbine starts controlling badly or losing efficiency. | Scatter plot, often with a filtered active-generation subset. |
| Yaw error vs wind direction | Alignment should usually stay fairly tight, with larger spread suggesting control issues. | Misalignment can reduce efficiency and point to control or sensor problems. | Scatter plot or directional comparison. |
| Vibration per RPM over time | Healthy machines should stay relatively stable; a rising trend is a warning sign. | Normalising by RPM helps separate speed effects from roughness. | Time series with a rolling mean and fitted trend. |
| Power vs blade pitch and power vs wind speed together | Healthy turbines should obey both relationships simultaneously. | If one relationship looks normal but the other does not, that is a useful clue. | Small-multiple panel or dashboard. |

## Turbine-specific story built into the simulation

### WT-01

- Intended to look healthy and stable.
- Good as a baseline for power, vibration, and temperature.
- Includes a flatline injection in the messy dataset to mimic a temporary data issue.

### WT-02

- Begins to degrade after about one month.
- Eventually reaches a seizure state and drops to zero output.
- Useful for showing how a machine can move from subtle warning signs to an obvious failure.

### WT-03

- Starts healthy, then gradually drifts into worse behaviour later in the series.
- Designed to be the one students should prioritise before it fails.
- Good for spotting early trend changes rather than only obvious outages.

## Features students should connect together

- Rising vibration often accompanies worsening bearing or drivetrain behaviour.
- Falling or zero power with abnormal temperature and vibration suggests a serious fault rather than just poor weather.
- Blade pitch near its upper limit can indicate curtailed or unhealthy operation.
- A mismatch between wind loading and electrical output can signal mechanical loss or control issues.
- Missing values are not always noise; they can be evidence of sensor dropout or system shutdown.

## Suggested prompts for the notebook

- Which turbine still looks normal when you compare its relationship plots against the others?
- Where do you first see one turbine depart from the healthy pattern?
- Which signal changes first: power, vibration, temperature, or pitch?
- Can weather alone explain the change, or do the relationships suggest asset deterioration?
- What happens if you normalise vibration by RPM instead of looking at raw vibration alone?

## Best plots to encourage

- Scatter: wind speed vs power.
- Scatter: turbulence vs vibration.
- Scatter: gust vs RPM.
- Scatter: blade pitch vs power.
- Scatter: yaw error vs wind direction.
- Time series: vibration per RPM with a rolling mean.
- Dashboard: healthy baseline vs drifting turbine trend.

## Teaching note

The goal is to get students to infer the structure of the system from the relationships, not just to describe individual columns. If they can explain why a plot changes, they are already part-way to the operational decision.