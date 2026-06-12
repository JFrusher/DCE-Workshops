# Blueprint: AI & ML in Data-Centric Engineering Workshop
**Target System:** High-Pressure Water Injection Pump Asset
**Timeframe:** 90–120 Minutes (Highly Scaffolder / Fast-Paced)
**Core Goal:** Move from a naive, model-centric "black box" approach to an agile, data-centric, physics-aware, and ethically auditable engineering deployment.

---

## 🏗️ Master Architecture Blueprint

This document splits your workshop development into three distinct operational artifacts for GitHub Copilot to generate, alongside full presentation delivery scripts.

   [generate_data.py]  <-- Created via Copilot Prompt 1
           │
           ▼
   [pump_telemetry.csv]
           │
  ┌────────┴────────┐
  ▼                 ▼

[participant.ipynb] [solution.ipynb] <-- Created via Copilot Prompt 2
(Scaffolded Tasks)  (Full Code + Plots)

---

## 🛠️ Copilot Prompt 1: The Production-Grade Data Emulator

Paste this precise prompt into GitHub Copilot to generate your background data generator (`generate_data.py`).


Write a complete, self-contained Python script named `generate_data.py` that outputs a synthetic engineering dataset named `pump_telemetry.csv`. 

The dataset must emulate a high-pressure industrial water injection pump over 500 operating hours.
- Assume a base instrumentation sampling rate of 10 Hz at the asset layer, downsampled to 1-minute intervals for workshop telemetry logs (30,000 total rows).
- Set a global random seed of 42 for absolute mathematical reproducibility.

Columns to generate:
1. `timestamp`: Datetime sequence starting from '2026-01-01 00:00:00' incremented by 1-minute steps.
2. `time_hours`: Float sequence from 0.0 to 500.0 representing elapsed operating time.
3. `raw_vibration`: Time-series signal (m/s²) modeling an accelerometer on the bearing housing. 
   - Base signal: A stable 50 Hz sine wave representing normal shaft rotation speed (3000 RPM).
   - Noise injection: Add high-frequency white noise and random high-amplitude voltage spikes (Gaussian noise + sparse random anomalies) to simulate electrical interference from a faulty variable frequency drive (VFD).
4. `drifted_pressure`: Pump intake/suction pressure reading (kPa). 
   - Base signal: Constant ambient pressure at 120 kPa with minor physical fluctuations.
   - Degradation injection: Apply a deterministic linear downward drift coefficient of exactly -0.05 kPa per hour to simulate sensor membrane degradation and calcification.
5. `motor_temp`: Continuous operating casing temperature (°C). Base baseline of 65°C, rising naturally under high-load states or bearing friction.
6. `fluid_temp`: Process fluid temperature (°C). Constant ambient fluid input at 20°C with minor white noise.
7. `pump_efficiency`: The target performance health metric (Float from 0.0 to 1.0).
   - Program a hidden ground-truth physical degradation mapping: Efficiency drops exponentially if `drifted_pressure` drops below a critical threshold of 95 kPa (simulating thermodynamic cavitation breakdown).
   - Efficiency drops linearly if the true underlying vibration RMS exceeds a mechanical fatigue threshold.

Save the final data matrix to a standard comma-separated file named `pump_telemetry.csv`. Print a summary of the shape and the first 5 rows to confirm successful compilation.

---

## 💻 Copilot Prompt 2: Synchronized Notebooks Generation

Paste this prompt into Copilot to build both your `participant.ipynb` and `solution.ipynb` code bases simultaneously.


Act as an expert Senior Data-Centric Software Engineer and Corporate Technical Trainer. 

Generate two structurally identical Jupyter Notebooks:
1. `participant.ipynb` (Contains clear scaffolding, explanatory text, and designated `### YOUR CODE HERE ###` execution blocks).
2. `solution.ipynb` (Contains the exact same structure but with all code blocks fully completed, producing functional metrics and visual plots).

Both notebooks must ingest `pump_telemetry.csv` and execute the following 4-part pedagogical architecture:

---

### Part 1: The "Dirty" Reality (Data Cleaning & Exploration)
- Participant Task: 
  1. Handle sparse telemetry dropouts (simulate 5% random NaN rows injected into the stream) using a forward-fill (`.ffill()`) method to respect continuous physical system states.
  2. Implement a Savitzky-Golay filter (`scipy.signal.savgol_filter`) on `raw_vibration` to smooth away high-frequency VFD electrical noise without flattening genuine transient mechanical shock peaks (use window_length=15, polyorder=2).
  3. Hardcode a deterministic calibration offset loop to cancel out the linear sensor drift (-0.05 kPa/hr) based on the `time_hours` tracking column.
- Solution Output: Cleaned columns added: `cleaned_vibration`, `calibrated_pressure`.

---

### Part 2: Feature Engineering (Bringing in the Physics)
- Participant Task:
  1. Train a baseline naive Scikit-Learn `RandomForestRegressor(n_estimators=50, random_state=42)` using the uncleaned features (`raw_vibration`, `drifted_pressure`, `motor_temp`) to predict `pump_efficiency`. Calculate and display the R² score (Expected score is highly unoptimized, ~0.15 to 0.30).
  2. Introduce physical feature extraction: Write a rolling window function to extract the Root Mean Square (RMS) acceleration profile of the cleaned vibration signal.
  3. Write a Fast Fourier Transform function using `scipy.fft.rfft` and `rfftfreq` over a sliding window to isolate the dominant operational peak frequency component.
  4. Re-train the exact same Random Forest algorithm on the engineered feature matrix: `['vibration_rms', 'peak_frequency', 'calibrated_pressure', 'motor_temp']`.
- Solution Output: Print a comparison of the naive R² score versus the data-centric engineered R² score (Expected engineered score > 0.90).

---

### Part 3: Physics-Informed Boundary Concepts & Uncertainty Quantification
- Participant Task:
  1. Simulate an extreme out-of-distribution environment. Create a test slice where `motor_temp` is artificially inflated by +45°C to simulate an un-encountered thermal runaway grid state.
  2. Swap out the random forest for a `GaussianProcessRegressor` from `sklearn.gaussian_process` using a `Matern(nu=1.5)` kernel. Fit the model on nominal training data and predict on the extreme test slice, returning both the mean predictions and full standard deviation arrays.
  3. Write a physics validation function `enforce_physical_limits(predictions, inputs)` that clips efficiency boundaries tightly between 0.0 and 1.0. If `calibrated_pressure` drops below the known absolute vapor pressure threshold of 10.0 kPa, force an automatic hydraulic efficiency ceiling drop of 80% to represent hard cavitation system lock.
- Solution Output: Matplotlib plot showing the mean prediction line wrapped inside a shaded ±2 standard deviation uncertainty band ($\pm 2\sigma$), showing the confidence boundaries widening significantly in the extrapolation zone.

---

### Part 4: Applied Engineering Ethics & Forensic Audits
- Participant Task:
  1. Instantiate a SHAP tree explainer (`shap.TreeExplainer`) using the optimized engineered asset model.
  2. Calculate SHAP values across the test partition.
  3. Generate a `shap.summary_plot` and isolate an individual anomalous telemetry frame to generate a localized `shap.force_plot`.
- Solution Output: A rendered SHAP dependency visualization indicating exactly which physical sensor channel (e.g., thermal delta vs structural vibration acceleration) is driving an automated emergency system shutdown flag.

Ensure all markdown instructions within the notebooks explain the engineering *why* behind every step. Keep matplotlib parameters standardized, clean, and publication-ready.



---

## 📊 Slides Master Outline & Presenter Scripts

Use this section as your direct script blueprint during presentation design.

### Slide 1: Title & The Shift to Data-Centric Engineering

* **Visual:** Split screen layout. Left side: "Model-Centric AI" represented by a hyperparameter tuning loop optimizing an architecture against corrupt raw numbers. Right side: "Data-Centric AI" showing an engineer cleaning data streams while locking the core model code constant.
* **Talking Points & Script:**
> "Welcome to today's workshop. Historically, software engineers taught machine learning as a code problem—tweaking layers, adding neurons, and swapping architectures to boost test accuracy by fractions of a percent. In safety-critical data-centric engineering, this paradigm fails completely. We aren't classifying images of cats or predicting click-through rates. We are monitoring high-consequence physical assets operating in harsh, real-world conditions. If your input data contains uncalibrated sensor drift, electrical cross-talk, or cavitation anomalies, your model isn't learning engineering dynamics—it is simply memorizing noise. Today, we will prove that fixing the data is infinitely more powerful than fixing the algorithm."



### Slide 2: The Core Problem: The Asset Breakdown

* **Visual:** Cross-sectional engineering schematic of a high-pressure industrial injection pump asset. Clearly label sensor placement nodes: accelerometer on the bearing block, pressure transducers on the intake and discharge manifolds, and thermocouples embedded in the motor casing.
* **Talking Points & Script:**
> "Our focus asset for this workshop is a high-pressure water injection pump, a mechanical workhorse found in desalination plants, chemical processing facilities, and energy systems. This asset is subject to extreme physical degradation. We are monitoring it through four primary channels: vibration, pressure, and two temperature sensors. We face two main failure states: mechanical bearing seizure from alignment fatigue, and hydraulic cavitation—where low inlet pressures cause fluid to boil locally, creating imploding micro-bubbles that literally eat away the steel impeller. Our goal is to use machine learning to accurately track and predict pump health before catastrophic asset failure occurs."



### Slide 3: Noise, Drift, and Anomalies

* **Visual:** Two parallel time-series plots. Top plot: Raw telemetry from the field, buried under high-frequency hash patterns and showing a steady uncalibrated downward slope. Bottom plot: The isolated, cleaned structural baseline after digital filtering and deterministic drift re-biasing.
* **Talking Points & Script:**
> "When data arrives from physical assets, it never looks like a clean textbook dataset. Look at Slide 3. Physical environments introduce severe corruption. Nearby variable frequency drives inject high-frequency electrical noise across sensor cables, masking structural vibration signatures. Concurrently, chemical scaling on a sensor membrane causes its calibration to drift systematically over time. If you input this raw data directly into an ML model, it will miscalculate structural risks. We will implement digital signal processing—specifically a Savitzky-Golay filter—to strip out electrical noise without destroying critical high-frequency transient mechanical shocks, and we will code a deterministic re-biasing loop to correct sensor drift."



### Slide 4: Feature Engineering: Speaking the Language of Physics

* **Visual:** A graphic displaying a raw time-domain vibration signal feeding into a fast Fourier transform block, outputting a clear, sparse frequency spectrum plot dominated by distinct rotational harmonic peaks.
* **Talking Points & Script:**
> "A naive approach to machine learning feeds thousands of raw time-series points directly into a complex model, hoping the algorithm will independently deduce the laws of mechanics. This is highly inefficient. Instead, we use feature engineering to inject domain knowledge into the model. By calculating the Root Mean Square (RMS) acceleration, we provide the model with a direct metric for continuous kinetic energy dispersion. By utilizing a Fast Fourier Transform, we shift our perspective from time to frequency, isolating the exact rotational signatures of shaft misalignments. We speak the language of physics to make our machine learning models smaller, faster, more accurate, and far more robust."



### Slide 5: Risk & Uncertainty in Engineering AI

* **Visual:** An elegant line plot mapping asset performance over time. As the operational envelope moves past the vertical dotted line marking the boundary of the known training data, a shaded Gaussian Process variance envelope ($\pm 2\sigma$) widens exponentially.
* **Talking Points & Script:**
> "In traditional software deployments, a false prediction carries low consequences. In engineering, an incorrect prediction without warning is a major safety and financial liability. Standard neural networks or tree-based algorithms are incapable of acknowledging their own ignorance; when pushed into un-encountered operational zones—like an unprecedented operational heatwave—they will happily provide a highly confident but dangerously incorrect answer. We address this by utilizing Gaussian Process Regression. When our pump encounters a state it has never seen before, the model's confidence intervals widen dramatically. This tells the operator visually: *'The model does not know the answer; switch to manual physical verification loops.'*"



### Slide 6: Applied Engineering Ethics & Explainable Frameworks

* **Visual:** High-fidelity replication of your exact ethical hazard matrix from image_84513d.png:
| Ethical Driver | Engineering Hazard | Mitigation Action Item |
| :--- | :--- | :--- |
| **Public Safety** | AI recommends thin structural walls based on model optimizations. | Mandatory dual mechanical verification via FEA/CFD models. |
| **Equity / Bias** | Anomalies in unmonitored local districts missed by unrepresentative sensors. | Enforcing spatial sensory density standards across asset layouts. |
| **Forensic Audit** | Unexplained sudden emergency shutdown of continuous factory assets. | Enforcing SHAP feature tracking within continuous logging pipelines. |
* **Talking Points & Script:**
> "We conclude our workshop framework with the most critical component: Professional Ethics and Liability. As a Professional Engineer of Record, you carry legal responsibility for public safety. If a black box machine learning algorithm triggers an emergency facility shutdown or fails to predict a catastrophic rupture, you cannot pass liability to the code. We require absolute auditability. We implement explainability toolkits like SHAP to dissect the model's inner decision-making. If an automated safety flag occurs, we can forensically track exactly which sensor variables drove that assertion. This allows us to quickly identify whether the alert represents a genuine physical hazard or simply an unrepresentative sensor bias, keeping human accountability firmly at the center of AI-driven engineering decisions."