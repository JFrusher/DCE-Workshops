# ML+AI Workshop Curriculum Alignment Guide

## Overview
The solution notebook should teach five core concepts aligned with the pump diagnostics problem. This guide identifies what needs to change to match the new curriculum structure and messaging.

---

## Current State vs. New Curriculum

### 1. **Problem Framing** (NEEDS EXPANSION)
**Current:** Implicit; not clearly stated in the notebook.  
**Curriculum:** Explicit pump problem—high-pressure water injection pump used for desalination, risks cavitation and bearing seizure.

**Action Required:**
- Add a new opening section that contextualizes the problem.
- Explain why data cleaning, feature engineering, uncertainty quantification, and explainability matter for this specific asset.
- Connect the telemetry data to real failure modes (cavitation, bearing wear).

---

### 2. **Part 1: Noise, Drift, and Anomalies** ✓ (GOOD)
**Current:** Forward-fill sparse dropouts, Savitzky-Golay smoothing, deterministic pressure calibration.  
**Curriculum:** Exactly the same approach.

**Status:** Aligned. No changes needed.

---

### 3. **Part 2: Feature Engineering** (PARTIAL MISMATCH)
**Current:**
- Vibration RMS (rolling window energy)
- Vibration trend (rate of change)
- Pressure reserve (deviation from rolling mean)

**Curriculum Slides Say:**
- Root Mean Square (RMS) ✓
- Fast Fourier Transforms (FFT) ✗ (NOT in current notebook)

**Action Required:**
- **Option A (Simplification):** Keep RMS and trend; add a comment explaining that FFT analysis is deferred to advanced courses because tree models already extract frequency features implicitly.
- **Option B (Full Alignment):** Replace `vibration_trend` with FFT peak frequency. Add a separate FFT cell that extracts frequency-domain features and shows why they matter for bearing diagnostics.
- **Recommended:** Option A + add a note comparing FFT to the trend approach.

**Explanation to Add:**
> "The curriculum emphasizes Root Mean Square (RMS) as a direct proxy for kinetic energy dispersion. RMS is sufficient for shallow ML models because it captures the envelope of vibration intensity. In practice, Fourier analysis (FFT) would decompose the signal into frequency components to detect specific bearing fault signatures (e.g., ball-pass frequency faults). For this workshop, we simplify by using RMS and a short-term trend instead."

---

### 4. **Part 3: Risk & Uncertainty** (MISMATCH)
**Current:**
- Tree ensemble spread (variance across individual tree predictions)
- Used as a proxy for model uncertainty in out-of-distribution regions.

**Curriculum Slides Say:**
- Gaussian Process Regression (GPR) with explicit uncertainty vectors (±2σ).
- "Tree ensembles and traditional neural networks cannot quantify their own ignorance when extrapolating."

**Action Required:**
- **Recommended Change:** Replace the current tree-spread approach with a true Gaussian Process Regressor (GPR) or keep trees but explain the simplification.
- If **keeping trees:** Add a clarification that tree spread is a pragmatic approximation when GPR is unavailable, noting its limitations.
- If **switching to GPR:** Implement `sklearn.gaussian_process.GaussianProcessRegressor` with a suitable kernel (e.g., Matern or RBF).

**Explanation to Add (if keeping trees):**
> "While the curriculum emphasizes Gaussian Processes for uncertainty quantification, we use tree ensemble spread as a simpler, more interpretable proxy. Each tree in the forest makes slightly different predictions based on its random subset of training data. When predictions diverge widely, the model is uncertain. This approximation breaks down at extreme extrapolation, which is why we layer in physics-informed boundary constraints."

---

### 5. **Physics Boundary Layer Constraints** ✓ (GOOD, but needs better narrative)
**Current:**
- Implemented in `enforce_physical_limits()`: if pressure reserve < -7.5, cap efficiency at 0.2.
- Simulates cavitation shock.

**Curriculum:**
- Explicitly mentioned: "hard boundary constraints to catch unphysical extrapolations."
- Cavitation override when intake pressure drops below vapor pressure threshold.

**Action Required:**
- **Enhance explanation** in the notebook before Part 3 cell.
- Add a dedicated subsection or markdown cell that explains:
  - Why pure data-driven models fail (they don't know conservation laws).
  - How the physics constraint implements cavitation detection.
  - Why efficiency collapses when pressure reserve falls below the threshold.

**Suggested Markdown:**
> "### Physics-Informed Safety Override
> 
> Raw ML models cannot understand physical laws. If intake pressure drops below the vapor pressure threshold (approximately -7.5 kPa reserve in this system), cavitation bubbles nucleate and collapse violently, destroying pump efficiency and damaging bearings.
> 
> Our model enforces a hard boundary: if the data predicts normal efficiency during cavitation, the physics layer overrides and forces efficiency to collapse (≤0.2). This rule is based on first principles, not learned from data, and is non-negotiable in production."

---

### 6. **Part 4: Ethics and Forensic Audit** ✓ (GOOD)
**Current:**
- SHAP explanations for feature attribution.
- Identifies dominant sensor (pressure_reserve in this dataset).
- Force plot showing contribution breakdown.

**Curriculum:**
- "Engineer of Record carries legal liability; black-box excuses are inadmissible."
- SHAP uses game theory for fair allocation.
- Distinguishes mechanical failures from sensor drift.

**Status:** Aligned. Could strengthen the legal/ethical framing in the markdown introduction.

**Optional Enhancement:**
Add a brief note about real-world deployment:
> "In production, if the force plot shows vibration RMS driving a shutdown prediction, maintenance must inspect bearings. If only pressure_reserve is flagged, recalibrate the pressure sensor first. SHAP allocates credit fairly so engineers can prioritize the right response."

---

## Summary of Required Changes

### High Priority
1. **Add problem context at the top** – Explain the pump, cavitation risk, and why ML matters.
2. **Clarify uncertainty approach** – Either switch to GPR or explain why tree spread is a pragmatic simplification.
3. **Strengthen physics constraint narrative** – Add dedicated explanation of cavitation boundary layer.

### Medium Priority
1. **FFT discussion** – Acknowledge the curriculum mentions FFT; explain why RMS + trend is equivalent for shallow models or add FFT as an optional extension.
2. **Enhance ethics section** – Strengthen the narrative around engineer responsibility and actionability.

### Low Priority
1. Minor wording refinements for clarity and alignment with slide language.

---

## Recommended Edits to Narrative Cells

### Before Part 1 (Add New Section)
```markdown
## Context: The Desalination Pump Problem

This workshop uses real telemetry from a high-pressure water injection pump designed for reverse-osmosis desalination plants. 
Inspection revealed hydraulic cavitation and bearing wear risk. Our goal: build a predictive model that engineers can trust 
to catch failure signatures in real time using raw telemetry data.

The challenge is not the algorithm—it is the data quality and feature design. We will show that:
1. Noisy, drifting raw signals are useless for prediction (R² ≈ 0.20).
2. Physics-informed cleaning and feature engineering make accuracy jump to R² ≈ 0.79.
3. Uncertainty quantification prevents over-confident extrapolation into failure regimes.
4. Explainability ensures the engineer can act on model output, not hide behind a black box.
```

### Before Part 3 (Enhance Uncertainty Intro)
```markdown
## Part 3: Boundary Concepts and Uncertainty

[EXISTING TEXT OK, but add before tree ensemble code:]

The curriculum emphasizes Gaussian Process Regression (GPR) as the gold standard for uncertainty. However, GPR is 
computationally expensive for large datasets. Here we use a pragmatic alternative: tree ensemble spread. When individual 
trees predict widely different values for the same input, the ensemble is uncertain. This approximation works well within 
the training envelope but breaks down at extreme extrapolation—which is why we layer in physics-informed constraints below.
```

### After Part 3 Plots (Add New Subsection)
```markdown
### Physics-Informed Cavitation Override

Pressure reserve below -7.5 kPa signals cavitation risk. Data alone cannot enforce this rule. We implement a hard 
physics boundary that collapses efficiency predictions to ≤0.2 whenever cavitation conditions exist. This is not learned 
from data; it is inviolable engineering knowledge embedded in the model.
```

---

## Participant Notebook Alignment
Ensure the participant notebook (non-solution) includes these same narrative elements so students understand the context 
and the connection between data science and physical engineering reality. The scaffolding should guide students toward 
the same conclusions without handing them the code.
