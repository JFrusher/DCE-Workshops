# DCE Code — Data-Centric Engineering

Internal repository for the Data-Centric Engineering (DCE) programme at the University of Southampton. Contains all workshop source materials, presenter notebooks, synthetic data generators, solutions, and supporting academic content.

The public participant-facing version of the workshops is maintained as a separate repository: **DCE-Code-Workshops**.

---

## Repository Structure

```
DCE Code/
│
├── analytics_workshop/              # Data Analytics & Visualisation workshop
│   ├── participant_analytics.ipynb  # Notebook given to participants (blank exercises)
│   ├── teachers_analytics.ipynb     # Presenter version with annotations and answers
│   ├── snippets.ipynb               # Scratch/dev notebook for building exercises
│   ├── turbine_telemetry.csv        # Primary synthetic dataset (wind turbine sensors)
│   ├── wind_turbine_telemetry.csv   # Alternate/extended telemetry dataset
│   ├── telemetry_schema.md          # Data dictionary for the telemetry CSVs
│   ├── Data Analytics & Visualisation in Data-Centric Engineering.pdf
│   ├── solutions/
│   │   └── solution_analytics.ipynb # Fully worked solutions
│   └── hidden/                      # Not distributed to participants
│       ├── emulator.py              # Synthetic wind turbine data generator
│       ├── generate_mock_wind_turbine_telemetry.py
│       └── modelled_interactions_and_relationships.md  # Design notes for the emulator
│
├── ML+AI_workshop/                  # Machine Learning & AI workshop
│   ├── Presentation.ipynb           # Live-demo presenter notebook (fully baked outputs)
│   ├── presentation_pump_telemetry.csv  # Dataset for the presentation demo
│   ├── solutions/
│   │   ├── solution_MLAI.ipynb      # Fully worked participant solutions
│   │   └── walkthrough_MLAI.ipynb   # Step-by-step walkthrough version
│   └── Hidden/                      # Not distributed to participants
│       ├── presentation_emulator.py # Synthetic pump telemetry generator (deterministic)
│       ├── emulator.py              # Earlier emulator iteration
│       ├── participant_MLAI.ipynb   # Participant notebook (blank exercises)
│       ├── bearing_telemetry.csv    # Bearing sensor data
│       ├── pump_telemetry*.csv      # Pump sensor data (multiple versions)
│       ├── setup.md                 # Session setup checklist for presenter
│       ├── curriculum_alignment_guide.md  # Maps exercises to learning outcomes
│       └── snippets.ipynb           # Dev scratch notebook
│
├── prosthetics_workshop/            # Prosthetics & Biomechanics data workshop
│   ├── participant_prosthetics.ipynb  # Participant notebook (blank exercises)
│   ├── teachers_prosthetics.ipynb     # Presenter version with answers
│   ├── snippets.ipynb               # Dev scratch notebook
│   ├── prosthetics_data.csv         # Synthetic prosthetics sensor dataset
│   ├── solutions/
│   │   └── solution_prosthetics.ipynb
│   └── Hidden/
│       └── generate_mock_prosthetics_data.py  # Synthetic data generator
│
├── AI_ethical_framework/            # Academic working paper
│   ├── SOTON-MECH-DCE-2026-WP01.tex  # LaTeX source for ethical framework paper
│   ├── references.bib               # Bibliography
│   ├── SOTON-MECH-DCE-2026-WP01.pdf  # Compiled output
│   └── CHANGELOG.md                 # Revision history for the paper
│
├── *.pptx                           # Slide decks for each workshop session
│   # Data Analytics and Visualisation.pptx
│   # Data Engineering and Management.pptx
│   # Machine Learning and AI in Data Analytics.pptx
│   # Versioning for Data-Centric Engineering.pptx
│   # Engineering your GenAI Assistant.pptx
│
├── Tree.drawio / Tree.png           # Visual map of repo/programme structure
├── Prepack.docx                     # Pre-session participant pack
├── publish-participants.ps1         # Script to sync participant notebooks to DCE-Code-Workshops
├── measure_mlai.py                  # Utility to measure/validate ML+AI notebook outputs
├── requirements.txt                 # Python dependencies for all workshops
└── README.md                        # This file
```

---

## Workshop Overview

| Workshop | Domain | Key Techniques |
|---|---|---|
| Data Analytics & Visualisation | Wind turbine operations | Pandas, Plotly, exploratory analysis |
| Machine Learning & AI | Pump condition monitoring | Feature engineering, GPR, SHAP, model comparison |
| Data Engineering and Management | Prosthetic limb sensor data | Data cleaning, statistical analysis, visualisation |

---

## Repo Roles

| File type | Purpose |
|---|---|
| `participant_*.ipynb` / `Hidden/participant_*.ipynb` | Distributed to participants — exercises blank |
| `teachers_*.ipynb` | Presenter-only — annotated with answers |
| `Presentation.ipynb` | Live screen-share demo — all outputs pre-baked, deterministic |
| `solutions/*.ipynb` | Post-session reference — fully worked |
| `hidden/` / `Hidden/` | Instructor-only: emulators, data generators, setup notes |
| `*.pptx` | Slide decks accompanying each workshop |

---

## Relationship to DCE-Code-Workshops

**DCE-Code-Workshops** is the public participant repository. It contains only the participant-facing notebooks, datasets, and solutions — no emulators, teacher notebooks, or presenter materials.

`publish-participants.ps1` handles the sync from this repo to DCE-Code-Workshops.

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended. Notebooks run in Jupyter Lab or VS Code with the Jupyter extension.

---

## Academic Content

`AI_ethical_framework/` contains a working paper (SOTON-MECH-DCE-2026-WP01) on ethical frameworks for AI in data-centric engineering. Compiled with LaTeX/BibTeX.
