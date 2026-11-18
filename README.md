# EV Battery Predictive Maintenance System

An end-to-end ML pipeline for Electric Vehicle (EV) battery health monitoring using the NASA Li-ion Battery Aging Dataset. Predicts State of Health (SOH), Remaining Useful Life (RUL), degradation severity, and failure risk with a rule-based maintenance decision engine and interactive Streamlit dashboard.

## Features

- **SOH Estimation** - Predict battery health from discharge capacity (MAE: 2.7%)
- **RUL Prediction** - Cycles until 70% EOL threshold (MAE: 83 cycles)
- **Degradation Classification** - Normal / Accelerated / Critical (F1: 40%)
- **Failure Risk Classification** - Low / Medium / High (F1: 53%)
- **Maintenance Decision Engine** - 15 safety-critical rules with timelines
- **Interactive Dashboard** - Streamlit UI with degradation curves, predictions, recommendations
- **Leakage-Free Training** - Battery-wise GroupShuffleSplit / GroupKFold

## Quick Start

### Prerequisites
- Python 3.10+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/NAREN747/ev-battery.git
cd ev-battery

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python run.py all
```

This executes:
1. **Preprocessing** - Loads NASA data, creates `data/processed/cycle_summary.csv`
2. **Feature Engineering** - Computes SOH, RUL, severity, risk + 189 features
3. **Model Training** - Trains 13 models, saves best per target to `models/`

### Launch Dashboard

```bash
python run.py dashboard
```

Open http://localhost:8501 in your browser.

## Project Structure

```
ev-battery/
├── run.py                    # CLI entry point
├── requirements.txt          # Python dependencies
├── src/
│   ├── data_preprocessing.py    # NASA data loading & cleaning
│   ├── feature_engineering.py   # SOH, RUL, severity, risk + features
│   ├── model_training.py        # Battery-wise split, model comparison
│   └── maintenance_decision.py  # 15-rule decision engine
├── dashboard/
│   └── app.py                 # Streamlit dashboard
├── data/
│   ├── processed/             # cycle_summary.csv (generated)
│   └── features/              # features.csv, targets.csv, enriched_data.csv (generated)
├── models/                    # best_*.pkl, preprocessor.pkl (generated, gitignored)
├── plots/                     # Evaluation plots (generated)
└── requirements.txt
```

## Requirements

See `ev_maintenance/requirements.txt`:

```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.25.0
joblib>=1.3.0
plotly>=5.15.0
```

Install with: `pip install -r requirements.txt`

## Model Performance

| Target | Best Model | Test Metric |
|--------|------------|-------------|
| SOH | ExtraTrees | MAE: 2.5%, R²: 0.83 |
| RUL | Ridge | MAE: 85 cycles |
| Degradation Severity | SVC | F1_macro: 34% |
| Failure Risk | MLP | F1_macro: 59% |

All splits are battery-wise (GroupShuffleSplit) to prevent data leakage.
Cross-validation uses GroupKFold with preprocessing inside the CV loop.

## Data Source

- **Dataset**: NASA Li-ion Battery Aging Dataset (NASA Ames Research Center)
- **Source**: https://data.nasa.gov/dataset/li-ion-battery-aging-datasets
- **Format**: 34 batteries, 2,750 cycles, 15 raw features → 189 engineered features

## Retraining

```bash
# Full retrain
python run.py all

# Individual stages
python run.py preprocess   # Raw data → cycle_summary.csv
python run.py features     # cycle_summary.csv → features + targets
python run.py train        # Features → trained models
```

## Configuration

Key parameters in `src/feature_engineering.py`:
- `eol_threshold=70.0` - SOH % for End-of-Life
- Severity thresholds: Normal < 0.10, Accelerated 0.10-0.30, Critical > 0.30 (% SOH/cycle)
- Risk weights: SOH 40%, Degradation rate 30%, Temperature 20%, Voltage 10%
- Risk thresholds: Low ≤ 0.30, Medium 0.30-0.65, High > 0.65

Key parameters in `src/maintenance_decision.py`:
- SOH thresholds: HEALTHY=90, DEGRADED=80, EOL=70, CRITICAL=60
- RUL thresholds: PLENTY=200, MODERATE=100, LOW=50, CRITICAL=20
- Temperature thresholds: NORMAL=40, ELEVATED=50, HIGH=60, CRITICAL=70°C

## Important Methodology Notes

**Failure Risk Classification**: The failure risk labels are generated using a rule-based formula (SOH 40%, degradation rate 30%, temperature 20%, voltage 10%). The classifier learns to reproduce this rule-based classification, not independently predict future battery failure events. This is a *risk classification* task, not a future-failure prediction task.

**RUL Label Construction**: RUL labels are calculated using each battery's eventual end-of-life cycle (when SOH ≤ 70%). For batteries that don't reach this threshold in the observed data, EOL is estimated by extrapolating the average degradation rate. These are *estimated* training labels, not observed remaining useful life.

**Cross-Validation**: Preprocessing (imputation and scaling) is performed inside the GroupKFold cross-validation loop to prevent leakage from validation folds into training statistics.

## Model Files

Model files (`.pkl`) and data CSVs are gitignored due to size. To share:
- Use Git LFS: `git lfs track "*.pkl" "data/**/*.csv"`
- Or upload as GitHub Release assets
- Or share via cloud storage (Google Drive, S3, HuggingFace Hub)

## Using Your Own Dataset

The pipeline expects a CSV with battery cycle data. Required columns:

```csv
battery_id,cycle,ambient_temp,capacity,n_measurements,duration,voltage_mean,voltage_min,voltage_max,current_mean,current_min,current_max,temp_mean,temp_min,temp_max
B001,1,25,1.1,100,3600,3.7,3.5,3.9,1.0,0.8,1.2,30,28,32
```

### Steps:

1. **Place your data** at `data/raw/your_battery_data.csv`

2. **Modify `src/data_preprocessing.py`** - replace `load_nasa_cycle_summary()`:

```python
def load_nasa_cycle_summary() -> pd.DataFrame:
    YOUR_DATA_PATH = Path("data/raw/your_battery_data.csv")
    df = pd.read_csv(YOUR_DATA_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Validate required columns
    required = ['battery_id', 'cycle', 'capacity', 'ambient_temp',
                'voltage_mean', 'voltage_min', 'voltage_max',
                'current_mean', 'current_min', 'current_max',
                'temp_mean', 'temp_min', 'temp_max']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    print(f"Loaded {len(df)} rows, {df['battery_id'].nunique()} batteries")
    return df
```

3. **Run pipeline**:
```bash
python run.py all
```

### Requirements:
- At least 20 cycles per battery
- Capacity must decrease over cycles (for SOH/RUL calculation)
- Some batteries should reach ~70% SOH (for valid RUL targets)
- If columns differ, rename them in the loader before validation

## License

MIT License - feel free to use for research and commercial purposes.

## Citation

If you use this in research, please cite:
- NASA Li-ion Battery Aging Dataset (NASA Ames Research Center)
- This implementation: https://github.com/NAREN747/ev-battery