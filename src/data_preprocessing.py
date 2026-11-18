"""
NASA Li-ion Battery Dataset - Data Extraction & Preprocessing
Uses the real NASA battery aging dataset from the AIML_project repository.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Source data from the cloned repo
AIML_DATA_DIR = Path(__file__).parent.parent.parent / "AIML_project" / "5. Battery Data Set"
CYCLE_SUMMARY_PATH = AIML_DATA_DIR / "battery_cycle_summary.csv"
NASA_PROCESSED_DIR = AIML_DATA_DIR / "NASA_Processed"


def ensure_dirs():
    """Create necessary directories."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_nasa_cycle_summary() -> pd.DataFrame:
    """Load the NASA battery cycle summary CSV."""
    print(f"Loading NASA cycle summary from {CYCLE_SUMMARY_PATH}")
    df = pd.read_csv(CYCLE_SUMMARY_PATH)
    
    # Standardize column names (already match spec)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Batteries: {df['battery_id'].nunique()}")
    print(f"Columns: {list(df.columns)}")
    print(f"Cycles per battery:\n{df.groupby('battery_id')['cycle'].count()}")
    
    return df


def load_nasa_processed_features() -> pd.DataFrame:
    """Load the rich NASA processed features with impedance data."""
    nasa_features_path = NASA_PROCESSED_DIR / "NASA_cycle_features.csv"
    print(f"Loading NASA processed features from {nasa_features_path}")
    df = pd.read_csv(nasa_features_path)
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Batteries: {df['battery_id'].nunique()}")
    
    return df


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the cycle summary data."""
    print("Validating and cleaning data...")
    
    original_len = len(df)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['battery_id', 'cycle'], keep='first')
    
    # Ensure cycle ordering per battery
    df = df.sort_values(['battery_id', 'cycle']).reset_index(drop=True)
    
    # Validate capacity - should be positive
    df = df[df['capacity'] > 0]
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            # Forward fill within each battery, then backward fill
            df[col] = df.groupby('battery_id')[col].transform(lambda x: x.ffill().bfill())
            # If still NaN, fill with global median
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
    
    # Validate temperature ranges (should be reasonable for battery operation)
    temp_cols = ['ambient_temp', 'temp_mean', 'temp_min', 'temp_max']
    for col in temp_cols:
        if col in df.columns:
            df[col] = df[col].clip(-40, 80)  # Reasonable battery temp range
    
    # Validate voltage ranges (typical Li-ion: 2.5V - 4.2V per cell)
    volt_cols = ['voltage_mean', 'voltage_min', 'voltage_max']
    for col in volt_cols:
        if col in df.columns:
            df[col] = df[col].clip(0, 5)
    
    # Validate current ranges
    curr_cols = ['current_mean', 'current_min', 'current_max']
    for col in curr_cols:
        if col in df.columns:
            df[col] = df[col].clip(-100, 100)  # Reasonable current range
    
    print(f"Cleaned: {original_len} -> {len(df)} rows")
    return df


def save_processed_data(df: pd.DataFrame, output_path: Path):
    """Save processed cycle summary to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Batteries: {df['battery_id'].nunique()}")
    print(f"Columns: {list(df.columns)}")


def main(use_synthetic: bool = False):
    """Main preprocessing pipeline using real NASA data.
    
    Args:
        use_synthetic: If True, generate synthetic data. 
                       If False (default), load real NASA data or raise error.
    """
    print("=" * 60)
    print("NASA Battery Dataset - Data Preprocessing Pipeline (Real Data)")
    print("=" * 60)
    
    ensure_dirs()
    
    # Check for existing processed data
    processed_path = PROCESSED_DIR / "cycle_summary.csv"
    if processed_path.exists():
        print(f"Processed data already exists at {processed_path}")
        df = pd.read_csv(processed_path)
        print(f"Loaded {len(df)} rows, {df['battery_id'].nunique()} batteries")
        return df
    
    # Load real NASA data
    if CYCLE_SUMMARY_PATH.exists():
        df = load_nasa_cycle_summary()
    elif use_synthetic:
        print("Real NASA data not found. Generating synthetic data (explicitly enabled)...")
        df = create_synthetic_data()
    else:
        raise FileNotFoundError(
            f"NASA data not found at {CYCLE_SUMMARY_PATH}. "
            f"Set use_synthetic=True to generate synthetic data, "
            f"or place real NASA data at {CYCLE_SUMMARY_PATH}"
        )
    
    # Validate and clean
    df = validate_and_clean(df)
    
    # Save
    save_processed_data(df, processed_path)
    
    return df


def create_synthetic_data(n_batteries: int = 8, n_cycles: int = 200) -> pd.DataFrame:
    """
    Create synthetic battery degradation data that mimics NASA dataset characteristics.
    Used as fallback when real data is not available.
    """
    print(f"Creating synthetic data: {n_batteries} batteries, ~{n_cycles} cycles each")

    np.random.seed(42)
    all_data = []

    # Battery IDs like NASA dataset
    battery_ids = [f"B{str(i).zfill(4)}" for i in range(5, 5 + n_batteries)]

    # Pre-define degradation profiles to ensure class balance
    # We want roughly equal numbers of Normal, Accelerated, Critical
    profiles = []
    n_per_profile = n_batteries // 3
    # Normal: slow degradation
    profiles.extend(['Normal'] * n_per_profile)
    # Accelerated: medium degradation
    profiles.extend(['Accelerated'] * n_per_profile)
    # Critical: fast degradation
    profiles.extend(['Critical'] * (n_batteries - len(profiles)))
    np.random.shuffle(profiles)

    for idx, bid in enumerate(battery_ids):
        # Initial capacity varies slightly per battery
        initial_capacity = np.random.normal(1.1, 0.02)  # ~1.1 Ah

        # Degradation rate varies per battery profile
        profile = profiles[idx]
        if profile == 'Normal':
            base_degradation = np.random.uniform(0.0003, 0.0008)  # Slow
        elif profile == 'Accelerated':
            base_degradation = np.random.uniform(0.0015, 0.0035)  # Medium
        else:  # Critical
            base_degradation = np.random.uniform(0.005, 0.012)  # Fast

        # Temperature profile
        base_temp = np.random.uniform(20, 30)
        temp_variation = np.random.uniform(2, 5)

        for cycle in range(1, n_cycles + 1):
            # Capacity with noise and degradation
            capacity = initial_capacity * (1 - base_degradation * cycle) + np.random.normal(0, 0.005)
            capacity = max(capacity, initial_capacity * 0.3)  # Floor at 30%

            # Temperature
            ambient_temp = base_temp + np.random.normal(0, temp_variation)
            temp_mean = ambient_temp + np.random.uniform(0, 5)  # Self-heating
            temp_min = temp_mean - np.random.uniform(1, 3)
            temp_max = temp_mean + np.random.uniform(1, 5)

            # Voltage (decreases slightly with degradation)
            voltage_mean = 3.7 - 0.1 * (1 - capacity / initial_capacity) + np.random.normal(0, 0.02)
            voltage_min = voltage_mean - np.random.uniform(0.1, 0.3)
            voltage_max = voltage_mean + np.random.uniform(0.1, 0.3)

            # Current (typical charge/discharge)
            current_mean = np.random.uniform(-1.5, 1.5)
            current_min = current_mean - np.random.uniform(0.5, 1.0)
            current_max = current_mean + np.random.uniform(0.5, 1.0)

            # Duration and measurements
            duration = np.random.uniform(3000, 7200)  # seconds
            n_measurements = np.random.randint(50, 200)

            all_data.append({
                'battery_id': bid,
                'cycle': cycle,
                'ambient_temp': round(ambient_temp, 2),
                'capacity': round(capacity, 4),
                'n_measurements': n_measurements,
                'duration': round(duration, 1),
                'voltage_mean': round(voltage_mean, 3),
                'voltage_min': round(voltage_min, 3),
                'voltage_max': round(voltage_max, 3),
                'current_mean': round(current_mean, 3),
                'current_min': round(current_min, 3),
                'current_max': round(current_max, 3),
                'temp_mean': round(temp_mean, 2),
                'temp_min': round(temp_min, 2),
                'temp_max': round(temp_max, 2),
            })

    df = pd.DataFrame(all_data)
    print(f"Created synthetic data: {len(df)} rows, {df['battery_id'].nunique()} batteries")
    return df


if __name__ == "__main__":
    df = main()
    print("\nPreprocessing complete!")
    print(df.head())
    print("\nData info:")
    print(df.info())
    print(f"\nBattery cycle counts:\n{df.groupby('battery_id')['cycle'].count()}")