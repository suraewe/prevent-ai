"""
PREVENT AI - Synthetic Data Generator
=====================================
Generates realistic synthetic data for vector-borne disease prediction
across Bhopal, Ashta, and VIT Bhopal regions.

This script creates training data with realistic correlations:
- Higher temperatures + rainfall = more mosquito breeding = higher risk
- Closer to water bodies = higher risk
- Higher vegetation density = higher risk (mosquito habitat)
- Seasonal patterns (monsoon = peak risk)

Run this ONCE to generate synthetic data, then replace with real data later.
See DATA_GUIDE.md for instructions on using your own data.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────
# Load config
# ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "config.json"), "r") as f:
    CONFIG = json.load(f)


def get_bhopal_climate(month: int) -> dict:
    """
    Returns realistic monthly climate averages for Bhopal region.
    Based on IMD (India Meteorological Department) historical data patterns.
    """
    # Bhopal climate data (approximate monthly averages)
    climate = {
        1:  {"temp_min": 9,  "temp_max": 24, "rainfall_mm": 12,  "humidity": 55, "season": "Winter"},
        2:  {"temp_min": 12, "temp_max": 27, "rainfall_mm": 8,   "humidity": 48, "season": "Winter"},
        3:  {"temp_min": 17, "temp_max": 33, "rainfall_mm": 5,   "humidity": 35, "season": "Summer"},
        4:  {"temp_min": 22, "temp_max": 38, "rainfall_mm": 3,   "humidity": 28, "season": "Summer"},
        5:  {"temp_min": 26, "temp_max": 42, "rainfall_mm": 10,  "humidity": 30, "season": "Summer"},
        6:  {"temp_min": 26, "temp_max": 38, "rainfall_mm": 130, "humidity": 60, "season": "Monsoon"},
        7:  {"temp_min": 24, "temp_max": 32, "rainfall_mm": 350, "humidity": 82, "season": "Monsoon"},
        8:  {"temp_min": 23, "temp_max": 30, "rainfall_mm": 310, "humidity": 85, "season": "Monsoon"},
        9:  {"temp_min": 22, "temp_max": 31, "rainfall_mm": 180, "humidity": 78, "season": "Monsoon"},
        10: {"temp_min": 18, "temp_max": 32, "rainfall_mm": 40,  "humidity": 58, "season": "Post-Monsoon"},
        11: {"temp_min": 13, "temp_max": 29, "rainfall_mm": 12,  "humidity": 50, "season": "Post-Monsoon"},
        12: {"temp_min": 9,  "temp_max": 25, "rainfall_mm": 8,   "humidity": 52, "season": "Winter"},
    }
    return climate[month]


def compute_risk_score(
    temp: float,
    rainfall: float,
    humidity: float,
    water_proximity_km: float,
    vegetation_index: float,
    prev_cases: int,
    population_density: float,
    disease: str,
) -> float:
    """
    Compute a realistic risk score (0-100) based on epidemiological factors.

    The formula is designed to mimic real-world vector-borne disease dynamics:
    - Mosquitoes breed optimally at 25-35°C
    - Standing water from rainfall increases breeding sites
    - Proximity to water bodies is a major risk factor
    - Dense vegetation provides mosquito habitat
    - Previous cases indicate existing vector populations
    """
    # Temperature factor: optimal breeding range 25-35°C
    if 25 <= temp <= 35:
        temp_factor = 1.0
    elif 20 <= temp < 25 or 35 < temp <= 40:
        temp_factor = 0.6
    else:
        temp_factor = 0.2

    # Rainfall factor: moderate rainfall increases risk, extreme can wash away
    if 50 <= rainfall <= 200:
        rain_factor = 1.0
    elif 200 < rainfall <= 400:
        rain_factor = 0.8
    elif 10 <= rainfall < 50:
        rain_factor = 0.4
    else:
        rain_factor = 0.15

    # Humidity factor
    humidity_factor = min(humidity / 100.0, 1.0)

    # Water proximity factor (inverse — closer = higher risk)
    water_factor = max(0, 1.0 - (water_proximity_km / 5.0))

    # Vegetation factor
    veg_factor = vegetation_index

    # Previous cases momentum factor
    prev_factor = min(prev_cases / 50.0, 1.0)

    # Population density factor (normalized)
    pop_factor = min(population_density / 10000.0, 1.0)

    # Disease-specific weight adjustments
    weights = {
        "Dengue":       {"temp": 0.22, "rain": 0.20, "humidity": 0.10, "water": 0.18, "veg": 0.10, "prev": 0.12, "pop": 0.08},
        "Malaria":      {"temp": 0.18, "rain": 0.22, "humidity": 0.12, "water": 0.20, "veg": 0.12, "prev": 0.10, "pop": 0.06},
        "Chikungunya":  {"temp": 0.25, "rain": 0.18, "humidity": 0.08, "water": 0.15, "veg": 0.12, "prev": 0.14, "pop": 0.08},
    }

    w = weights.get(disease, weights["Dengue"])

    score = (
        w["temp"] * temp_factor +
        w["rain"] * rain_factor +
        w["humidity"] * humidity_factor +
        w["water"] * water_factor +
        w["veg"] * veg_factor +
        w["prev"] * prev_factor +
        w["pop"] * pop_factor
    ) * 100

    return round(min(max(score, 0), 100), 2)


def risk_to_label(score: float) -> str:
    """Convert numeric risk score to categorical label."""
    if score < 25:
        return "Low"
    elif score < 50:
        return "Medium"
    elif score < 75:
        return "High"
    else:
        return "Critical"


def generate_synthetic_data(
    start_year: int = 2020,
    end_year: int = 2025,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic monthly disease data for all zones across all regions.

    Returns a DataFrame with one row per (zone × month × disease) combination.
    """
    np.random.seed(seed)
    rows = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            climate = get_bhopal_climate(month)

            for region_key, region_data in CONFIG["regions"].items():
                for zone in region_data["zones"]:
                    # Add realistic noise to climate data per zone
                    temp_noise = np.random.normal(0, 1.5)
                    rain_noise = np.random.normal(0, climate["rainfall_mm"] * 0.2)
                    humidity_noise = np.random.normal(0, 5)

                    avg_temp = (climate["temp_min"] + climate["temp_max"]) / 2.0 + temp_noise
                    rainfall = max(0, climate["rainfall_mm"] + rain_noise)
                    humidity = min(100, max(10, climate["humidity"] + humidity_noise))

                    # Population density varies by area type
                    if "VIT" in zone["name"]:
                        pop_density = np.random.uniform(3000, 5000)
                    elif "Rural" in zone["name"] or "Village" in zone["name"]:
                        pop_density = np.random.uniform(500, 2000)
                    elif "Industrial" in zone["name"]:
                        pop_density = np.random.uniform(2000, 4000)
                    else:
                        pop_density = np.random.uniform(4000, 12000)

                    for disease in CONFIG["diseases"]:
                        # Previous cases with seasonal pattern and noise
                        season_multiplier = {
                            "Winter": 0.2, "Summer": 0.4,
                            "Monsoon": 1.0, "Post-Monsoon": 0.6,
                        }[climate["season"]]

                        base_cases = np.random.poisson(
                            lam=max(1, 15 * season_multiplier * zone["vegetation_index"])
                        )

                        # Closer to water → more cases
                        water_boost = max(0, int(10 * (1 - zone["water_proximity_km"] / 5.0)))
                        prev_cases = base_cases + water_boost

                        # Compute risk
                        risk_score = compute_risk_score(
                            temp=avg_temp,
                            rainfall=rainfall,
                            humidity=humidity,
                            water_proximity_km=zone["water_proximity_km"],
                            vegetation_index=zone["vegetation_index"],
                            prev_cases=prev_cases,
                            population_density=pop_density,
                            disease=disease,
                        )

                        risk_label = risk_to_label(risk_score)

                        rows.append({
                            "year": year,
                            "month": month,
                            "date": f"{year}-{month:02d}-01",
                            "season": climate["season"],
                            "region": region_data["name"],
                            "zone": zone["name"],
                            "latitude": zone["lat"],
                            "longitude": zone["lon"],
                            "avg_temperature_c": round(avg_temp, 1),
                            "rainfall_mm": round(rainfall, 1),
                            "humidity_pct": round(humidity, 1),
                            "water_proximity_km": zone["water_proximity_km"],
                            "vegetation_index": zone["vegetation_index"],
                            "population_density": round(pop_density, 0),
                            "previous_month_cases": prev_cases,
                            "disease": disease,
                            "risk_score": risk_score,
                            "risk_level": risk_label,
                        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def main():
    """Generate and save synthetic dataset."""
    print("=" * 60)
    print("  PREVENT AI - Synthetic Data Generator")
    print("=" * 60)

    # Create data directory
    data_dir = os.path.join(SCRIPT_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Generate data
    print("\n Generating synthetic data (2020-2025)...")
    df = generate_synthetic_data(start_year=2020, end_year=2025)

    # Save full dataset
    output_path = os.path.join(data_dir, "disease_data.csv")
    df.to_csv(output_path, index=False)
    print(f"[OK] Saved: {output_path}")

    # Print summary stats
    print(f"\n Dataset Summary:")
    print(f"   Total records: {len(df):,}")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Regions: {df['region'].nunique()} ({', '.join(df['region'].unique())})")
    print(f"   Zones: {df['zone'].nunique()}")
    print(f"   Diseases: {', '.join(df['disease'].unique())}")
    print(f"\n   Risk Level Distribution:")
    for level, count in df["risk_level"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    [{level}] {level}: {count:,} ({pct:.1f}%)")

    print(f"\n{'=' * 60}")
    print("  Data generation complete! Run train_models.py next.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
