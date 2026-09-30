# 📋 Data Guide — How to Use Your Own Data

This guide explains exactly how to replace the synthetic data with your real datasets.

---

## 📂 Required CSV Format

Your data file must be saved as: `data/disease_data.csv`

### Required Columns

Your CSV **must** have these columns (exact names, case-sensitive):

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `year` | int | `2024` | Year of the record |
| `month` | int | `7` | Month (1-12) |
| `date` | string | `2024-07-01` | Date in YYYY-MM-DD format (1st of the month) |
| `season` | string | `Monsoon` | One of: `Winter`, `Summer`, `Monsoon`, `Post-Monsoon` |
| `region` | string | `Bhopal City` | Region name |
| `zone` | string | `Old Bhopal (Chowk)` | Zone/area name |
| `latitude` | float | `23.2676` | Zone latitude |
| `longitude` | float | `77.4120` | Zone longitude |
| `avg_temperature_c` | float | `30.5` | Average temperature for that month (°C) |
| `rainfall_mm` | float | `185.3` | Total rainfall for that month (mm) |
| `humidity_pct` | float | `78.2` | Average humidity (%) |
| `water_proximity_km` | float | `0.5` | Distance to nearest water body (km) |
| `vegetation_index` | float | `0.65` | NDVI vegetation index (0.0 to 1.0) |
| `population_density` | float | `8500` | People per square km |
| `previous_month_cases` | int | `23` | Number of cases reported in the previous month |
| `disease` | string | `Dengue` | One of: `Dengue`, `Malaria`, `Chikungunya` |
| `risk_score` | float | `67.5` | Risk score (0-100) — assign based on actual outbreak severity |
| `risk_level` | string | `High` | One of: `Low` (0-25), `Medium` (25-50), `High` (50-75), `Critical` (75+) |

---

## 📝 Step-by-Step: Adding Your Own Data

### Step 1: Prepare Your CSV

Create a CSV file with the columns listed above. Example:

```csv
year,month,date,season,region,zone,latitude,longitude,avg_temperature_c,rainfall_mm,humidity_pct,water_proximity_km,vegetation_index,population_density,previous_month_cases,disease,risk_score,risk_level
2024,7,2024-07-01,Monsoon,Bhopal City,Old Bhopal (Chowk),23.2676,77.4120,30.5,185.3,78.2,0.5,0.65,8500,23,Dengue,67.5,High
2024,7,2024-07-01,Monsoon,Bhopal City,New Market,23.2350,77.4230,31.2,180.0,76.5,2.1,0.35,9200,12,Dengue,42.3,Medium
```

### Step 2: Save File

Save your CSV as:
```
prevent ai/data/disease_data.csv
```

### Step 3: Retrain Models

After replacing the data, **you must retrain** the models:

```bash
python train_models.py
```

### Step 4: Launch Dashboard

```bash
streamlit run app.py
```

---

## 🌡️ Season Mapping

Use this mapping for the `season` column:

| Months | Season |
|--------|--------|
| January, February, December | `Winter` |
| March, April, May | `Summer` |
| June, July, August, September | `Monsoon` |
| October, November | `Post-Monsoon` |

---

## 📊 How to Assign Risk Scores

If your real data doesn't have a risk score, use this guide:

| Situation | Risk Score | Risk Level |
|-----------|-----------|------------|
| 0-5 cases, dry season, far from water | 0-25 | `Low` |
| 5-15 cases, moderate conditions | 25-50 | `Medium` |
| 15-40 cases, monsoon, near water | 50-75 | `High` |
| 40+ cases, active outbreak | 75-100 | `Critical` |

You can also keep the `risk_score` and `risk_level` columns empty and let the model train only on the input features — but then you'll need to modify `train_models.py` to compute risk labels yourself.

---

## 📦 Where to Get Real Data

### Weather Data (Temperature, Rainfall, Humidity)
- **IMD (India Meteorological Department)**: https://mausam.imd.gov.in/
- **Open-Meteo Historical API**: https://open-meteo.com/ (free, no key needed)
- **Kaggle**: Search "Bhopal weather data" or "India rainfall dataset"

### Disease Case Data
- **IDSP (Integrated Disease Surveillance Programme)**: https://idsp.nic.in/
- **Local Municipal Corporation / Health Department**: Request zone-wise case data
- **Kaggle**: Search "dengue india dataset", "malaria india"

### Geographic Data (Water Bodies, Vegetation)
- **Google Earth Engine**: NDVI vegetation data
- **OpenStreetMap**: Water body locations
- **ISRO Bhuvan**: https://bhuvan.nrsc.gov.in/

### Population Density
- **Census India**: https://censusindia.gov.in/
- **WorldPop**: https://www.worldpop.org/

---

## 🗺️ Adding New Zones

To add new zones to the map, edit `config.json`:

```json
{
    "regions": {
        "your_region_key": {
            "name": "Your Region Name",
            "lat": 23.0,
            "lon": 77.0,
            "zones": [
                {
                    "name": "Zone Name",
                    "lat": 23.05,
                    "lon": 77.05,
                    "water_proximity_km": 1.5,
                    "vegetation_index": 0.5
                }
            ]
        }
    }
}
```

Make sure your CSV data includes records for any new zones you add.

---

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| `KeyError: 'column_name'` | Check that your CSV has all required columns with exact names |
| Model accuracy drops | More data = better accuracy. Aim for at least 500+ rows |
| Map doesn't show zones | Verify lat/lon values are correct for your zones |
| `ValueError` during training | Check that `risk_level` values are exactly: `Low`, `Medium`, `High`, `Critical` |
| Encoders crash | Make sure `disease` values are exactly: `Dengue`, `Malaria`, `Chikungunya` |
