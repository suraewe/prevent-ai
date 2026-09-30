# 🦟 PREVENT AI

**Predictive Vector-borne Epidemic Network Tracker using AI**

An AI/ML-powered dashboard that predicts vector-borne disease outbreak risk (Dengue, Malaria, Chikungunya) across Bhopal, Ashta, and VIT Bhopal using environmental, geographic, and epidemiological data.

---

## 🎯 Features

- **🗺️ Interactive Risk Map** — Real-time risk heatmap across 20 monitored zones
- **🔮 Live Prediction** — Enter parameters and get instant risk assessment
- **📊 Model Comparison** — Side-by-side comparison of XGBoost, Random Forest, and Logistic Regression
- **📈 Trend Analysis** — Seasonal patterns, zone-wise heatmaps, and correlation matrices
- **📋 Data Explorer** — Browse and download raw/filtered data

## 🧠 ML Models

| Model | Description |
|-------|-------------|
| **XGBoost** | Gradient-boosted decision trees — best for tabular data |
| **Random Forest** | Ensemble of decision trees — robust and interpretable |
| **Logistic Regression** | Linear baseline model — simple but effective |

## 📍 Regions Covered

- **Bhopal City** — 12 zones (Old Bhopal, New Market, Upper/Lower Lake, Kolar Road, etc.)
- **Ashta** — 4 zones (Town Center, Industrial Area, Rural East, Nadi Belt)
- **VIT Bhopal** — 4 zones (Main Campus, Hostel Area, Surrounding Village, Kotri-Kalan)

## 📊 Input Features

| Feature | Description |
|---------|-------------|
| `avg_temperature_c` | Average monthly temperature (°C) |
| `rainfall_mm` | Monthly rainfall (mm) |
| `humidity_pct` | Monthly average humidity (%) |
| `water_proximity_km` | Distance to nearest water body (km) |
| `vegetation_index` | NDVI vegetation density (0-1) |
| `population_density` | People per km² |
| `previous_month_cases` | Reported cases last month |
| `disease` | Dengue / Malaria / Chikungunya |
| `season` | Winter / Summer / Monsoon / Post-Monsoon |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd "prevent ai"
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
python generate_data.py
```

### 3. Train ML Models

```bash
python train_models.py
```

### 4. Launch Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## 📁 Project Structure

```
prevent ai/
├── app.py                  # Streamlit dashboard (main app)
├── generate_data.py        # Synthetic data generator
├── train_models.py         # ML model training pipeline
├── config.json             # Region/zone configuration
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── DATA_GUIDE.md           # Instructions for using your own data
├── data/
│   └── disease_data.csv    # Generated/real dataset
└── models/
    ├── xgboost.joblib      # Trained XGBoost model
    ├── random_forest.joblib # Trained Random Forest model
    ├── logistic_regression.joblib # Trained Logistic Regression model
    ├── logistic_scaler.joblib     # Feature scaler
    ├── encoders.joblib     # Label encoders
    ├── model_meta.json     # Feature/target metadata
    └── model_comparison.json # Model performance metrics
```

---

## 📦 Tech Stack

- **Python 3.10+**
- **Streamlit** — Dashboard framework
- **scikit-learn** — Random Forest, Logistic Regression
- **XGBoost** — Gradient boosting
- **Folium** — Interactive maps
- **Plotly** — Charts and visualizations
- **Pandas / NumPy** — Data processing

---

## 🔄 Using Your Own Data

See **[DATA_GUIDE.md](DATA_GUIDE.md)** for detailed instructions on replacing synthetic data with your real datasets from Kaggle, local authorities, or other sources.

---

## 👥 Team

Built as a college project for AI/ML-based prediction model.

## 📄 License

This project is for educational purposes.
