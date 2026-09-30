"""
PREVENT AI — Streamlit Dashboard
=================================
Predictive Vector-borne Epidemic Network Tracker using AI

Interactive dashboard featuring:
  🗺️  Risk heatmap of Bhopal, Ashta, VIT Bhopal
  📊  Model comparison metrics
  🔮  Live prediction form
  📈  Trend analysis charts
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium

# ─────────────────────────────────────────────────────────────────────
# Page Config & Paths
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PREVENT AI — Disease Outbreak Predictor",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")

# ─────────────────────────────────────────────────────────────────────
# Custom CSS for premium dark theme
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ─── Import Google Font ─── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ─── Global ─── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ─── Header Hero ─── */
    .hero-container {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.6);
        font-size: 1.05rem;
        font-weight: 400;
        letter-spacing: 0.3px;
    }

    /* ─── Metric Cards ─── */
    .metric-card {
        background: linear-gradient(145deg, rgba(30,30,50,0.95) 0%, rgba(20,20,35,0.98) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(99,102,241,0.15);
        border-color: rgba(99,102,241,0.3);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: rgba(255,255,255,0.5);
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    /* ─── Risk Level Badges ─── */
    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
    }
    .risk-low { background: rgba(46,204,113,0.15); color: #2ecc71; border: 1px solid rgba(46,204,113,0.3); }
    .risk-medium { background: rgba(243,156,18,0.15); color: #f39c12; border: 1px solid rgba(243,156,18,0.3); }
    .risk-high { background: rgba(230,126,34,0.15); color: #e67e22; border: 1px solid rgba(230,126,34,0.3); }
    .risk-critical { background: rgba(231,76,60,0.15); color: #e74c3c; border: 1px solid rgba(231,76,60,0.3); }

    /* ─── Section Headers ─── */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(99,102,241,0.3);
    }

    /* ─── Prediction Result Box ─── */
    .prediction-box {
        background: linear-gradient(145deg, rgba(30,30,50,0.95), rgba(20,20,35,0.98));
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 12px 40px rgba(0,0,0,0.25);
    }
    .prediction-risk-score {
        font-size: 4rem;
        font-weight: 900;
        line-height: 1;
        margin: 0.5rem 0;
    }
    .score-low { color: #2ecc71; }
    .score-medium { color: #f39c12; }
    .score-high { color: #e67e22; }
    .score-critical { color: #e74c3c; }

    /* ─── Sidebar Styling ─── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: rgba(255,255,255,0.7);
    }

    /* ─── Model Card ─── */
    .model-card {
        background: linear-gradient(145deg, rgba(30,30,50,0.9), rgba(20,20,35,0.95));
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 1rem;
    }
    .model-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #a78bfa;
        margin-bottom: 0.8rem;
    }
    .model-metric-row {
        display: flex;
        justify-content: space-between;
        padding: 0.3rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .model-metric-label {
        color: rgba(255,255,255,0.5);
        font-size: 0.85rem;
    }
    .model-metric-value {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Data Loading (cached)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    csv_path = os.path.join(DATA_DIR, "disease_data.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "XGBoost": "xgboost.joblib",
        "Random Forest": "random_forest.joblib",
        "Logistic Regression": "logistic_regression.joblib",
    }
    for name, filename in model_files.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)

    # Load extras
    enc_path = os.path.join(MODEL_DIR, "encoders.joblib")
    scaler_path = os.path.join(MODEL_DIR, "logistic_scaler.joblib")
    comp_path = os.path.join(MODEL_DIR, "model_comparison.json")

    encoders = joblib.load(enc_path) if os.path.exists(enc_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    comparison = None
    if os.path.exists(comp_path):
        with open(comp_path, "r") as f:
            comparison = json.load(f)

    return models, encoders, scaler, comparison


# ─────────────────────────────────────────────────────────────────────
# Map Builder
# ─────────────────────────────────────────────────────────────────────
def build_risk_map(df: pd.DataFrame, config: dict, selected_disease: str, selected_month: int, selected_year: int):
    """Build a Folium map with risk zones colored by risk level."""
    # Center on Bhopal
    m = folium.Map(
        location=[23.2599, 77.4126],
        zoom_start=10,
        tiles=None,
    )

    # Use Streamlit secrets for the API token so it doesn't get pushed to GitHub
    MAPBOX_TOKEN = st.secrets["MAPBOX_TOKEN"]
    
    # Dark tile layer
    folium.TileLayer(
        tiles=f"https://api.mapbox.com/styles/v1/mapbox/dark-v11/tiles/256/{{z}}/{{x}}/{{y}}@2x?access_token={MAPBOX_TOKEN}",
        attr="Mapbox",
        name="Mapbox Dark",
    ).add_to(m)

    # Filter data
    filtered = df[
        (df["disease"] == selected_disease) &
        (df["month"] == selected_month) &
        (df["year"] == selected_year)
    ]

    if filtered.empty:
        return m

    # Color map
    risk_colors = {
        "Low": "#2ecc71",
        "Medium": "#f39c12",
        "High": "#e67e22",
        "Critical": "#e74c3c",
    }

    risk_icons = {
        "Low": "ok-sign",
        "Medium": "info-sign",
        "High": "warning-sign",
        "Critical": "exclamation-sign",
    }

    for _, row in filtered.iterrows():
        color = risk_colors.get(row["risk_level"], "#95a5a6")

        # Popup content
        popup_html = f"""
        <div style="font-family: Inter, sans-serif; min-width: 200px; padding: 8px;">
            <h4 style="color: {color}; margin: 0 0 8px 0; font-size: 14px;">{row['zone']}</h4>
            <table style="font-size: 12px; width: 100%;">
                <tr><td style="color: #888;">Region</td><td style="font-weight: 600;">{row['region']}</td></tr>
                <tr><td style="color: #888;">Disease</td><td style="font-weight: 600;">{row['disease']}</td></tr>
                <tr><td style="color: #888;">Risk Level</td><td style="font-weight: 700; color: {color};">{row['risk_level']}</td></tr>
                <tr><td style="color: #888;">Risk Score</td><td style="font-weight: 600;">{row['risk_score']:.1f}/100</td></tr>
                <tr><td style="color: #888;">Temp</td><td>{row['avg_temperature_c']}°C</td></tr>
                <tr><td style="color: #888;">Rainfall</td><td>{row['rainfall_mm']} mm</td></tr>
                <tr><td style="color: #888;">Cases (prev)</td><td>{row['previous_month_cases']}</td></tr>
            </table>
        </div>
        """

        # Circle marker with radius based on risk score
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8 + row["risk_score"] / 8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            weight=2,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{row['zone']} — {row['risk_level']} ({row['risk_score']:.0f})",
        ).add_to(m)

    return m


# ─────────────────────────────────────────────────────────────────────
# Prediction Logic
# ─────────────────────────────────────────────────────────────────────
def predict_risk(
    models: dict,
    encoders: dict,
    scaler,
    model_name: str,
    month: int,
    temperature: float,
    rainfall: float,
    humidity: float,
    water_proximity: float,
    vegetation_index: float,
    population_density: float,
    previous_cases: int,
    disease: str,
    season: str,
) -> tuple:
    """Run prediction using specified model. Returns (risk_label, probabilities)."""
    # Encode inputs
    disease_enc = encoders["disease"].transform([disease])[0]
    season_enc = encoders["season"].transform([season])[0]

    features = np.array([[
        month, temperature, rainfall, humidity,
        water_proximity, vegetation_index,
        population_density, previous_cases,
        disease_enc, season_enc,
    ]])

    model = models[model_name]

    if model_name == "Logistic Regression" and scaler is not None:
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
    else:
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

    risk_label = encoders["target"].inverse_transform([pred])[0]
    return risk_label, proba


# ─────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────
def main():
    # Load everything
    df = load_data()
    config = load_config()
    models, encoders, scaler, comparison = load_models()

    # ── Hero Header ──
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🦟 PREVENT AI</div>
        <div class="hero-subtitle">
            Predictive Vector-borne Epidemic Network Tracker &nbsp;•&nbsp;
            Bhopal &nbsp;|&nbsp; Ashta &nbsp;|&nbsp; VIT Bhopal
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.error("⚠️ No data found! Run `python generate_data.py` first, then `python train_models.py`.")
        return

    if not models:
        st.warning("⚠️ No trained models found! Run `python train_models.py` first.")

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### 🎛️ Controls")
        st.markdown("---")

        # Filters
        selected_disease = st.selectbox("🦠 Disease", config["diseases"], index=0)

        years = sorted(df["year"].unique())
        selected_year = st.selectbox("📅 Year", years, index=len(years) - 1)

        month_names = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        selected_month_name = st.selectbox("📆 Month", month_names, index=7)  # August default (monsoon)
        selected_month = month_names.index(selected_month_name) + 1

        selected_region = st.selectbox(
            "📍 Region",
            ["All Regions"] + [r["name"] for r in config["regions"].values()],
        )

        st.markdown("---")
        st.markdown("### 📋 About")
        st.markdown(
            "PREVENT AI uses **XGBoost**, **Random Forest**, and **Logistic Regression** "
            "to predict vector-borne disease outbreak risk based on climate, geography, "
            "and epidemiological data."
        )
        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; color: rgba(255,255,255,0.3); font-size: 0.75rem;'>"
            "Built for College Project • 2025</div>",
            unsafe_allow_html=True,
        )

    # ── Filter data ──
    view_df = df[
        (df["disease"] == selected_disease) &
        (df["year"] == selected_year) &
        (df["month"] == selected_month)
    ]
    if selected_region != "All Regions":
        view_df = view_df[view_df["region"] == selected_region]

    # ── Top Metrics Row ──
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_zones = view_df["zone"].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_zones}</div>
            <div class="metric-label">Zones Monitored</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        avg_risk = view_df["risk_score"].mean() if not view_df.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_risk:.1f}</div>
            <div class="metric-label">Avg Risk Score</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        critical_count = (view_df["risk_level"] == "Critical").sum() + (view_df["risk_level"] == "High").sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="background: linear-gradient(135deg, #e74c3c, #e67e22); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{critical_count}</div>
            <div class="metric-label">High/Critical Zones</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        total_cases = view_df["previous_month_cases"].sum() if not view_df.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_cases:,}</div>
            <div class="metric-label">Total Prev. Cases</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tab Layout ──
    tab_map, tab_predict, tab_models, tab_trends, tab_data = st.tabs([
        "🗺️ Risk Map", "🔮 Predict", "📊 Model Comparison", "📈 Trends", "📋 Data"
    ])

    # ════════════════════════════════════════════════════════════════
    # TAB 1: RISK MAP
    # ════════════════════════════════════════════════════════════════
    with tab_map:
        st.markdown('<div class="section-header">🗺️ Outbreak Risk Map</div>', unsafe_allow_html=True)
        st.caption(f"Showing {selected_disease} risk for {month_names[selected_month - 1]} {selected_year}")

        risk_map = build_risk_map(df, config, selected_disease, selected_month, selected_year)
        st_folium(risk_map, width=None, height=550, use_container_width=True)

        # Risk legend
        col_l1, col_l2, col_l3, col_l4 = st.columns(4)
        with col_l1:
            st.markdown('<span class="risk-badge risk-low">🟢 Low (0-25)</span>', unsafe_allow_html=True)
        with col_l2:
            st.markdown('<span class="risk-badge risk-medium">🟡 Medium (25-50)</span>', unsafe_allow_html=True)
        with col_l3:
            st.markdown('<span class="risk-badge risk-high">🟠 High (50-75)</span>', unsafe_allow_html=True)
        with col_l4:
            st.markdown('<span class="risk-badge risk-critical">🔴 Critical (75-100)</span>', unsafe_allow_html=True)

        # Zone-wise risk table
        if not view_df.empty:
            st.markdown('<div class="section-header">📋 Zone-wise Risk Breakdown</div>', unsafe_allow_html=True)
            display_df = view_df[["zone", "region", "risk_score", "risk_level", "avg_temperature_c", "rainfall_mm", "previous_month_cases"]].copy()
            display_df.columns = ["Zone", "Region", "Risk Score", "Risk Level", "Temp (°C)", "Rainfall (mm)", "Prev Cases"]
            display_df = display_df.sort_values("Risk Score", ascending=False).reset_index(drop=True)
            st.dataframe(display_df, use_container_width=True, height=400)

    # ════════════════════════════════════════════════════════════════
    # TAB 2: LIVE PREDICTION
    # ════════════════════════════════════════════════════════════════
    with tab_predict:
        st.markdown('<div class="section-header">🔮 Live Risk Prediction</div>', unsafe_allow_html=True)
        st.caption("Enter environmental and demographic parameters to get an instant risk assessment.")

        if not models:
            st.warning("⚠️ Train models first to use predictions!")
        else:
            pred_col1, pred_col2 = st.columns([1, 1])

            with pred_col1:
                st.markdown("##### 🌡️ Climate Parameters")
                p_month = st.slider("Month", 1, 12, 8, key="pred_month")
                p_temp = st.slider("Temperature (°C)", 10.0, 45.0, 30.0, step=0.5, key="pred_temp")
                p_rain = st.slider("Rainfall (mm)", 0.0, 500.0, 200.0, step=5.0, key="pred_rain")
                p_humidity = st.slider("Humidity (%)", 10.0, 100.0, 75.0, step=1.0, key="pred_humidity")

                st.markdown("##### 🦠 Disease")
                p_disease = st.selectbox("Disease", config["diseases"], key="pred_disease")

                # Determine season from month
                season_map = {1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer",
                              5: "Summer", 6: "Monsoon", 7: "Monsoon", 8: "Monsoon",
                              9: "Monsoon", 10: "Post-Monsoon", 11: "Post-Monsoon", 12: "Winter"}
                p_season = season_map[p_month]
                st.info(f"Season: **{p_season}**")

            with pred_col2:
                st.markdown("##### 📍 Geographic Parameters")
                p_water = st.slider("Distance to Water Body (km)", 0.1, 5.0, 1.5, step=0.1, key="pred_water")
                p_veg = st.slider("Vegetation Index (NDVI)", 0.0, 1.0, 0.5, step=0.05, key="pred_veg")

                st.markdown("##### 👥 Demographic Parameters")
                p_pop = st.slider("Population Density (per km²)", 100, 15000, 5000, step=100, key="pred_pop")
                p_prev = st.number_input("Previous Month Cases", min_value=0, max_value=500, value=15, key="pred_prev")

                st.markdown("##### 🤖 Model Selection")
                p_model = st.selectbox("Select Model", list(models.keys()), key="pred_model")

            # Predict button
            if st.button("🔮 Predict Risk", use_container_width=True, type="primary"):
                risk_label, proba = predict_risk(
                    models, encoders, scaler, p_model,
                    p_month, p_temp, p_rain, p_humidity,
                    p_water, p_veg, p_pop, p_prev,
                    p_disease, p_season,
                )

                # Map risk label to score class
                score_class = f"score-{risk_label.lower()}"
                risk_class = f"risk-{risk_label.lower()}"

                # Confidence
                confidence = max(proba) * 100

                res_col1, res_col2 = st.columns([1, 1])

                with res_col1:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div style="color: rgba(255,255,255,0.5); font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px;">Predicted Risk Level</div>
                        <div class="prediction-risk-score {score_class}">{risk_label}</div>
                        <div style="margin-top: 0.5rem;">
                            <span class="risk-badge {risk_class}">{risk_label} Risk</span>
                        </div>
                        <div style="color: rgba(255,255,255,0.4); margin-top: 1rem; font-size: 0.85rem;">
                            Model: {p_model} &nbsp;•&nbsp; Confidence: {confidence:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with res_col2:
                    # Probability bar chart
                    labels = encoders["target"].classes_
                    colors = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]
                    fig_prob = go.Figure(go.Bar(
                        x=proba * 100,
                        y=labels,
                        orientation="h",
                        marker_color=colors,
                        text=[f"{p*100:.1f}%" for p in proba],
                        textposition="auto",
                    ))
                    fig_prob.update_layout(
                        title="Class Probabilities",
                        xaxis_title="Probability (%)",
                        yaxis_title="",
                        height=300,
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 3: MODEL COMPARISON
    # ════════════════════════════════════════════════════════════════
    with tab_models:
        st.markdown('<div class="section-header">📊 Model Performance Comparison</div>', unsafe_allow_html=True)

        if comparison is None:
            st.warning("⚠️ Train models first to see comparison!")
        else:
            # Model cards
            model_cols = st.columns(3)
            for i, (model_name, metrics) in enumerate(comparison.items()):
                with model_cols[i]:
                    accent = ["#667eea", "#2ecc71", "#f39c12"][i]
                    st.markdown(f"""
                    <div class="model-card" style="border-top: 3px solid {accent};">
                        <div class="model-name" style="color: {accent};">{model_name}</div>
                        <div class="model-metric-row">
                            <span class="model-metric-label">Accuracy</span>
                            <span class="model-metric-value">{metrics['accuracy']*100:.2f}%</span>
                        </div>
                        <div class="model-metric-row">
                            <span class="model-metric-label">F1 Score</span>
                            <span class="model-metric-value">{metrics['f1_score']*100:.2f}%</span>
                        </div>
                        <div class="model-metric-row">
                            <span class="model-metric-label">Precision</span>
                            <span class="model-metric-value">{metrics['precision']*100:.2f}%</span>
                        </div>
                        <div class="model-metric-row">
                            <span class="model-metric-label">Recall</span>
                            <span class="model-metric-value">{metrics['recall']*100:.2f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Bar chart comparison
            st.markdown("<br>", unsafe_allow_html=True)

            comp_df = pd.DataFrame([
                {"Model": name, "Metric": metric, "Value": vals[metric]}
                for name, vals in comparison.items()
                for metric in ["accuracy", "f1_score", "precision", "recall"]
            ])

            fig_comp = px.bar(
                comp_df,
                x="Metric",
                y="Value",
                color="Model",
                barmode="group",
                color_discrete_sequence=["#667eea", "#2ecc71", "#f39c12"],
                title="Model Metrics Comparison",
            )
            fig_comp.update_layout(
                yaxis_tickformat=".0%",
                height=400,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            # Feature importance
            st.markdown('<div class="section-header">📌 Feature Importance</div>', unsafe_allow_html=True)

            fi_model = st.selectbox("Select Model", list(comparison.keys()), key="fi_model")
            fi = comparison[fi_model]["feature_importance"]
            fi_df = pd.DataFrame({"Feature": fi.keys(), "Importance": fi.values()})
            fi_df = fi_df.sort_values("Importance", ascending=True)

            fig_fi = px.bar(
                fi_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
                title=f"Feature Importance — {fi_model}",
            )
            fig_fi.update_layout(
                height=400,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 4: TRENDS
    # ════════════════════════════════════════════════════════════════
    with tab_trends:
        st.markdown('<div class="section-header">📈 Disease Trend Analysis</div>', unsafe_allow_html=True)

        trend_col1, trend_col2 = st.columns(2)

        # Monthly average risk across all zones
        with trend_col1:
            monthly_avg = df[df["disease"] == selected_disease].groupby(["year", "month"])["risk_score"].mean().reset_index()
            monthly_avg["date"] = pd.to_datetime(monthly_avg[["year", "month"]].assign(day=1))

            fig_trend = px.line(
                monthly_avg,
                x="date",
                y="risk_score",
                title=f"{selected_disease} — Average Risk Score Over Time",
                color_discrete_sequence=["#667eea"],
            )
            fig_trend.update_layout(
                xaxis_title="Date",
                yaxis_title="Risk Score",
                height=380,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            # Add risk level bands
            fig_trend.add_hrect(y0=0, y1=25, fillcolor="#2ecc71", opacity=0.08, line_width=0)
            fig_trend.add_hrect(y0=25, y1=50, fillcolor="#f39c12", opacity=0.08, line_width=0)
            fig_trend.add_hrect(y0=50, y1=75, fillcolor="#e67e22", opacity=0.08, line_width=0)
            fig_trend.add_hrect(y0=75, y1=100, fillcolor="#e74c3c", opacity=0.08, line_width=0)
            st.plotly_chart(fig_trend, use_container_width=True)

        # Seasonal risk distribution
        with trend_col2:
            season_risk = df[df["disease"] == selected_disease].groupby("season")["risk_score"].mean().reset_index()
            season_order = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
            season_risk["season"] = pd.Categorical(season_risk["season"], categories=season_order, ordered=True)
            season_risk = season_risk.sort_values("season")

            fig_season = px.bar(
                season_risk,
                x="season",
                y="risk_score",
                color="risk_score",
                color_continuous_scale=["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"],
                title=f"{selected_disease} — Risk by Season",
            )
            fig_season.update_layout(
                xaxis_title="Season",
                yaxis_title="Avg Risk Score",
                height=380,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig_season, use_container_width=True)

        # Risk heatmap: Zone × Month
        st.markdown('<div class="section-header">🔥 Risk Heatmap (Zone × Month)</div>', unsafe_allow_html=True)

        heat_year = st.selectbox("Year for heatmap", sorted(df["year"].unique()), index=len(years) - 1, key="heat_year")
        heat_data = df[
            (df["disease"] == selected_disease) &
            (df["year"] == heat_year)
        ].pivot_table(index="zone", columns="month", values="risk_score", aggfunc="mean")

        fig_heat = px.imshow(
            heat_data,
            color_continuous_scale=["#1a1a2e", "#2ecc71", "#f39c12", "#e67e22", "#e74c3c"],
            labels=dict(x="Month", y="Zone", color="Risk Score"),
            title=f"{selected_disease} Risk Heatmap — {heat_year}",
            aspect="auto",
        )
        fig_heat.update_layout(
            height=500,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_heat.update_xaxes(tickvals=list(range(1, 13)), ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        st.plotly_chart(fig_heat, use_container_width=True)

        # Correlation matrix
        st.markdown('<div class="section-header">🔗 Feature Correlation Matrix</div>', unsafe_allow_html=True)
        corr_cols = ["avg_temperature_c", "rainfall_mm", "humidity_pct", "water_proximity_km",
                     "vegetation_index", "population_density", "previous_month_cases", "risk_score"]
        corr_matrix = df[corr_cols].corr()

        fig_corr = px.imshow(
            corr_matrix,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
            title="Feature Correlation Matrix",
        )
        fig_corr.update_layout(
            height=500,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 5: RAW DATA
    # ════════════════════════════════════════════════════════════════
    with tab_data:
        st.markdown('<div class="section-header">📋 Raw Dataset</div>', unsafe_allow_html=True)
        st.caption(f"Showing {len(view_df):,} records for current filters")

        st.dataframe(view_df, use_container_width=True, height=500)

        # Download button
        csv_export = view_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Filtered Data as CSV",
            csv_export,
            f"prevent_ai_{selected_disease}_{selected_year}_{selected_month}.csv",
            "text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
