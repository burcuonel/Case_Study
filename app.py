import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime
from typing import List, Dict, Optional

st.set_page_config(
    page_title="Bologna University - Digital Twin Prototype",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #1f4e79;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        margin: 1rem 0;
        color: #2c5282;
    }
    .feature-box {
        background: #f7fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3182ce;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# =====================
# ---- HELPER FUNCTIONS -----
# =====================
@st.cache_data(show_spinner=False)
def load_dataframe(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    name = getattr(file, "name", "uploaded").lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    
    # Try to find and convert datetime column
    dt_candidates = [c for c in df.columns if any(x in str(c).lower() for x in ["datetime","time","timestamp","date","hour"])]
    for c in dt_candidates:
        try:
            original_values = df[c].copy()
            df[c] = pd.to_datetime(df[c], errors="coerce")
            if df[c].notna().any():
                df = df.sort_values(c)
                break
            else:
                df[c] = original_values
        except Exception:
            continue
    return df

def get_datetime_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        try:
            if np.issubdtype(df[c].dtype, np.datetime64):
                return c
        except Exception:
            continue
    return None

@st.cache_data(show_spinner=False)
def resample_df(df: pd.DataFrame, dt_col: Optional[str], rule: str) -> pd.DataFrame:
    if dt_col is None:
        return df
    g = df.set_index(dt_col)
    num_cols = g.select_dtypes(include=[np.number]).columns
    out = g[num_cols].resample(rule).mean().reset_index()
    return out

def prepare_ml_data(df):
    """Automatically prepares dataset for machine learning"""
    df_ml = df.copy()
    
    # 1. Wall temperature (target variable)
    wall_cols = [c for c in df_ml.columns if "wall" in c.lower() and "temperature" in c.lower() and "radiator" not in c.lower()]
    if wall_cols:
        df_ml["WallTemp"] = df_ml[wall_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    
    # 2. Radiator temperature
    rad_cols = [c for c in df_ml.columns if "radiator" in c.lower() and "temperature" in c.lower()]
    if rad_cols:
        df_ml["RadiatorTemp"] = df_ml[rad_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    
    # 3. Occupancy
    occ_cols = [c for c in df_ml.columns if "occupancy" in c.lower() and "%" not in c]
    if occ_cols:
        df_ml["Occupancy"] = df_ml[occ_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    
    # 4. CO2
    co2_cols = [c for c in df_ml.columns if "co2" in c.lower()]
    if co2_cols:
        df_ml["CO2"] = df_ml[co2_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    
    # 5. External temperature
    ext_cols = [c for c in df_ml.columns if any(x in c.lower() for x in ["roof", "external", "outdoor", "outside"])]
    if ext_cols:
        df_ml["ExternalTemp"] = df_ml[ext_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    else:
        df_ml["ExternalTemp"] = 15.0  # Default
    
    # 6. Time features (with better error handling)
    dt_col = get_datetime_column(df_ml)
    if dt_col:
        try:
            df_ml[dt_col] = pd.to_datetime(df_ml[dt_col], errors="coerce")
            valid_dates = df_ml[dt_col].notna()
            if valid_dates.any():
                df_ml["Hour"] = df_ml[dt_col].dt.hour
                df_ml["Weekday"] = df_ml[dt_col].dt.weekday
                df_ml["Month"] = df_ml[dt_col].dt.month
        except Exception as e:
            st.warning(f"Could not process datetime column '{dt_col}': {str(e)}")
    
    return df_ml

def train_models(df_ml):
    """Trains Random Forest and XGBoost models"""
    
    # Define features based on available columns
    features = []
    if "Occupancy" in df_ml.columns and df_ml["Occupancy"].notna().any():
        features.append("Occupancy")
    if "RadiatorTemp" in df_ml.columns and df_ml["RadiatorTemp"].notna().any():
        features.append("RadiatorTemp")
    if "CO2" in df_ml.columns and df_ml["CO2"].notna().any():
        features.append("CO2")
    if "ExternalTemp" in df_ml.columns and df_ml["ExternalTemp"].notna().any():
        features.append("ExternalTemp")
    if "Hour" in df_ml.columns and df_ml["Hour"].notna().any():
        features.append("Hour")
    if "Weekday" in df_ml.columns and df_ml["Weekday"].notna().any():
        features.append("Weekday")
    
    if "WallTemp" not in df_ml.columns or len(features) < 2:
        return None, None, None, "Insufficient data columns"
    
    # Clean data
    df_clean = df_ml[features + ["WallTemp"]].dropna()
    
    if len(df_clean) < 10:
        return None, None, None, "Insufficient data rows"
    
    X = df_clean[features]
    y = df_clean["WallTemp"]
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        # XGBoost
        try:
            import xgboost as xgb
            xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
            xgb_model.fit(X_train, y_train)
        except ImportError:
            xgb_model = None
        
        # Evaluate models
        rf_pred = rf_model.predict(X_test)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        
        metrics = {
            "rf_mae": rf_mae,
            "rf_r2": rf_r2,
            "features": features,
            "n_samples": len(df_clean)
        }
        
        if xgb_model:
            xgb_pred = xgb_model.predict(X_test)
            xgb_mae = mean_absolute_error(y_test, xgb_pred)
            xgb_r2 = r2_score(y_test, xgb_pred)
            metrics["xgb_mae"] = xgb_mae
            metrics["xgb_r2"] = xgb_r2
        
        return rf_model, xgb_model, metrics, None
        
    except Exception as e:
        return None, None, None, str(e)

# =====================
# ---- MAIN LAYOUT ----
# =====================

# Header with University Branding
st.markdown('<div class="main-header">🏛️ Bologna University - Aula 0.4 (Digital Twin Prototype)</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### 📂 Upload Data (CSV/XLSX)")
    uploaded = st.file_uploader("", type=["csv","xlsx","xls"], label_visibility="collapsed")
    
    if uploaded:
        st.success("✅ File uploaded successfully")

# Main content
if uploaded is None:
    st.info("👈 Please upload a data file to get started")
    st.stop()

# Load and process data
df = load_dataframe(uploaded)
if df.empty:
    st.error("❌ Could not read the uploaded file")
    st.stop()

# =====================
# ---- SENSOR VISUALIZATION ----
# =====================
st.markdown('<div class="sub-header">📊 Sensor Visualization</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("**Select Parameter to Display:**")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        param = st.selectbox("", options=num_cols, label_visibility="collapsed")
    else:
        st.warning("No numerical columns found")
        st.stop()

with col2:
    st.markdown("**Time Scale:**")
    agg = st.selectbox("", ["Raw", "Hourly", "Daily", "Monthly"], label_visibility="collapsed")

# Apply resampling
DT = get_datetime_column(df)
base = df.copy()
if DT and agg != "Raw":
    rule = {"Hourly":"H", "Daily":"D", "Monthly":"MS"}[agg]
    base = resample_df(df, DT, rule)

# Date range filter
if DT and DT in base.columns:
    st.markdown("**📅 Date Range:**")
    base[DT] = pd.to_datetime(base[DT])
    min_dt = base[DT].min()
    max_dt = base[DT].max()
    
    if pd.notna(min_dt) and pd.notna(max_dt):
        min_date = min_dt.date()
        max_date = max_dt.date()
        
        start_date, end_date = st.slider(
            "",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            format="DD/MM/YYYY",
            label_visibility="collapsed"
        )
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        mask = (base[DT] >= start_dt) & (base[DT] <= end_dt)
        filtered = base.loc[mask]
    else:
        filtered = base
else:
    filtered = base

# Display chart
if not filtered.empty and param:
    fig = px.line(
        filtered,
        x=DT if DT and DT in filtered.columns else filtered.index,
        y=param,
        title=f"{param} - Time Series (Monthly)",
        labels={"x": "Time", "y": param}
    )
    fig.update_layout(
        height=400,
        hovermode='x',
        title_font_size=16
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average", f"{filtered[param].mean():.2f}")
    with col2:
        st.metric("Max", f"{filtered[param].max():.2f}")
    with col3:
        st.metric("Min", f"{filtered[param].min():.2f}")
    with col4:
        st.metric("Std Dev", f"{filtered[param].std():.2f}")

st.markdown("---")

# =====================
# ---- AI PREDICTION MODEL ----
# =====================
st.markdown('<div class="sub-header">🤖 AI Prediction Model</div>', unsafe_allow_html=True)

# Auto-prepare data for ML
df_ml = prepare_ml_data(df)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**🔍 Data Inspection**")
    if st.button("Inspect Data", use_container_width=True):
        with st.expander("Data Details", expanded=True):
            st.write("**Original columns:**")
            st.write(list(df.columns))
            
            ml_features = ["WallTemp", "RadiatorTemp", "Occupancy", "CO2", "ExternalTemp", "Hour", "Weekday"]
            available_ml_features = [f for f in ml_features if f in df_ml.columns]
            st.write("**Generated ML features:**")
            st.write(available_ml_features)
            
            if available_ml_features:
                st.write("**Sample data:**")
                st.dataframe(df_ml[available_ml_features].head())

with col2:
    st.markdown("**🎯 Train Model**")
    if st.button("Train Model", use_container_width=True):
        with st.spinner("🔄 Training models..."):
            rf_model, xgb_model, metrics, error = train_models(df_ml)
            
            if error:
                st.error(f"❌ Error: {error}")
            else:
                # Store in session state
                st.session_state['rf_model'] = rf_model
                st.session_state['xgb_model'] = xgb_model
                st.session_state['metrics'] = metrics
                st.session_state['features'] = metrics['features']
                
                st.success("✅ Models trained successfully!")

# Model Information Box
if 'metrics' in st.session_state:
    metrics = st.session_state['metrics']
    
    st.markdown("**Model Information**")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(f"✅ Models Ready\n\nData: {metrics['n_samples']} rows")
    with col_info2:
        st.info(f"🔧 Features: {len(metrics['features'])}\n\nTarget: Wall Temperature")

# =====================
# ---- FEATURE SELECTION ----
# =====================
if 'features' in st.session_state:
    st.markdown("**📋 Select Features ➽ Number of Features: " + str(len(st.session_state['features'])) + " ➕➖**")
    
    features = st.session_state['features']
    feature_cols = st.columns(min(len(features), 3))
    inputs = {}
    
    for i, feature in enumerate(features):
        with feature_cols[i % len(feature_cols)]:
            if feature == "CO2":
                inputs[feature] = st.number_input(f"🌬️ {feature}", value=400, min_value=300, max_value=2000, key=f"input_{feature}")
            elif feature == "Occupancy":
                inputs[feature] = st.number_input(f"👥 {feature}", value=20, min_value=0, max_value=100, key=f"input_{feature}")
            elif feature == "RadiatorTemp":
                inputs[feature] = st.number_input(f"🔥 RadiatorTemp", value=55.0, min_value=0.0, max_value=100.0, key=f"input_{feature}")
            elif feature == "Hour":
                inputs[feature] = st.number_input(f"🕐 {feature}", value=12, min_value=0, max_value=23, key=f"input_{feature}")
            elif feature == "ExternalTemp":
                inputs[feature] = st.number_input(f"🌡️ Ext.Temp", value=15.0, min_value=-20.0, max_value=40.0, key=f"input_{feature}")
            else:
                inputs[feature] = st.number_input(f"📊 {feature}", value=0.0, key=f"input_{feature}")

# =====================
# ---- PREDICTION RESULTS ----
# =====================
if 'rf_model' in st.session_state and 'features' in st.session_state:
    st.markdown("**🎯 Prediction Results**")
    
    if st.button("🔮 Predict", use_container_width=True):
        try:
            rf_model = st.session_state['rf_model']
            xgb_model = st.session_state.get('xgb_model')
            
            # Prepare input data
            input_df = pd.DataFrame([inputs])
            
            # Make predictions
            rf_pred = rf_model.predict(input_df)[0]
            
            col_rf, col_xgb = st.columns(2)
            
            with col_rf:
                st.markdown("**Random Forest**")
                st.markdown(f"### {rf_pred:.1f}°C")
                if 'metrics' in st.session_state:
                    st.caption(f"MAE ± {st.session_state['metrics']['rf_mae']:.2f}°C")
            
            with col_xgb:
                st.markdown("**XGBoost**")
                if xgb_model:
                    xgb_pred = xgb_model.predict(input_df)[0]
                    st.markdown(f"### {xgb_pred:.1f}°C")
                    if 'metrics' in st.session_state and 'xgb_mae' in st.session_state['metrics']:
                        st.caption(f"MAE ± {st.session_state['metrics']['xgb_mae']:.2f}°C")
                else:
                    st.markdown("### Not Available")
                    st.caption("XGBoost not installed")
                    
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

else:
    st.info("🎯 Train the models first to make predictions.")

# Footer
st.markdown("---")
st.caption("🏛️ Bologna University - Digital Twin Prototype - Built with Streamlit")
