import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime
from typing import List, Dict, Optional

st.set_page_config(
    page_title="Digital Twin - Sensor Dashboard + AI Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================
# ---- SIDEBAR UI -----
# =====================
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Upload Data (CSV/XLSX)", type=["csv","xlsx","xls"], accept_multiple_files=False)
    
    st.subheader("⏱️ Aggregation Level")
    agg = st.selectbox("Time Scale", ["Raw", "Hourly", "Daily", "Monthly"], index=0)

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
    
    # Try to find datetime column
    dt_candidates = [c for c in df.columns if any(x in str(c).lower() for x in ["datetime","time","timestamp","date","hour"])]
    for c in dt_candidates:
        try:
            df[c] = pd.to_datetime(df[c], errors="raise")
            df = df.sort_values(c)
            break
        except Exception:
            continue
    return df

def get_datetime_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
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
    
    # 6. Time features
    dt_col = get_datetime_column(df_ml)
    if dt_col:
        df_ml["Hour"] = df_ml[dt_col].dt.hour
        df_ml["Weekday"] = df_ml[dt_col].dt.weekday
        df_ml["Month"] = df_ml[dt_col].dt.month
    
    return df_ml

def train_models(df_ml):
    """Trains Random Forest and XGBoost models"""
    
    # Define features
    features = []
    if "Occupancy" in df_ml.columns:
        features.append("Occupancy")
    if "RadiatorTemp" in df_ml.columns:
        features.append("RadiatorTemp")
    if "CO2" in df_ml.columns:
        features.append("CO2")
    if "ExternalTemp" in df_ml.columns:
        features.append("ExternalTemp")
    if "Hour" in df_ml.columns:
        features.append("Hour")
    if "Weekday" in df_ml.columns:
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
        
        # Evaluate
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
st.title("🏠 Digital Twin Prototype")
st.markdown("**Sensor Dashboard + AI Prediction System**")

# Load data
df = load_dataframe(uploaded)
if df.empty:
    st.info("👈 Please upload a data file (CSV/XLSX) from the sidebar")
    st.stop()

# Apply resampling
DT = get_datetime_column(df)
base = df.copy()
if DT and agg != "Raw":
    rule = {"Hourly":"H", "Daily":"D", "Monthly":"MS"}[agg]
    base = resample_df(df, DT, rule)

# Data preview
with st.expander("📊 Data Preview", expanded=False):
    st.dataframe(base.head())
    st.caption(f"Rows: {len(base):,} | Columns: {len(base.columns):,}")

# =====================
# ---- CHARTS --------
# =====================
st.subheader("📈 Sensor Visualization")

num_cols = base.select_dtypes(include=[np.number]).columns.tolist()
if not num_cols:
    st.warning("No numerical columns found.")
else:
    # Parameter selection
    param = st.selectbox("📊 Select parameter to display:", options=num_cols, index=0)
    
    # Date filter
    filtered = base
    if DT and DT in base.columns:
        base[DT] = pd.to_datetime(base[DT])
        min_dt = base[DT].min()
        max_dt = base[DT].max()
        
        if pd.notna(min_dt) and pd.notna(max_dt):
            min_date = min_dt.date()
            max_date = max_dt.date()
            
            start_date, end_date = st.slider(
                "📅 Date range:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                format="DD/MM/YYYY"
            )
            
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            mask = (base[DT] >= start_dt) & (base[DT] <= end_dt)
            filtered = base.loc[mask]
    
    if not filtered.empty:
        # Chart
        fig = px.line(
            filtered,
            x=DT if DT and DT in filtered.columns else filtered.index,
            y=param,
            title=f"{param} - Time Series",
            labels={"x": "Time", "y": param}
        )
        fig.update_layout(hovermode='x')
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Average", f"{filtered[param].mean():.2f}")
        with col2:
            st.metric("⬆️ Maximum", f"{filtered[param].max():.2f}")
        with col3:
            st.metric("⬇️ Minimum", f"{filtered[param].min():.2f}")
        with col4:
            st.metric("📈 Std Dev", f"{filtered[param].std():.2f}")

# =====================
# ---- AI MODELS -----
# =====================
st.subheader("🤖 AI Prediction Models")

# Auto-prepare data for ML
df_ml = prepare_ml_data(df)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🎯 Train Models", use_container_width=True):
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
                
                # Show metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("🌲 RF R²", f"{metrics['rf_r2']:.3f}")
                    st.metric("🌲 RF MAE", f"{metrics['rf_mae']:.2f}°C")
                with col_b:
                    if 'xgb_r2' in metrics:
                        st.metric("🚀 XGB R²", f"{metrics['xgb_r2']:.3f}")
                        st.metric("🚀 XGB MAE", f"{metrics['xgb_mae']:.2f}°C")
                
                st.info(f"📋 Features used: {', '.join(metrics['features'])}")
                st.info(f"📊 Training data: {metrics['n_samples']} rows")

with col2:
    st.markdown("**📈 Model Information**")
    if 'metrics' in st.session_state:
        metrics = st.session_state['metrics']
        st.write("✅ Models ready")
        st.write(f"🎯 Target: Wall Temperature")
        st.write(f"📊 Data: {metrics['n_samples']} rows")
        st.write(f"🔧 Features: {len(metrics['features'])}")
    else:
        st.write("❌ Models not trained yet")
        st.write("👆 Click the button above")

# =====================
# ---- PREDICTION ----
# =====================
if 'rf_model' in st.session_state and 'features' in st.session_state:
    st.subheader("🔮 Make Predictions")
    
    features = st.session_state['features']
    rf_model = st.session_state['rf_model']
    xgb_model = st.session_state.get('xgb_model')
    
    # Create input form
    st.markdown("**📝 Enter values:**")
    
    cols = st.columns(min(len(features), 4))
    inputs = {}
    
    for i, feature in enumerate(features):
        with cols[i % len(cols)]:
            if feature == "Occupancy":
                inputs[feature] = st.number_input(f"👥 {feature}", value=20, min_value=0, max_value=100)
            elif feature == "RadiatorTemp":
                inputs[feature] = st.number_input(f"🔥 {feature}", value=55.0, min_value=0.0, max_value=100.0)
            elif feature == "CO2":
                inputs[feature] = st.number_input(f"🌬️ {feature}", value=400, min_value=300, max_value=2000)
            elif feature == "ExternalTemp":
                inputs[feature] = st.number_input(f"🌡️ {feature}", value=15.0, min_value=-20.0, max_value=40.0)
            elif feature == "Hour":
                inputs[feature] = st.number_input(f"🕐 {feature}", value=12, min_value=0, max_value=23)
            elif feature == "Weekday":
                inputs[feature] = st.selectbox(f"📅 {feature}", 
                                             options=[0,1,2,3,4,5,6], 
                                             format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
            else:
                inputs[feature] = st.number_input(f"📊 {feature}", value=0.0)
    
    # Predict button
    if st.button("🎯 Predict", use_container_width=True):
        # Prepare input data
        input_df = pd.DataFrame([inputs])
        
        try:
            # RF Prediction
            rf_pred = rf_model.predict(input_df)[0]
            
            # XGB Prediction
            xgb_pred = None
            if xgb_model:
                xgb_pred = xgb_model.predict(input_df)[0]
            
            # Show results
            st.markdown("### 🎯 Prediction Results")
            
            col_rf, col_xgb = st.columns(2)
            
            with col_rf:
                st.metric(
                    label="🌲 Random Forest",
                    value=f"{rf_pred:.1f}°C",
                    delta=None
                )
            
            with col_xgb:
                if xgb_pred:
                    st.metric(
                        label="🚀 XGBoost", 
                        value=f"{xgb_pred:.1f}°C",
                        delta=f"{xgb_pred-rf_pred:.1f}°C"
                    )
                else:
                    st.info("XGBoost not available")
            
            # Show input summary
            with st.expander("📋 Input Summary"):
                for feature, value in inputs.items():
                    st.write(f"**{feature}:** {value}")
                    
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

else:
    st.info("🎯 Train the models first to make predictions.")

# Footer
st.markdown("---")
st.caption("🏠 Digital Twin Prototype - Built with Streamlit")
