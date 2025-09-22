import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import List, Dict, Optional
from anthropic import Anthropic, APIStatusError


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
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .logo-placeholder {
        width: 60px;
        height: 60px;
        background: #1f4e79;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
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
    .parameter-group {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
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

def categorize_sensors(df):
    """Categorizes sensors and calculates averages for each type - Fixed to handle 3 temperature types correctly"""
    sensor_categories = {}
    
    # First, collect all temperature-related columns
    all_temp_cols = [c for c in df.columns if "temperature" in c.lower()]
    
    # 1. Wall Temperature - columns with "wall" and "Temperature" (not "Radiator Temperature")
    wall_temp_cols = [c for c in all_temp_cols 
                      if "wall" in c.lower() 
                      and "temperature" in c.lower()
                      and "radiator temperature" not in c.lower()]
    if wall_temp_cols:
        sensor_categories["Wall Temperature"] = df[wall_temp_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    
    # 2. Radiator Temperature - columns with "Radiator Temperature" specifically
    rad_temp_cols = [c for c in all_temp_cols 
                     if "radiator temperature" in c.lower()]
    if rad_temp_cols:
        sensor_categories["Radiator Temperature"] = df[rad_temp_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    
    # 3. Roof Temperature - columns with "roof" and "Temperature"
    roof_temp_cols = [c for c in all_temp_cols 
                      if "roof" in c.lower() 
                      and "temperature" in c.lower()]
    if roof_temp_cols:
        sensor_categories["Roof Temperature"] = df[roof_temp_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    

    
    # Relative Humidity
    humidity_cols = [c for c in df.columns if "humidity" in c.lower() or "rh" in c.lower()]
    if humidity_cols:
        sensor_categories["Relative Humidity"] = df[humidity_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    
    # CO2
    co2_cols = [c for c in df.columns if "co2" in c.lower()]
    if co2_cols:
        sensor_categories["CO2"] = df[co2_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    
    # TVOC
    tvoc_cols = [c for c in df.columns if "tvoc" in c.lower()]
    if tvoc_cols:
        sensor_categories["TVOC"] = df[tvoc_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    
    # HCHO
    hcho_cols = [c for c in df.columns if "hcho" in c.lower()]
    if hcho_cols:
        sensor_categories["HCHO"] = df[hcho_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    
    # Light
    light_cols = [c for c in df.columns if "light" in c.lower() or "lux" in c.lower()]
    if light_cols:
        sensor_categories["Light"] = df[light_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    
    # Occupancy
    occ_cols = [c for c in df.columns if "occupancy" in c.lower() and "%" not in c]
    if occ_cols:
        sensor_categories["Occupancy"] = df[occ_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    
    return sensor_categories

def prepare_ml_data(df):
    """Prepares data for machine learning with all available features"""
    sensor_data = categorize_sensors(df)
    
    # Add time features if datetime column exists
    dt_col = get_datetime_column(df)
    if dt_col:
        try:
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
            valid_dates = df[dt_col].notna()
            if valid_dates.any():
                sensor_data["Hour"] = df[dt_col].dt.hour
                sensor_data["Weekday"] = df[dt_col].dt.weekday
                sensor_data["Month"] = df[dt_col].dt.month
        except Exception as e:
            st.warning(f"Could not process datetime column '{dt_col}': {str(e)}")
    
    return pd.DataFrame(sensor_data)

def train_custom_model(df_ml, target_var, selected_features):
    """Trains models with user-selected target and features"""
    
    if target_var not in df_ml.columns:
        return None, None, None, f"Target variable '{target_var}' not found"
    
    if len(selected_features) < 1:
        return None, None, None, "At least 1 feature must be selected"
    
    # Check if selected features exist
    available_features = [f for f in selected_features if f in df_ml.columns]
    if len(available_features) < 1:
        return None, None, None, "None of the selected features are available in the data"
    
    # Clean data
    feature_cols = available_features + [target_var]
    df_clean = df_ml[feature_cols].dropna()
    
    if len(df_clean) < 10:
        return None, None, None, "Insufficient data rows (need at least 10)"
    
    X = df_clean[available_features]
    y = df_clean[target_var]
    
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
            "features": available_features,
            "target": target_var,
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
st.markdown('''
<div class="main-header">
    <div class="logo-placeholder">UNIBO</div>
    <div>
        <div style="font-size: 2.5rem;">Bologna University - Aula 0.4</div>
        <div style="font-size: 1.2rem; color: #666;">(Digital Twin Prototype)</div>
    </div>
</div>
''', unsafe_allow_html=True)

st.caption("🏛️ University of Bologna - Founded 1088 - The First University in the World")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### 📂 Upload Data (CSV/XLSX)")
    uploaded = st.file_uploader("", type=["csv","xlsx","xls"], label_visibility="collapsed")

# === Cleaning integration (auto) ===
try:
    _df_exists = 'df' in locals() or 'df' in globals()
except Exception:
    _df_exists = False

if _df_exists and isinstance(df, pd.DataFrame):
    try:
        df = clean_uploaded_dataset(df)
    except Exception as _e:
        # Fail silently to avoid altering UI
        pass
# === End cleaning integration (auto) ===


    
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
    agg = st.selectbox("", ["Hourly", "Daily", "Monthly"], label_visibility="collapsed")

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

# Display first chart
if not filtered.empty and param:
    fig = px.line(
        filtered,
        x=DT if DT and DT in filtered.columns else filtered.index,
        y=param,
        title=f"{param} - Time Series ({agg})",
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
# ---- AVERAGED SENSOR PARAMETERS ----
# =====================
st.markdown('<div class="sub-header">📈 Averaged Sensor Parameters</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("**Select Averaged Parameter:**")
    # Get categorized sensor data first
    sensor_categories = categorize_sensors(df)
    if sensor_categories:
        available_params = list(sensor_categories.keys())
        selected_param = st.selectbox("", options=available_params, key="averaged_param_select", label_visibility="collapsed")
    else:
        st.warning("No sensor categories could be identified from the data")
        st.stop()

with col2:
    st.markdown("**Time Scale:**")
    avg_agg = st.selectbox("", ["Hourly", "Daily", "Monthly"], key="avg_agg_select", label_visibility="collapsed")

# Apply resampling for averaged sensor parameters
if DT and avg_agg != "Raw":
    avg_rule = {"Hourly":"H", "Daily":"D", "Monthly":"MS"}[avg_agg]
    avg_base = resample_df(df, DT, avg_rule)
else:
    avg_base = df.copy()

# Date range filter for averaged sensor parameters
if DT and DT in avg_base.columns:
    st.markdown("**📅 Date Range:**")
    avg_base[DT] = pd.to_datetime(avg_base[DT])
    avg_min_dt = avg_base[DT].min()
    avg_max_dt = avg_base[DT].max()
    
    if pd.notna(avg_min_dt) and pd.notna(avg_max_dt):
        avg_min_date = avg_min_dt.date()
        avg_max_date = avg_max_dt.date()
        
        avg_start_date, avg_end_date = st.slider(
            "",
            value=(avg_min_date, avg_max_date),
            min_value=avg_min_date,
            max_value=avg_max_date,
            format="DD/MM/YYYY",
            key="avg_date_range",
            label_visibility="collapsed"
        )
        
        avg_start_dt = pd.to_datetime(avg_start_date)
        avg_end_dt = pd.to_datetime(avg_end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        avg_mask = (avg_base[DT] >= avg_start_dt) & (avg_base[DT] <= avg_end_dt)
        avg_filtered = avg_base.loc[avg_mask]
    else:
        avg_filtered = avg_base
else:
    avg_filtered = avg_base

# Get categorized sensor data for filtered data
sensor_categories = categorize_sensors(avg_filtered)

if sensor_categories:
    # Display selected parameter chart
    if selected_param in sensor_categories:
        param_data = sensor_categories[selected_param]
        
        if param_data.notna().any():
            # Create single line chart
            fig = px.line(
                x=avg_filtered[DT] if DT and DT in avg_filtered.columns else avg_filtered.index,
                y=param_data,
                title=f"{selected_param} - Averaged Values ({avg_agg})",
                labels={"x": "Time", "y": selected_param}
            )
            
            fig.update_layout(
                height=400,
                hovermode='x',
                title_font_size=16,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show statistics for selected parameter
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average", f"{param_data.mean():.2f}")
            with col2:
                st.metric("Max", f"{param_data.max():.2f}")
            with col3:
                st.metric("Min", f"{param_data.min():.2f}")
            with col4:
                st.metric("Std Dev", f"{param_data.std():.2f}")
        else:
            st.warning(f"No valid data available for {selected_param}")

else:
    st.warning("No sensor categories could be identified from the data")

st.markdown("---")

# =====================
# ---- AI PREDICTION MODEL ----
# =====================
st.markdown('<div class="sub-header">🤖 AI Prediction Model</div>', unsafe_allow_html=True)

# Prepare ML data
df_ml = prepare_ml_data(df)

# Model configuration section
st.markdown("**🎯 Model Configuration**")

col_config1, col_config2 = st.columns(2)

with col_config1:
    st.markdown("**Select Target Variable:**")
    available_targets = [col for col in df_ml.columns if df_ml[col].notna().any()]
    if available_targets:
        target_variable = st.selectbox("What do you want to predict?", available_targets, key="target_select")
    else:
        st.error("No valid target variables found")
        st.stop()

with col_config2:
    st.markdown("**Select Input Features:**")
    available_features = [col for col in df_ml.columns if col != target_variable and df_ml[col].notna().any()]
    if available_features:
        selected_features = st.multiselect(
            "Choose input parameters:",
            available_features,
            default=available_features[:min(5, len(available_features))],
            key="features_select"
        )
    else:
        st.error("No valid features found")
        st.stop()

# Data inspection and training
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**🔍 Data Inspection**")
    if st.button("Inspect Data", use_container_width=True):
        with st.expander("Data Details", expanded=True):
            st.write("**Available parameters:**")
            st.write(list(df_ml.columns))
            
            st.write(f"**Selected target:** {target_variable}")
            st.write(f"**Selected features:** {selected_features}")
            
            if selected_features:
                st.write("**Sample data:**")
                display_cols = selected_features + [target_variable]
                st.dataframe(df_ml[display_cols].head())
                
                st.write("**Data completeness:**")
                completeness = df_ml[display_cols].notna().mean() * 100
                for col in display_cols:
                    st.write(f"- {col}: {completeness[col]:.1f}% complete")

with col2:
    st.markdown("**🎯 Train Model**")
    if st.button("Train Custom Model", use_container_width=True):
        if not selected_features:
            st.error("Please select at least one feature")
        else:
            with st.spinner("🔄 Training models..."):
                rf_model, xgb_model, metrics, error = train_custom_model(df_ml, target_variable, selected_features)
                
                if error:
                    st.error(f"❌ Error: {error}")
                else:
                    # Store in session state
                    st.session_state['rf_model'] = rf_model
                    st.session_state['xgb_model'] = xgb_model
                    st.session_state['metrics'] = metrics
                    st.session_state['selected_features'] = selected_features
                    st.session_state['target_variable'] = target_variable
                    
                    st.success("✅ Models trained successfully!")

# Model Information
if 'metrics' in st.session_state:
    metrics = st.session_state['metrics']
    
    st.markdown("**📈 Model Information**")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(f"✅ Models Ready\n\nTarget: {metrics['target']}\nData: {metrics['n_samples']} rows")
    with col_info2:
        st.info(f"🔧 Features: {len(metrics['features'])}\n\nFeatures: {', '.join(metrics['features'][:3])}{'...' if len(metrics['features']) > 3 else ''}")

# =====================
# ---- FEATURE INPUT ----
# =====================
if 'selected_features' in st.session_state and 'target_variable' in st.session_state:
    st.markdown(f"**📋 Input Features for {st.session_state['target_variable']} Prediction**")
    st.markdown(f"**Number of Features: {len(st.session_state['selected_features'])} ➕➖**")
    
    features = st.session_state['selected_features']
    
    # Create input widgets based on feature type
    feature_cols = st.columns(min(len(features), 3))
    inputs = {}
    
    for i, feature in enumerate(features):
        with feature_cols[i % len(feature_cols)]:
            # Set appropriate defaults and ranges based on feature name
            if "CO2" in feature:
                inputs[feature] = st.number_input(f"🌬️ {feature}", value=400, min_value=300, max_value=2000, key=f"input_{feature}")
            elif "Occupancy" in feature:
                inputs[feature] = st.number_input(f"👥 {feature}", value=20, min_value=0, max_value=100, key=f"input_{feature}")
            elif "Temperature" in feature:
                if "Radiator" in feature:
                    inputs[feature] = st.number_input(f"🔥 {feature}", value=55.0, min_value=0.0, max_value=100.0, key=f"input_{feature}")
                elif "Wall" in feature:
                    inputs[feature] = st.number_input(f"🧱 {feature}", value=20.0, min_value=0.0, max_value=50.0, key=f"input_{feature}")
                elif "Roof" in feature:
                    inputs[feature] = st.number_input(f"🏠 {feature}", value=18.0, min_value=-20.0, max_value=50.0, key=f"input_{feature}")
                else:
                    inputs[feature] = st.number_input(f"🌡️ {feature}", value=22.0, min_value=0.0, max_value=50.0, key=f"input_{feature}")
            elif "Humidity" in feature:
                inputs[feature] = st.number_input(f"💧 {feature}", value=45.0, min_value=0.0, max_value=100.0, key=f"input_{feature}")
            elif "Hour" in feature:
                inputs[feature] = st.number_input(f"🕒 {feature}", value=12, min_value=0, max_value=23, key=f"input_{feature}")
            elif "Weekday" in feature:
                inputs[feature] = st.selectbox(f"📅 {feature}", 
                                             options=[0,1,2,3,4,5,6], 
                                             format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x],
                                             key=f"input_{feature}")
            elif "Light" in feature:
                inputs[feature] = st.number_input(f"💡 {feature}", value=300, min_value=0, max_value=2000, key=f"input_{feature}")
            elif "TVOC" in feature:
                inputs[feature] = st.number_input(f"🧪 {feature}", value=100, min_value=0, max_value=1000, key=f"input_{feature}")
            elif "HCHO" in feature:
                inputs[feature] = st.number_input(f"🧪 {feature}", value=50, min_value=0, max_value=500, key=f"input_{feature}")
            else:
                inputs[feature] = st.number_input(f"📊 {feature}", value=0.0, key=f"input_{feature}")

# =====================
# ---- PREDICTION RESULTS ----
# =====================
if 'rf_model' in st.session_state and 'selected_features' in st.session_state:
    st.markdown("**🎯 Prediction Results**")
    
    if st.button("🔮 Predict", use_container_width=True):
        try:
            rf_model = st.session_state['rf_model']
            xgb_model = st.session_state.get('xgb_model')
            target_var = st.session_state['target_variable']
            
            # Prepare input data
            input_df = pd.DataFrame([inputs])
            
            # Make predictions
            rf_pred = rf_model.predict(input_df)[0]
            
            col_rf, col_xgb = st.columns(2)
            
            with col_rf:
                st.markdown("**Random Forest**")
                # Add units based on target variable
                unit = "°C" if "Temperature" in target_var else ""
                unit = "%" if "Humidity" in target_var else unit
                unit = "ppm" if "CO2" in target_var else unit
                unit = "lux" if "Light" in target_var else unit
                
                st.markdown(f"### {rf_pred:.1f}{unit}")
                if 'metrics' in st.session_state:
                    st.caption(f"MAE ± {st.session_state['metrics']['rf_mae']:.2f}{unit}")
            
            with col_xgb:
                st.markdown("**XGBoost**")
                if xgb_model:
                    xgb_pred = xgb_model.predict(input_df)[0]
                    st.markdown(f"### {xgb_pred:.1f}{unit}")
                    if 'metrics' in st.session_state and 'xgb_mae' in st.session_state['metrics']:
                        st.caption(f"MAE ± {st.session_state['metrics']['xgb_mae']:.2f}{unit}")
                else:
                    st.markdown("### Not Available")
                    st.caption("XGBoost not installed")
                    
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

else:
    st.info("🎯 Configure and train the model first to make predictions.")

# Footer
st.markdown("---")

# =========================
# 🧠 Claude Chat — Enhanced UI
# =========================
import uuid

def clean_uploaded_dataset(df):
    """
    Apply dataset cleaning steps derived from the user's notebook.
    The function is defensive: it only operates on columns it finds.
    It returns a new cleaned DataFrame with the same columns (unless noted).
    """
    import pandas as pd
    import numpy as np

    dfc = df.copy()

    # 1) Try to unify decimal separators and convert numeric-looking columns
    for col in dfc.columns:
        if dfc[col].dtype == object:
            # replace comma decimal to dot
            dfc[col] = dfc[col].str.replace(",", ".", regex=False)
            # strip spaces
            dfc[col] = dfc[col].str.strip()

    # Convert obvious numeric columns
    for col in dfc.columns:
        try:
            dfc[col] = pd.to_numeric(dfc[col])
        except Exception:
            pass

    # 2) Try to parse a datetime index if present
    for candidate in ["timestamp","time","date","datetime","Date","DATETIME","TIMESTAMP"]:
        if candidate in dfc.columns:
            try:
                dfc[candidate] = pd.to_datetime(dfc[candidate], errors="coerce", dayfirst=True)
            except Exception:
                pass
    # set index to first datetime-like column if exists
    dt_col = None
    for c in dfc.columns:
        if str(dfc[c].dtype).startswith("datetime64"):
            dt_col = c
            break
    if dt_col:
        dfc = dfc.sort_values(dt_col).set_index(dt_col)

    # 3) Specific fixes often seen in the user's datasets
    # Relative Humidity scale error like 470.0 -> 47.00
    for col in dfc.columns:
        if "humidity" in col.lower() or "rh" in col.lower():
            # If many values > 100, divide by 10 if looks like factor 10
            over100_ratio = (dfc[col] > 100).mean()
            if over100_ratio > 0.2:
                dfc[col] = dfc[col] / 10.0

    # Temperature outliers starting with 45xxx (e.g., rainfall anomaly pattern)
    for col in dfc.columns:
        if "temp" in col.lower() or "temperature" in col.lower():
            dfc.loc[dfc[col] > 200, col] = np.nan  # drop impossible values

    # TVOC scale normalize if looks like factor 100
    for col in dfc.columns:
        if "tvoc" in col.lower():
            # If median is very large, scale down
            med = pd.to_numeric(dfc[col], errors="coerce").median()
            if med and med > 5000:
                dfc[col] = dfc[col] / 100.0

    # Wind speed "4,5" or "4.5" already handled by decimal unification

    # 4) Interpolate mild gaps for numeric columns
    num_cols = dfc.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        dfc[num_cols] = dfc[num_cols].replace([np.inf, -np.inf], np.nan)
        try:
            dfc[num_cols] = dfc[num_cols].interpolate(limit_direction="both")
        except Exception:
            pass

    # 5) Optionally resample hourly if index is datetime-like and high frequency
    if isinstance(dfc.index, pd.DatetimeIndex):
        try:
            freq = pd.infer_freq(dfc.index[:20])
        except Exception:
            freq = None
        if freq and freq.lower() not in ("h", "1h"):
            # aggregate: mean
            dfc = dfc.resample("1H").mean()

    # 6) Round numeric columns to 2 decimals for consistency
    if len(num_cols) > 0:
        dfc[num_cols] = dfc[num_cols].round(2)

    return dfc


# --- API key ---
API_KEY = (st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or "").strip()
if not API_KEY:
    st.error("ANTHROPIC_API_KEY is missing in Settings → Secrets.")
    st.stop()
client = Anthropic(api_key=API_KEY)

st.markdown('<div class="sub-header">🧠 Claude Chat</div>', unsafe_allow_html=True)

# --- state init ---
if "qa_list" not in st.session_state:
    # her eleman: {"id": str, "q": "...", "a": "..."}
    st.session_state.qa_list = []

# --- toolbar ---
col1, col2 = st.columns([1,1])
with col1:
    keep_latest = st.toggle("Keep only latest", value=False, help="Yeni yanıt gelince eskileri otomatik temizle.")
with col2:
    if st.button("Clear all"):
        st.session_state.qa_list.clear()
        st.rerun()

st.divider()

# --- input ---
prompt = st.chat_input("Ask Claude…")
if prompt:
    # isteği gönder
    try:
        resp = client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = resp.content[0].text
    except APIStatusError as e:
        answer = f"API error {e.status_code}:\n\n{e.response.text}"
    except Exception as e:
        answer = f"{type(e).__name__}: {e}"

    # listeyi güncelle
    if keep_latest:
        st.session_state.qa_list = []  # sadece son yanıt kalsın
    st.session_state.qa_list.append({"id": str(uuid.uuid4()), "q": prompt, "a": answer})
    st.rerun()

# --- render (son soru en üstte) ---
for i, item in enumerate(reversed(st.session_state.qa_list)):
    # benzersiz anahtarlar
    exp_key = f"exp_{item['id']}"
    del_key = f"del_{item['id']}"
    # son öğe açık, diğerleri kapalı
    expanded_default = (i == 0)

    with st.expander(f"Q: {item['q']}", expanded=expanded_default, icon="💬"):
        st.markdown(item["a"])
        # tek öğe silme
        if st.button("Delete this", key=del_key):
            st.session_state.qa_list = [x for x in st.session_state.qa_list if x["id"] != item["id"]]
            st.rerun() 
