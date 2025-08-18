import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime
from typing import List, Dict, Optional

import joblib

st.set_page_config(
    page_title="Sensor Dashboard + RF/XGB Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================
# ---- USER SETTINGS ---
# =====================
# 1) Eğitimde kullandığın feature sırasını buraya yaz veya soldaki text alandan gir.
FEATURE_COLUMNS: List[str] = []
# 2) Hedef adını (ekran etiketi) belirt.
TARGET_NAME = "Prediction"
# 3) Varsayılan model yolları
RF_MODEL_PATH = os.getenv("RF_MODEL_PATH", "rf_model.pkl")
XGB_MODEL_PATH = os.getenv("XGB_MODEL_PATH", "xgb_model.pkl")

# =====================
# ---- HELPERS --------
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
    # Try several datetime candidates
    dt_candidates = [c for c in df.columns if any(x in str(c).lower() for x in ["datetime","time","timestamp","date"])]
    for c in dt_candidates:
        try:
            df[c] = pd.to_datetime(df[c], errors="raise")
            df = df.sort_values(c)
            return df
        except Exception:
            continue
    return df

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Model could not be loaded from {path}: {e}")
        return None

@st.cache_data(show_spinner=False)
def resample_df(df: pd.DataFrame, dt_col: Optional[str], rule: str) -> pd.DataFrame:
    if dt_col is None:
        return df
    g = df.set_index(dt_col)
    # numeric mean only for resample
    num_cols = g.select_dtypes(include=[np.number]).columns
    out = g[num_cols].resample(rule).mean().reset_index()
    return out

def get_datetime_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    return None

def build_feature_input_ui(columns: List[str]) -> pd.DataFrame:
    values: Dict[str, float] = {}
    if not columns:
        return pd.DataFrame()
    cols = st.columns(min(4, max(1, len(columns))))
    for i, c in enumerate(columns):
        with cols[i % len(cols)]:
            v = st.number_input(c, value=0.0, step=0.1, format="%.4f")
            values[c] = v
    return pd.DataFrame([values])

def align_features(input_df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    """Reorder/align columns to match training feature order. Missing -> 0, extras dropped."""
    aligned = pd.DataFrame()
    for col in feature_list:
        aligned[col] = input_df[col] if col in input_df.columns else 0.0
    return aligned[feature_list]

# =====================
# ---- SIDEBAR UI -----
# =====================
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Veri yükle (CSV/XLSX)", type=["csv","xlsx","xls"], accept_multiple_files=False)

    st.subheader("📦 Modeller")
    rf_path = st.text_input("RandomForest model yolu", value=RF_MODEL_PATH)
    xgb_path = st.text_input("XGBoost model yolu", value=XGB_MODEL_PATH)

    st.subheader("🎯 Özellikler (FEATURE_COLUMNS)")
    feats_text = st.text_area(
        "Virgülle ayır",
        value=", ".join(FEATURE_COLUMNS) if FEATURE_COLUMNS else "",
        height=80,
        placeholder="HourOfDay, MonthNum, Weekday, Temperature, RelativeHumidity, CO2",
    )

    st.subheader("⏱️ Toplama Seviyesi (Grafik)")
    agg = st.selectbox("Zaman ölçeği", ["Ham", "Saatlik", "Günlük", "Aylık"], index=0)

# Parse features
if feats_text.strip():
    FEATURE_COLUMNS = [f.strip() for f in feats_text.split(",") if f.strip()]

# =====================
# ---- MAIN LAYOUT ----
# =====================
st.title("📊 Sensor Dashboard + 🤖 RF/XGB Tahmin")

# Load data
df = load_dataframe(uploaded)
if df.empty:
    st.info("Soldan veri dosyası yükle.")
    st.stop()

# Identify datetime col and apply resampling if needed
DT = get_datetime_column(df)
base = df.copy()
if DT and agg != "Ham":
    rule = {"Saatlik":"H", "Günlük":"D", "Aylık":"MS"}[agg]
    base = resample_df(df, DT, rule)

# Preview
with st.expander("Veri Önizleme", expanded=False):
    st.write(base.head())
    st.caption(f"Satır: {len(base):,} | Sütun: {len(base.columns):,}")

# =============
# CHART SECTION
# =============
st.subheader("📈 Parametre Grafiği")
num_cols = base.select_dtypes(include=[np.number]).columns.tolist()
if not num_cols:
    st.warning("Sayısal sütun yok.")
else:
    param = st.selectbox("Parametre seç", options=num_cols, index=0)

    # Optional date filter - FIXED SECTION
    filtered = base
    if DT and DT in base.columns:
        # Convert to datetime if needed and ensure consistent types
        base[DT] = pd.to_datetime(base[DT])
        
        # Get min/max as datetime objects
        min_dt = base[DT].min()
        max_dt = base[DT].max()
        
        # Ensure both are pandas Timestamp objects
        if pd.isna(min_dt) or pd.isna(max_dt):
            st.warning("Tarih sütununda geçersiz değerler var.")
        else:
            # Convert to datetime.date for slider compatibility
            min_date = min_dt.date() if hasattr(min_dt, 'date') else min_dt
            max_date = max_dt.date() if hasattr(max_dt, 'date') else max_dt
            
            start_date, end_date = st.slider(
                "Tarih aralığı", 
                value=(min_date, max_date), 
                min_value=min_date, 
                max_value=max_date,
                format="DD/MM/YYYY"
            )
            
            # Convert back to datetime for filtering
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            
            mask = (base[DT] >= start_dt) & (base[DT] <= end_dt)
            filtered = base.loc[mask]

    if not filtered.empty:
        fig = px.line(
            filtered, 
            x=DT if DT and DT in filtered.columns else filtered.index, 
            y=param, 
            title=f"{param} Zaman Serisi"
        )
        fig.update_layout(
            xaxis_title="Zaman" if DT else "Index",
            yaxis_title=param,
            hovermode='x'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show some stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Ortalama", f"{filtered[param].mean():.2f}")
        with col2:
            st.metric("Maksimum", f"{filtered[param].max():.2f}")
        with col3:
            st.metric("Minimum", f"{filtered[param].min():.2f}")
        with col4:
            st.metric("Standart Sapma", f"{filtered[param].std():.2f}")
    else:
        st.warning("Seçilen tarih aralığında veri yok.")

# ==================
# PREDICTION SECTION
# ==================
st.subheader("🔮 Tahmin Ekranı (RF & XGB)")
rf_model = load_model(rf_path)
xgb_model = load_model(xgb_path)

# Single input form
st.markdown("**Tek Satır Giriş** – Feature alanlarını doldur ve modellerle tahmin al.")
if FEATURE_COLUMNS:
    single_input = build_feature_input_ui(FEATURE_COLUMNS)
    aligned_single = align_features(single_input, FEATURE_COLUMNS)
else:
    st.error("FEATURE_COLUMNS boş. Soldan gir veya kodda tanımla.")
    aligned_single = pd.DataFrame()

col1, col2 = st.columns(2)
with col1:
    st.markdown("**RandomForest**")
    if rf_model is not None and not aligned_single.empty:
        try:
            yhat = rf_model.predict(aligned_single)[0]
            st.metric(label=f"{TARGET_NAME} (RF)", value=f"{yhat:.4f}")
        except Exception as e:
            st.error(f"RF tahmin hatası: {e}")
    else:
        st.info("RF modeli yok veya girişler eksik.")

with col2:
    st.markdown("**XGBoost**")
    if xgb_model is not None and not aligned_single.empty:
        try:
            yhat = xgb_model.predict(aligned_single)[0]
            st.metric(label=f"{TARGET_NAME} (XGB)", value=f"{yhat:.4f}")
        except Exception as e:
            st.error(f"XGB tahmin hatası: {e}")
    else:
        st.info("XGB modeli yok veya girişler eksik.")

st.divider()

# Batch prediction from dataset
st.markdown("### 📦 Toplu Tahmin (Dataset'ten)")
st.caption("Veri setinde FEATURE_COLUMNS mevcutsa, aynı sırayla kullanılarak tahmin edilir. Eksik olanlar 0 kabul edilir.")
if FEATURE_COLUMNS:
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if available:
        X_all = pd.DataFrame()
        for c in FEATURE_COLUMNS:
            X_all[c] = df[c] if c in df.columns else 0.0
        pred_cols = []
        if rf_model is not None:
            try:
                df["pred_rf"] = rf_model.predict(X_all)
                pred_cols.append("pred_rf")
            except Exception as e:
                st.error(f"RF toplu tahmin hatası: {e}")
        if xgb_model is not None:
            try:
                df["pred_xgb"] = xgb_model.predict(X_all)
                pred_cols.append("pred_xgb")
            except Exception as e:
                st.error(f"XGB toplu tahmin hatası: {e}")
        if pred_cols:
            st.success(f"Toplu tahmin tamamlandı. Eksik feature sayısı: {len(missing)}")
            st.dataframe(df[[*available, *pred_cols]].head())
            # Download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Sonuçları CSV olarak indir", data=csv, file_name="predictions.csv", mime="text/csv")
    else:
        st.info("Dataset'te FEATURE_COLUMNS bulunamadı. İsimleri eşleştir.")
else:
    st.info("FEATURE_COLUMNS tanımlı değil.")

st.caption("Not: Modeller sklearn Pipeline ise (ör. scaler + model) doğrudan yüklenip çalışır.")
