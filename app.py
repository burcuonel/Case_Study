import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import re

# Page configuration
st.set_page_config(
    page_title="Bologna University - Aula 0.4",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Bologna University branding
st.markdown("""
<style>
    .main-header {
        display: flex;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 2px solid #003c7f;
    }
    .logo {
        width: 60px;
        height: 60px;
        margin-right: 20px;
    }
    .title-container {
        flex-grow: 1;
    }
    .main-title {
        color: #003c7f;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    .subtitle {
        color: #666;
        font-size: 1.2rem;
        margin: 0;
    }
    .university-info {
        color: #888;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header with logo
st.markdown("""
<div class="main-header">
    <div class="logo">🏛️</div>
    <div class="title-container">
        <h1 class="main-title">Bologna University - Aula 0.4</h1>
        <p class="subtitle">(Digital Twin Prototype)</p>
        <p class="university-info">🏛️ University of Bologna · Founded 1088 · The First University in the World</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Session state for cleaned data
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None

# Data Cleaning Section (NEW)
with st.expander("🔧 **Data Cleaning Module** (Process raw data before visualization)", expanded=False):
    st.info("Upload your raw dataset (dataset_2.xlsx) to clean and prepare it for visualization")
    
    raw_file = st.file_uploader(
        "Upload raw Excel file for cleaning",
        type=['xlsx'],
        key="raw_cleaner",
        help="Upload dataset_2.xlsx for automatic cleaning"
    )
    
    if raw_file is not None:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"✅ Raw file uploaded: {raw_file.name}")
        with col2:
            if st.button("🚀 Clean Data", type="primary"):
                with st.spinner("Cleaning data... Please wait"):
                    try:
                        # Read Excel
                        df = pd.read_excel(raw_file)
                        
                        # Combine Date + Time
                        df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
                        df['Hour'] = df['Datetime'].dt.floor('h')
                        
                        # Remove Electrical Panel columns
                        electrical_cols = [col for col in df.columns if 'electrical panel' in col.lower()]
                        if electrical_cols:
                            df = df.drop(columns=electrical_cols)
                        
                        # Remove January
                        df = df[df['Datetime'].dt.month != 1]
                        
                        # Fix Rainfall Amount
                        rain_cols = [c for c in df.columns if "rainfall amount" in str(c).lower()]
                        if rain_cols:
                            pattern_5digit = r"^\s*\d{5}([.,]\d+)?\s*$"
                            for col in rain_cols:
                                raw = df[col].astype(str).str.strip()
                                mask_5 = raw.str.match(pattern_5digit)
                                num = pd.to_numeric(raw.str.replace(",", ".", regex=False), errors="coerce")
                                num.loc[mask_5] = num.loc[mask_5] / 1000.0
                                df[col] = num.round(2)
                        
                        # Fix Temperature sensor errors
                        temperature_columns_df = [col for col in df.columns if "Temperature" in col]
                        for col in temperature_columns_df:
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                            df.loc[df[col].astype(str).str.startswith("45"), col] = np.nan
                        
                        # Calculate hourly averages
                        numeric_cols = df.select_dtypes(include='number').columns.difference(['Hour'])
                        hourly_avg = df.groupby('Hour', as_index=True)[numeric_cols].mean()
                        result = hourly_avg.reset_index()
                        
                        # Filter weekdays 8-19
                        result['Weekday'] = result['Hour'].dt.weekday
                        result['HourOfDay'] = result['Hour'].dt.hour
                        result = result[(result['Weekday'] <= 4) & 
                                       (result['HourOfDay'] >= 8) & 
                                       (result['HourOfDay'] <= 19)]
                        result = result.drop(columns=['Weekday', 'HourOfDay'])
                        
                        # Fix Temperature values
                        temperature_columns = [col for col in result.columns if "Temperature" in col]
                        for col in temperature_columns:
                            result[col] = result[col].apply(lambda x: int(str(int(x))[:2]) if pd.notna(x) else np.nan)
                        
                        # Fix TVOC
                        tvoc_columns = [col for col in result.columns if 'TVOC' in col]
                        for col in tvoc_columns:
                            result[col] = (result[col] / 100).round(2)
                        
                        # Fix Wind Speed
                        wind_columns = [col for col in result.columns if 'Wind Speed' in col]
                        for col in wind_columns:
                            result[col] = result[col].apply(
                                lambda x: round(float(str(int(x))[:1] + '.' + str(int(x))[1:2]), 2) if pd.notna(x) else np.nan
                            )
                        
                        # Round other columns
                        other_numeric_cols = [col for col in result.select_dtypes(include='number').columns
                                             if "Temperature" not in col and "Wind Speed" not in col]
                        for col in other_numeric_cols:
                            result[col] = result[col].round(2)
                        
                        # Interpolation
                        occupancy_columns = [col for col in result.columns if 'Occupancy' in col]
                        numeric_cols_for_interp = result.select_dtypes(include='number').columns.difference(['Hour'])
                        numeric_cols_for_interp = numeric_cols_for_interp.difference(occupancy_columns)
                        
                        for col in numeric_cols_for_interp:
                            result[col] = result[col].interpolate(method='linear', limit_direction='both')
                            result[col] = result[col].ffill().bfill()
                        
                        for col in occupancy_columns:
                            result[col] = result[col].fillna(0)
                        
                        # Save to session state
                        st.session_state.cleaned_data = result
                        
                        # Download button
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            result.to_excel(writer, index=False, sheet_name='Cleaned_Data')
                        output.seek(0)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Records", f"{len(result):,}")
                        with col2:
                            st.metric("Total Columns", len(result.columns))
                        with col3:
                            st.download_button(
                                label="📥 Download Cleaned Data",
                                data=output,
                                file_name="cleaned_data.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
                        st.success("✅ Data cleaning completed! You can now use this data for visualization below.")
                        
                    except Exception as e:
                        st.error(f"Error during cleaning: {str(e)}")

st.markdown("---")

# Original Upload Data Section
with st.sidebar:
    st.markdown("## 📁 Upload Data (CSV/XLSX)")
    
    # Check if cleaned data exists
    if st.session_state.cleaned_data is not None:
        use_cleaned = st.checkbox("Use cleaned data from above", value=True)
        if use_cleaned:
            uploaded_file = "cleaned_data"
            df = st.session_state.cleaned_data
            st.success("Using cleaned data")
        else:
            uploaded_file = st.file_uploader(
                "Drag and drop file here",
                type=['csv', 'xlsx', 'xls'],
                help="Limit 200MB per file • CSV, XLSX, XLS"
            )
    else:
        uploaded_file = st.file_uploader(
            "Drag and drop file here",
            type=['csv', 'xlsx', 'xls'],
            help="Limit 200MB per file • CSV, XLSX, XLS"
        )

# Main content area
if uploaded_file is not None:
    try:
        # Load data if not using cleaned data
        if uploaded_file != "cleaned_data":
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        
        # Ensure Hour column is datetime
        if 'Hour' in df.columns:
            df['Hour'] = pd.to_datetime(df['Hour'])
        
        # Display data info
        st.info(f"📊 **Data loaded successfully!** Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Get numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Time Series", "📊 Distribution", "🔥 Heatmap", "📉 Correlation", "📋 Data View"])
        
        with tab1:
            st.subheader("Time Series Analysis")
            
            if 'Hour' in df.columns:
                col1, col2 = st.columns([3, 1])
                with col1:
                    selected_metrics = st.multiselect(
                        "Select metrics to visualize:",
                        numeric_cols,
                        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                    )
                
                with col2:
                    chart_type = st.radio("Chart type:", ["Lines", "Scatter", "Area"])
                
                if selected_metrics:
                    fig = go.Figure()
                    
                    for metric in selected_metrics:
                        if chart_type == "Lines":
                            fig.add_trace(go.Scatter(
                                x=df['Hour'],
                                y=df[metric],
                                mode='lines',
                                name=metric
                            ))
                        elif chart_type == "Scatter":
                            fig.add_trace(go.Scatter(
                                x=df['Hour'],
                                y=df[metric],
                                mode='markers',
                                name=metric
                            ))
                        else:
                            fig.add_trace(go.Scatter(
                                x=df['Hour'],
                                y=df[metric],
                                mode='lines',
                                fill='tozeroy',
                                name=metric
                            ))
                    
                    fig.update_layout(
                        title="Sensor Measurements Over Time",
                        xaxis_title="Time",
                        yaxis_title="Value",
                        hovermode='x unified',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    st.subheader("📊 Quick Statistics")
                    stats_df = df[selected_metrics].describe().round(2)
                    st.dataframe(stats_df)
        
        with tab2:
            st.subheader("Distribution Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                selected_col = st.selectbox("Select column:", numeric_cols)
            with col2:
                bins = st.slider("Number of bins:", 10, 50, 20)
            
            if selected_col:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_hist = px.histogram(
                        df,
                        x=selected_col,
                        nbins=bins,
                        title=f"Distribution of {selected_col}"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    fig_box = px.box(
                        df,
                        y=selected_col,
                        title=f"Box Plot of {selected_col}"
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
        
        with tab3:
            st.subheader("Correlation Heatmap")
            
            # Select columns for correlation
            corr_cols = st.multiselect(
                "Select columns for correlation analysis:",
                numeric_cols,
                default=numeric_cols[:10] if len(numeric_cols) >= 10 else numeric_cols
            )
            
            if len(corr_cols) > 1:
                corr_matrix = df[corr_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Matrix",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Scatter Plot Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                x_var = st.selectbox("X axis:", numeric_cols, key="scatter_x")
            with col2:
                y_var = st.selectbox("Y axis:", numeric_cols, key="scatter_y")
            with col3:
                color_var = st.selectbox("Color by (optional):", [None] + categorical_cols)
            
            fig = px.scatter(
                df,
                x=x_var,
                y=y_var,
                color=color_var,
                title=f"{y_var} vs {x_var}",
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            st.subheader("Data Overview")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
                st.metric("Total Columns", len(df.columns))
                st.metric("Numeric Columns", len(numeric_cols))
                st.metric("Categorical Columns", len(categorical_cols))
            
            with col2:
                st.dataframe(df.head(100), height=400)
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

else:
    # Welcome message
    st.markdown("""
    <div style='text-align: center; padding: 50px; background-color: #f0f2f6; border-radius: 10px;'>
        <h2>📊 Please upload a data file to get started</h2>
        <p style='color: #666; font-size: 1.1rem;'>
            Use the sidebar to upload your CSV or Excel file<br>
            Or use the Data Cleaning Module above to process raw data first
        </p>
    </div>
    """, unsafe_allow_html=True)
