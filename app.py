import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import re

# Sayfa yapılandırması
st.set_page_config(
    page_title="Okul Sensör Verileri Analizi",
    page_icon="🏫",
    layout="wide"
)

# Başlık
st.title("🏫 Okul Sensör Verileri Analiz Platformu")
st.markdown("---")

# Session state başlatma
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None

# Ana sekmeler
main_tab1, main_tab2, main_tab3 = st.tabs(["🔧 Veri Temizleme", "📊 Görselleştirme", "📈 Detaylı Analiz"])

# VERİ TEMİZLEME SEKMESİ
with main_tab1:
    st.header("Veri Temizleme Modülü")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info("""
        **Temizleme İşlemleri:**
        - ✅ Ocak ayı verileri çıkarılır
        - ✅ Hafta içi 08:00-19:00 arası filtresi
        - ✅ Electrical Panel sütunları silinir
        - ✅ Sensör hataları düzeltilir
        - ✅ Eksik değerler interpolasyon ile doldurulur
        - ✅ Occupancy NaN → 0
        """)
    
    with col2:
        # Ham veri yükleme
        raw_file = st.file_uploader(
            "📁 Ham Excel dosyasını yükleyin (dataset_2.xlsx)",
            type=['xlsx'],
            key="raw_data_uploader"
        )
    
    if raw_file is not None:
        if st.button("🚀 Temizleme İşlemini Başlat", type="primary", key="clean_button"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 1. Read the Excel file
                status_text.text("📖 Excel dosyası okunuyor...")
                progress_bar.progress(10)
                df = pd.read_excel(raw_file)
                
                # 2. Combine Date + Time columns
                status_text.text("📅 Tarih ve saat birleştiriliyor...")
                progress_bar.progress(20)
                df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
                df['Hour'] = df['Datetime'].dt.floor('h')
                
                # 3. ELECTRICAL PANEL SÜTUNLARINI SİL
                status_text.text("⚡ Electrical Panel sütunları siliniyor...")
                progress_bar.progress(30)
                electrical_cols = [col for col in df.columns if 'electrical panel' in col.lower()]
                if electrical_cols:
                    df = df.drop(columns=electrical_cols)
                
                # 4. OCAK AYINI ÇIKAR
                status_text.text("📅 Ocak ayı çıkarılıyor...")
                progress_bar.progress(40)
                df = df[df['Datetime'].dt.month != 1]
                
                # 5. Rainfall Amount düzeltme
                status_text.text("🌧️ Rainfall Amount düzeltiliyor...")
                progress_bar.progress(50)
                rain_cols = [c for c in df.columns if "rainfall amount" in str(c).lower()]
                if rain_cols:
                    pattern_5digit = r"^\s*\d{5}([.,]\d+)?\s*$"
                    for col in rain_cols:
                        raw = df[col].astype(str).str.strip()
                        mask_5 = raw.str.match(pattern_5digit)
                        num = pd.to_numeric(raw.str.replace(",", ".", regex=False), errors="coerce")
                        num.loc[mask_5] = num.loc[mask_5] / 1000.0
                        df[col] = num.round(2)
                
                # 6. Temperature sensor hatalarını düzelt
                status_text.text("🌡️ Sıcaklık sensör hataları düzeltiliyor...")
                progress_bar.progress(60)
                temperature_columns_df = [col for col in df.columns if "Temperature" in col]
                for col in temperature_columns_df:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    df.loc[df[col].astype(str).str.startswith("45"), col] = np.nan
                
                # 7. Saatlik ortalamalar
                status_text.text("⏰ Saatlik ortalamalar hesaplanıyor...")
                progress_bar.progress(70)
                numeric_cols = df.select_dtypes(include='number').columns.difference(['Hour'])
                hourly_avg = df.groupby('Hour', as_index=True)[numeric_cols].mean()
                result = hourly_avg.reset_index()
                
                # 8. Hafta içi ve saat filtrelemesi
                status_text.text("📅 Hafta içi 08:00-19:00 filtreleniyor...")
                progress_bar.progress(75)
                result['Weekday'] = result['Hour'].dt.weekday
                result['HourOfDay'] = result['Hour'].dt.hour
                result = result[(result['Weekday'] <= 4) & 
                               (result['HourOfDay'] >= 8) & 
                               (result['HourOfDay'] <= 19)]
                result = result.drop(columns=['Weekday', 'HourOfDay'])
                
                # 9. Temperature değerlerini düzelt
                status_text.text("🌡️ Sıcaklık değerleri düzeltiliyor...")
                progress_bar.progress(80)
                temperature_columns = [col for col in result.columns if "Temperature" in col]
                for col in temperature_columns:
                    result[col] = result[col].apply(lambda x: int(str(int(x))[:2]) if pd.notna(x) else np.nan)
                
                # 10. TVOC değerlerini düzelt
                tvoc_columns = [col for col in result.columns if 'TVOC' in col]
                for col in tvoc_columns:
                    result[col] = (result[col] / 100).round(2)
                
                # 11. Wind Speed değerlerini düzelt
                wind_columns = [col for col in result.columns if 'Wind Speed' in col]
                for col in wind_columns:
                    result[col] = result[col].apply(
                        lambda x: round(float(str(int(x))[:1] + '.' + str(int(x))[1:2]), 2) if pd.notna(x) else np.nan
                    )
                
                # 12. Diğer numeric kolonları yuvarla
                other_numeric_cols = [col for col in result.select_dtypes(include='number').columns
                                     if "Temperature" not in col and "Wind Speed" not in col]
                for col in other_numeric_cols:
                    result[col] = result[col].round(2)
                
                # 13. INTERPOLASYON
                status_text.text("📊 Eksik değerler dolduruluyor...")
                progress_bar.progress(90)
                
                # Occupancy sütunlarını bul
                occupancy_columns = [col for col in result.columns if 'Occupancy' in col]
                
                # Hour ve Occupancy hariç interpolasyon
                numeric_cols_for_interp = result.select_dtypes(include='number').columns.difference(['Hour'])
                numeric_cols_for_interp = numeric_cols_for_interp.difference(occupancy_columns)
                
                for col in numeric_cols_for_interp:
                    result[col] = result[col].interpolate(method='linear', limit_direction='both')
                    result[col] = result[col].ffill().bfill()
                
                # Occupancy NaN → 0
                for col in occupancy_columns:
                    result[col] = result[col].fillna(0)
                
                # Session state'e kaydet
                st.session_state.cleaned_data = result
                
                progress_bar.progress(100)
                status_text.text("✅ Temizleme tamamlandı!")
                
                # Özet bilgiler
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Toplam Kayıt", f"{len(result):,}")
                with col2:
                    st.metric("Sütun Sayısı", len(result.columns))
                with col3:
                    st.metric("Başlangıç", result['Hour'].min().strftime('%Y-%m-%d'))
                with col4:
                    st.metric("Bitiş", result['Hour'].max().strftime('%Y-%m-%d'))
                
                # İndirme butonu
                st.success("✅ Veri başarıyla temizlendi!")
                
                # Excel indirme
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    result.to_excel(writer, index=False, sheet_name='Temizlenmiş Veri')
                output.seek(0)
                
                st.download_button(
                    label="📥 Temizlenmiş Veriyi İndir (Excel)",
                    data=output,
                    file_name=f"temizlenmis_veri_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # Veri önizleme
                with st.expander("👁️ Temizlenmiş Veri Önizleme"):
                    st.dataframe(result.head(20))
                
            except Exception as e:
                st.error(f"❌ Hata oluştu: {str(e)}")
                progress_bar.empty()
                status_text.empty()

# GÖRSELLEŞTİRME SEKMESİ
with main_tab2:
    st.header("Veri Görselleştirme")
    
    # Veri kaynağı seçimi
    data_source = st.radio(
        "Veri Kaynağı Seçin:",
        ["Temizlenmiş Veri (Önceki Sekmeden)", "Yeni Dosya Yükle"],
        horizontal=True
    )
    
    df = None
    
    if data_source == "Temizlenmiş Veri (Önceki Sekmeden)":
        if st.session_state.cleaned_data is not None:
            df = st.session_state.cleaned_data
            st.success(f"✅ Temizlenmiş veri yüklendi: {len(df)} kayıt")
        else:
            st.warning("⚠️ Henüz temizlenmiş veri yok. Lütfen önce 'Veri Temizleme' sekmesini kullanın.")
    else:
        uploaded_file = st.file_uploader(
            "Excel dosyasını yükleyin",
            type=['xlsx', 'xls'],
            key="viz_uploader"
        )
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                st.success(f"✅ Dosya başarıyla yüklendi: {len(df)} satır, {len(df.columns)} sütun")
            except Exception as e:
                st.error(f"❌ Dosya yüklenirken hata: {str(e)}")
    
    if df is not None:
        # Tarih sütunu kontrolü ve dönüşümü
        if 'Hour' in df.columns:
            df['Hour'] = pd.to_datetime(df['Hour'])
        
        # Görselleştirme sekmeleri
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Zaman Serisi", "📊 Dağılım", "🔥 Isı Haritası", "📉 Karşılaştırma"])
        
        with tab1:
            st.subheader("Zaman Serisi Analizi")
            
            # Numerik sütunları al
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Hour' in df.columns:
                selected_columns = st.multiselect(
                    "Görselleştirmek istediğiniz sütunları seçin:",
                    numeric_columns,
                    default=numeric_columns[:2] if len(numeric_columns) >= 2 else numeric_columns
                )
                
                if selected_columns:
                    fig = go.Figure()
                    for col in selected_columns:
                        fig.add_trace(go.Scatter(
                            x=df['Hour'],
                            y=df[col],
                            mode='lines',
                            name=col,
                            hovertemplate='%{x}<br>%{y:.2f}<extra></extra>'
                        ))
                    
                    fig.update_layout(
                        title="Zaman Serisi Grafiği",
                        xaxis_title="Zaman",
                        yaxis_title="Değer",
                        hovermode='x unified',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Dağılım Analizi")
            
            col1, col2 = st.columns(2)
            with col1:
                selected_col = st.selectbox(
                    "Dağılımını görmek istediğiniz sütun:",
                    numeric_columns
                )
            
            with col2:
                chart_type = st.radio(
                    "Grafik tipi:",
                    ["Histogram", "Box Plot"],
                    horizontal=True
                )
            
            if selected_col:
                if chart_type == "Histogram":
                    fig = px.histogram(
                        df,
                        x=selected_col,
                        nbins=30,
                        title=f"{selected_col} Dağılımı"
                    )
                else:
                    fig = px.box(
                        df,
                        y=selected_col,
                        title=f"{selected_col} Box Plot"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # İstatistikler
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Ortalama", f"{df[selected_col].mean():.2f}")
                with col2:
                    st.metric("Medyan", f"{df[selected_col].median():.2f}")
                with col3:
                    st.metric("Min", f"{df[selected_col].min():.2f}")
                with col4:
                    st.metric("Max", f"{df[selected_col].max():.2f}")
        
        with tab3:
            st.subheader("Korelasyon Isı Haritası")
            
            # Korelasyon için sütun seçimi
            corr_columns = st.multiselect(
                "Korelasyon için sütunlar seçin:",
                numeric_columns,
                default=numeric_columns[:10] if len(numeric_columns) >= 10 else numeric_columns
            )
            
            if len(corr_columns) > 1:
                corr_matrix = df[corr_columns].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Korelasyon Matrisi",
                    color_continuous_scale="RdBu_r"
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Sütun Karşılaştırması")
            
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X ekseni:", numeric_columns, key="x_comp")
            with col2:
                y_axis = st.selectbox("Y ekseni:", numeric_columns, key="y_comp")
            
            if x_axis and y_axis:
                fig = px.scatter(
                    df,
                    x=x_axis,
                    y=y_axis,
                    title=f"{x_axis} vs {y_axis}",
                    trendline="ols"
                )
                st.plotly_chart(fig, use_container_width=True)

# DETAYLI ANALİZ SEKMESİ
with main_tab3:
    st.header("Detaylı İstatistiksel Analiz")
    
    if df is not None:
        # Özet istatistikler
        st.subheader("📊 Özet İstatistikler")
        st.dataframe(df.describe())
        
        # Eksik veri analizi
        st.subheader("❓ Eksik Veri Analizi")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            fig = px.bar(
                x=missing_data.index,
                y=missing_data.values,
                title="Eksik Veri Sayısı",
                labels={'x': 'Sütun', 'y': 'Eksik Veri Sayısı'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ Veri setinde eksik değer bulunmuyor!")
        
        # Veri tipi bilgileri
        with st.expander("📋 Veri Tipi Bilgileri"):
            dtype_df = pd.DataFrame({
                'Sütun': df.columns,
                'Veri Tipi': df.dtypes.astype(str),
                'Eksik Değer': df.isnull().sum(),
                'Benzersiz Değer': df.nunique()
            })
            st.dataframe(dtype_df)
