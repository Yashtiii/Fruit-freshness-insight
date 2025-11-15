# pages/1_📊_Dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

st.set_page_config(
    page_title="Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Fruit Analysis Dashboard")

CSV_FILE = 'fruit_analysis_results.csv'

# Load data
@st.cache_data
def load_data():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

df = load_data()

if df is None or len(df) == 0:
    st.warning("⚠️ No data available yet. Please analyze some fruits first!")
    st.info("👈 Go to the main page to start analyzing fruits.")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filters")

if 'Date' in df.columns:
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['Date'].min(), df['Date'].max()),
        min_value=df['Date'].min().date(),
        max_value=df['Date'].max().date()
    )
    mask = (df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])
    df_filtered = df[mask].copy()
else:
    df_filtered = df.copy()

fruit_filter = st.sidebar.multiselect(
    "Select Fruit Type",
    options=df['Fruit_Type'].unique(),
    default=df['Fruit_Type'].unique()
)
df_filtered = df_filtered[df_filtered['Fruit_Type'].isin(fruit_filter)]

ripeness_filter = st.sidebar.multiselect(
    "Select Ripeness",
    options=df['Ripeness'].unique(),
    default=df['Ripeness'].unique()
)
df_filtered = df_filtered[df_filtered['Ripeness'].isin(ripeness_filter)]

# ✅ NEW: Temperature/Humidity stats
st.header("📈 Key Metrics")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Total Analyses", len(df_filtered))

with col2:
    avg_fruit_conf = df_filtered['Fruit_Confidence'].mean()
    st.metric("Avg Fruit Conf", f"{avg_fruit_conf:.1f}%")

with col3:
    avg_ripeness_conf = df_filtered['Ripeness_Confidence'].mean()
    st.metric("Avg Ripeness Conf", f"{avg_ripeness_conf:.1f}%")

with col4:
    most_common_fruit = df_filtered['Fruit_Type'].mode()[0] if len(df_filtered) > 0 else "N/A"
    st.metric("Most Common Fruit", most_common_fruit)

# ✅ NEW: Show temperature and humidity averages
with col5:
    if 'Temperature_C' in df_filtered.columns:
        avg_temp = df_filtered['Temperature_C'].dropna().mean()
        st.metric("Avg Temp (°C)", f"{avg_temp:.1f}" if pd.notna(avg_temp) else "N/A")
    else:
        st.metric("Avg Temp (°C)", "N/A")

with col6:
    if 'Humidity_pct' in df_filtered.columns:
        avg_hum = df_filtered['Humidity_pct'].dropna().mean()
        st.metric("Avg Humidity (%)", f"{avg_hum:.1f}" if pd.notna(avg_hum) else "N/A")
    else:
        st.metric("Avg Humidity (%)", "N/A")

st.divider()

# ✅ NEW: Temperature and Humidity Trend Charts
if 'Temperature_C' in df_filtered.columns and 'Humidity_pct' in df_filtered.columns:
    st.header("🌡️ Environmental Conditions")
    
    col_t, col_h = st.columns(2)
    
    with col_t:
        # Temperature trend
        temp_data = df_filtered[df_filtered['Temperature_C'].notna()]
        if len(temp_data) > 0:
            fig_temp = px.line(temp_data, x='Timestamp', y='Temperature_C', 
                              title="Temperature Over Time", markers=True)
            fig_temp.update_layout(yaxis_title="Temperature (°C)")
            st.plotly_chart(fig_temp, use_container_width=True)
    
    with col_h:
        # Humidity trend
        hum_data = df_filtered[df_filtered['Humidity_pct'].notna()]
        if len(hum_data) > 0:
            fig_hum = px.line(hum_data, x='Timestamp', y='Humidity_pct', 
                             title="Humidity Over Time", markers=True)
            fig_hum.update_layout(yaxis_title="Humidity (%)")
            st.plotly_chart(fig_hum, use_container_width=True)
    
    st.divider()

# Rest of your existing dashboard code...
st.header("🧠 Automated Insights")

insights = []

if len(df_filtered) > 0:
    freq_fruit = df_filtered['Fruit_Type'].mode()[0]
    insights.append(f"Most frequently analyzed fruit: **{freq_fruit}**")

if len(df_filtered['Ripeness'].unique()) > 1:
    ripe_counts = df_filtered['Ripeness'].value_counts()
    top_ripeness = ripe_counts.index[0]
    insights.append(f"Most common ripeness detected: **{top_ripeness}**")

low_conf = df_filtered[df_filtered['Fruit_Confidence'] < 50]
if len(low_conf) > 0:
    insights.append("⚠️ Several analyses have **low fruit confidence (<50%)**. Check lighting or camera.")

source_counts = df_filtered['Source'].value_counts()
if len(source_counts) > 0:
    top_source = source_counts.index[0]
    insights.append(f"Most used source: **{top_source}**")

for tip in insights:
    st.info(tip)

# [Rest of your existing dashboard code remains the same...]
# (Keep all your existing charts: fruit distribution, ripeness, confidence scores, etc.)

# Download button
st.header("📋 Export Data")
csv_download = df_filtered.to_csv(index=False)
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv_download,
    file_name=f"filtered_fruit_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()
