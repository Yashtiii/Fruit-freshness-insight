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

# Define CSV file
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

# Date range filter
if 'Date' in df.columns:
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['Date'].min(), df['Date'].max()),
        min_value=df['Date'].min().date(),
        max_value=df['Date'].max().date()
    )
    
    # Filter by date
    mask = (df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])
    df_filtered = df[mask].copy()
else:
    df_filtered = df.copy()

# Fruit type filter
fruit_filter = st.sidebar.multiselect(
    "Select Fruit Type",
    options=df['Fruit_Type'].unique(),
    default=df['Fruit_Type'].unique()
)
df_filtered = df_filtered[df_filtered['Fruit_Type'].isin(fruit_filter)]

# Ripeness filter
ripeness_filter = st.sidebar.multiselect(
    "Select Ripeness",
    options=df['Ripeness'].unique(),
    default=df['Ripeness'].unique()
)
df_filtered = df_filtered[df_filtered['Ripeness'].isin(ripeness_filter)]


# Key Metrics
st.header("📈 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Analyses", len(df_filtered))

with col2:
    avg_fruit_conf = df_filtered['Fruit_Confidence'].mean()
    st.metric("Avg Fruit Confidence", f"{avg_fruit_conf:.1f}%")

with col3:
    avg_ripeness_conf = df_filtered['Ripeness_Confidence'].mean()
    st.metric("Avg Ripeness Confidence", f"{avg_ripeness_conf:.1f}%")

with col4:
    most_common_fruit = df_filtered['Fruit_Type'].mode()[0] if len(df_filtered) > 0 else "N/A"
    st.metric("Most Common Fruit", most_common_fruit)

st.divider()


# Row 1: Fruit Distribution and Ripeness Distribution
st.header("🍎 Distribution Analysis")

col1, col2 = st.columns(2)

with col1:
    # Fruit Type Distribution - Pie Chart
    fruit_counts = df_filtered['Fruit_Type'].value_counts()
    fig_fruit = px.pie(
        values=fruit_counts.values,
        names=fruit_counts.index,
        title="Fruit Type Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_fruit.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_fruit, use_container_width=True)

with col2:
    # Ripeness Distribution - Pie Chart
    ripeness_counts = df_filtered['Ripeness'].value_counts()
    colors = {'Unripe': '#90EE90', 'Ripe': '#FFD700', 'Overripe': '#FF6347'}
    color_sequence = [colors.get(r, '#808080') for r in ripeness_counts.index]
    
    fig_ripeness = px.pie(
        values=ripeness_counts.values,
        names=ripeness_counts.index,
        title="Ripeness Distribution",
        color_discrete_sequence=color_sequence
    )
    fig_ripeness.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_ripeness, use_container_width=True)

st.divider()


# Row 2: Fruit-Ripeness Combination
st.header("🔀 Combined Analysis")

col1, col2 = st.columns(2)

with col1:
    # Stacked Bar Chart: Fruit Type by Ripeness
    fruit_ripeness = df_filtered.groupby(['Fruit_Type', 'Ripeness']).size().reset_index(name='Count')
    fig_stacked = px.bar(
        fruit_ripeness,
        x='Fruit_Type',
        y='Count',
        color='Ripeness',
        title="Fruit Type by Ripeness Level",
        barmode='stack',
        color_discrete_map={'Unripe': '#90EE90', 'Ripe': '#FFD700', 'Overripe': '#FF6347'}
    )
    st.plotly_chart(fig_stacked, use_container_width=True)

with col2:
    # Grouped Bar Chart
    fig_grouped = px.bar(
        fruit_ripeness,
        x='Fruit_Type',
        y='Count',
        color='Ripeness',
        title="Fruit Type by Ripeness (Grouped)",
        barmode='group',
        color_discrete_map={'Unripe': '#90EE90', 'Ripe': '#FFD700', 'Overripe': '#FF6347'}
    )
    st.plotly_chart(fig_grouped, use_container_width=True)

st.divider()


# Row 3: Confidence Scores Analysis
st.header("📊 Confidence Score Analysis")

col1, col2 = st.columns(2)

with col1:
    # Box plot for Fruit Confidence
    fig_box_fruit = px.box(
        df_filtered,
        x='Fruit_Type',
        y='Fruit_Confidence',
        title="Fruit Type Confidence Distribution",
        color='Fruit_Type',
        points="all"
    )
    fig_box_fruit.update_layout(showlegend=False)
    st.plotly_chart(fig_box_fruit, use_container_width=True)

with col2:
    # Box plot for Ripeness Confidence
    fig_box_ripeness = px.box(
        df_filtered,
        x='Ripeness',
        y='Ripeness_Confidence',
        title="Ripeness Confidence Distribution",
        color='Ripeness',
        points="all",
        color_discrete_map={'Unripe': '#90EE90', 'Ripe': '#FFD700', 'Overripe': '#FF6347'}
    )
    fig_box_ripeness.update_layout(showlegend=False)
    st.plotly_chart(fig_box_ripeness, use_container_width=True)

st.divider()


# Row 4: Time Series Analysis (if enough data)
if 'Date' in df_filtered.columns and len(df_filtered) > 1:
    st.header("📅 Time Series Analysis")
    
    # Daily counts
    daily_counts = df_filtered.groupby('Date').size().reset_index(name='Count')
    fig_time = px.line(
        daily_counts,
        x='Date',
        y='Count',
        title="Analyses Over Time",
        markers=True
    )
    fig_time.update_layout(xaxis_title="Date", yaxis_title="Number of Analyses")
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Fruit type over time
    daily_fruit = df_filtered.groupby(['Date', 'Fruit_Type']).size().reset_index(name='Count')
    fig_time_fruit = px.line(
        daily_fruit,
        x='Date',
        y='Count',
        color='Fruit_Type',
        title="Fruit Type Analyses Over Time",
        markers=True
    )
    st.plotly_chart(fig_time_fruit, use_container_width=True)

st.divider()


# Row 5: Data Source Analysis
st.header("📷 Source Analysis")

source_counts = df_filtered['Source'].value_counts()
fig_source = px.bar(
    x=source_counts.index,
    y=source_counts.values,
    title="Analysis by Source",
    labels={'x': 'Source', 'y': 'Count'},
    color=source_counts.index
)
st.plotly_chart(fig_source, use_container_width=True)

st.divider()


# Detailed Data Table
st.header("📋 Detailed Data")

# Show filtered data
st.dataframe(
    df_filtered.sort_values('ID', ascending=False),
    use_container_width=True,
    hide_index=True
)

# Download filtered data
csv_download = df_filtered.to_csv(index=False)
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv_download,
    file_name=f"filtered_fruit_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

# Refresh data button
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()
