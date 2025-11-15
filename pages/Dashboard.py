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



# === INSIGHTS SECTION ===
#yukti added
st.header("🧠 Automated Insights")

insights = []

# Most frequent fruit
if len(df_filtered) > 0:
    freq_fruit = df_filtered['Fruit_Type'].mode()[0]
    insights.append(f"Most frequently analyzed fruit: **{freq_fruit}**")

# Ripeness trends
if len(df_filtered['Ripeness'].unique()) > 1:
    ripe_counts = df_filtered['Ripeness'].value_counts()
    top_ripeness = ripe_counts.index[0]
    insights.append(f"Most common ripeness detected: **{top_ripeness}**")

# Confidence alerts
low_conf = df_filtered[df_filtered['Fruit_Confidence'] < 50]
if len(low_conf) > 0:
    insights.append("⚠️ Several analyzes have **low fruit confidence (<50%)**. Check lighting or camera.")

# Source insights
source_counts = df_filtered['Source'].value_counts()
if len(source_counts) > 0:
    top_source = source_counts.index[0]
    insights.append(f"Most used source: **{top_source}**")

# Display insights
for tip in insights:
    st.info(tip)



#yukti added
#confidence vs ripeness
st.header("🎯 Confidence vs Ripeness")

fig_scatter = px.scatter(
    df_filtered,
    x='Fruit_Confidence',
    y='Ripeness_Confidence',
    color='Ripeness',
    size='Fruit_Confidence',
    hover_data=['Fruit_Type', 'Source', 'Date'],
    title="Fruit vs Ripeness Confidence Relationship",
    color_discrete_map={'Unripe': '#90EE90', 'Ripe': '#FFD700', 'Overripe': '#FF6347'}
)
st.plotly_chart(fig_scatter, use_container_width=True)


#yukti added
#source donut chart
st.header("📷 Source Distribution (Camera vs Upload)")

source_counts = df_filtered['Source'].value_counts()

fig_source2 = px.pie(
    values=source_counts.values,
    names=source_counts.index,
    hole=0.5,
    title="Data Source Breakdown"
)
fig_source2.update_traces(textinfo='percent+label')
st.plotly_chart(fig_source2, use_container_width=True)




#yukti added
# Rolling averages and WoW change
if 'Date' in df_filtered.columns:
    tmp = df_filtered.copy()
    tmp = tmp.set_index('Date').sort_index()
    # if you have daily aggregation fields
    daily = tmp.resample('D').agg({
        'Fruit_Confidence':'mean',
        'Ripeness_Confidence':'mean'
    }).fillna(method='ffill')

    daily['fc_7d'] = daily['Fruit_Confidence'].rolling(7, min_periods=1).mean()
    daily['rc_7d'] = daily['Ripeness_Confidence'].rolling(7, min_periods=1).mean()

    st.subheader("7-day rolling average (confidence)")
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=daily.index, y=daily['fc_7d'], name='Fruit Conf 7d'))
    fig_roll.add_trace(go.Scatter(x=daily.index, y=daily['rc_7d'], name='Ripeness Conf 7d'))
    fig_roll.update_layout(yaxis_title="Confidence %")
    st.plotly_chart(fig_roll, use_container_width=True)

    # week over week percent change (last available)
    if len(daily) >= 14:
        last_week = daily['Fruit_Confidence'].iloc[-7:].mean()
        prev_week = daily['Fruit_Confidence'].iloc[-14:-7].mean()
        wow = (last_week - prev_week) / prev_week * 100 if prev_week else 0
        st.metric("Fruit Confidence WoW", f"{wow:.1f}%", delta=f"{(last_week - prev_week):.2f}")








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



# Detailed Data Table
st.header("📋 Detailed Data")

# Prepare table
table = df_filtered.sort_values('ID', ascending=False).copy()

# Define highlighter function
def highlight_low_conf(row):
    color = 'background-color: #ffb3b3'  # Light red
    default = [''] * len(row)

    # If fruit confidence exists and is < 50 → highlight whole row
    if 'Fruit_Confidence' in row and pd.notna(row['Fruit_Confidence']) and row['Fruit_Confidence'] < 50:
        return [color] * len(row)

    return default

# Show styled table
st.write(table.style.apply(highlight_low_conf, axis=1))

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
