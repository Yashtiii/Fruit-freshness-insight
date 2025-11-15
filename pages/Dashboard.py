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
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    return None

df = load_data()

if df is None or len(df) == 0:
    st.warning("⚠️ No data available yet. Please analyze some fruits first!")
    st.info("👈 Go to the main page to start analyzing fruits.")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Date filter (handle missing dates)
if 'Date' in df.columns:
    valid_dates = df['Date'].dropna()
    if len(valid_dates) > 0:
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(valid_dates.min(), valid_dates.max()),
            min_value=valid_dates.min().date(),
            max_value=valid_dates.max().date()
        )
        mask = (df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])
        df_filtered = df[mask].copy()
    else:
        df_filtered = df.copy()
else:
    df_filtered = df.copy()

# Fruit filter
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

# Temperature and Humidity Trends
if 'Temperature_C' in df_filtered.columns and 'Humidity_pct' in df_filtered.columns:
    st.header("🌡️ Environmental Conditions")
    
    col_t, col_h = st.columns(2)
    
    with col_t:
        temp_data = df_filtered[df_filtered['Temperature_C'].notna()]
        if len(temp_data) > 0:
            fig_temp = px.line(temp_data, x='Timestamp', y='Temperature_C', 
                              title="Temperature Over Time", markers=True)
            fig_temp.update_layout(yaxis_title="Temperature (°C)")
            st.plotly_chart(fig_temp, width="stretch")
    
    with col_h:
        hum_data = df_filtered[df_filtered['Humidity_pct'].notna()]
        if len(hum_data) > 0:
            fig_hum = px.line(hum_data, x='Timestamp', y='Humidity_pct', 
                             title="Humidity Over Time", markers=True)
            fig_hum.update_layout(yaxis_title="Humidity (%)")
            st.plotly_chart(fig_hum, width="stretch")
    
    st.divider()

# ========== ENHANCED TEMPERATURE & HUMIDITY VISUALIZATIONS ==========

# Check if we have temp/humidity data
df_iot = df_filtered[(df_filtered['Temperature_C'].notna()) & (df_filtered['Humidity_pct'].notna())]

if len(df_iot) > 0:
    st.header("🌡️💧 Detailed Environmental Analysis")
    
    # Row 1: Combined Temperature & Humidity Timeline
    st.subheader("📈 Temperature & Humidity Over Time")
    
    fig_combined = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Temperature (°C)", "Humidity (%)"),
        vertical_spacing=0.12,
        shared_xaxes=True
    )
    
    # Temperature trace
    fig_combined.add_trace(
        go.Scatter(
            x=df_iot['Timestamp'], 
            y=df_iot['Temperature_C'],
            mode='lines+markers',
            name='Temperature',
            line=dict(color='#FF6347', width=2),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Humidity trace
    fig_combined.add_trace(
        go.Scatter(
            x=df_iot['Timestamp'], 
            y=df_iot['Humidity_pct'],
            mode='lines+markers',
            name='Humidity',
            line=dict(color='#4682B4', width=2),
            marker=dict(size=8)
        ),
        row=2, col=1
    )
    
    fig_combined.update_xaxes(title_text="Time", row=2, col=1)
    fig_combined.update_yaxes(title_text="°C", row=1, col=1)
    fig_combined.update_yaxes(title_text="%", row=2, col=1)
    fig_combined.update_layout(height=600, showlegend=False)
    
    st.plotly_chart(fig_combined, width="stretch")
    
    st.divider()
    
    # Row 2: Temperature & Humidity Distribution
    st.subheader("📊 Temperature & Humidity Distribution")
    
    col_t, col_h = st.columns(2)
    
    with col_t:
        # Temperature histogram
        fig_temp_hist = px.histogram(
            df_iot, 
            x='Temperature_C',
            nbins=20,
            title="Temperature Distribution",
            color_discrete_sequence=['#FF6347']
        )
        fig_temp_hist.update_layout(
            xaxis_title="Temperature (°C)",
            yaxis_title="Frequency",
            showlegend=False
        )
        st.plotly_chart(fig_temp_hist, width="stretch")
    
    with col_h:
        # Humidity histogram
        fig_hum_hist = px.histogram(
            df_iot, 
            x='Humidity_pct',
            nbins=20,
            title="Humidity Distribution",
            color_discrete_sequence=['#4682B4']
        )
        fig_hum_hist.update_layout(
            xaxis_title="Humidity (%)",
            yaxis_title="Frequency",
            showlegend=False
        )
        st.plotly_chart(fig_hum_hist, width="stretch")
    
    st.divider()
    
    # Row 3: Temperature vs Humidity Correlation
    st.subheader("🔗 Temperature vs Humidity Correlation")
    
    col_scatter, col_stats = st.columns([2, 1])
    
    with col_scatter:
        fig_correlation = px.scatter(
            df_iot,
            x='Temperature_C',
            y='Humidity_pct',
            color='Ripeness',
            size='Fruit_Confidence',
            hover_data=['Fruit_Type', 'Timestamp'],
            title="Temperature vs Humidity by Ripeness",
            color_discrete_map={'Unripe': '#90EE90', 'Ripe': '#FFD700', 'Overripe': '#FF6347'}
        )
        fig_correlation.update_layout(
            xaxis_title="Temperature (°C)",
            yaxis_title="Humidity (%)"
        )
        st.plotly_chart(fig_correlation, width="stretch")
    
    with col_stats:
        st.write("### Statistics")
        
        # Temperature stats
        st.metric("Min Temp", f"{df_iot['Temperature_C'].min():.1f}°C")
        st.metric("Max Temp", f"{df_iot['Temperature_C'].max():.1f}°C")
        st.metric("Avg Temp", f"{df_iot['Temperature_C'].mean():.1f}°C")
        
        st.write("")
        
        # Humidity stats
        st.metric("Min Humidity", f"{df_iot['Humidity_pct'].min():.1f}%")
        st.metric("Max Humidity", f"{df_iot['Humidity_pct'].max():.1f}%")
        st.metric("Avg Humidity", f"{df_iot['Humidity_pct'].mean():.1f}%")
        
        # Correlation
        correlation = df_iot['Temperature_C'].corr(df_iot['Humidity_pct'])
        st.metric("Correlation", f"{correlation:.3f}")
    
    st.divider()
    
    # Row 4: Environmental Conditions by Fruit/Ripeness
    st.subheader("🍎 Environmental Conditions by Category")
    
    col_fruit, col_ripeness = st.columns(2)
    
    with col_fruit:
        # Box plot: Temp by Fruit Type
        fig_temp_fruit = px.box(
            df_iot,
            x='Fruit_Type',
            y='Temperature_C',
            color='Fruit_Type',
            title="Temperature by Fruit Type",
            points="all"
        )
        fig_temp_fruit.update_layout(showlegend=False, yaxis_title="Temperature (°C)")
        st.plotly_chart(fig_temp_fruit, width="stretch")
        
        # Box plot: Humidity by Fruit Type
        fig_hum_fruit = px.box(
            df_iot,
            x='Fruit_Type',
            y='Humidity_pct',
            color='Fruit_Type',
            title="Humidity by Fruit Type",
            points="all"
        )
        fig_hum_fruit.update_layout(showlegend=False, yaxis_title="Humidity (%)")
        st.plotly_chart(fig_hum_fruit, width="stretch")
    
    with col_ripeness:
        # Box plot: Temp by Ripeness
        fig_temp_ripeness = px.box(
            df_iot,
            x='Ripeness',
            y='Temperature_C',
            color='Ripeness',
            title="Temperature by Ripeness",
            color_discrete_map={'Unripe': '#90EE90', 'Ripe': '#FFD700', 'Overripe': '#FF6347'},
            points="all"
        )
        fig_temp_ripeness.update_layout(showlegend=False, yaxis_title="Temperature (°C)")
        st.plotly_chart(fig_temp_ripeness, width="stretch")
        
        # Box plot: Humidity by Ripeness
        fig_hum_ripeness = px.box(
            df_iot,
            x='Ripeness',
            y='Humidity_pct',
            color='Ripeness',
            title="Humidity by Ripeness",
            color_discrete_map={'Unripe': '#90EE90', 'Ripe': '#FFD700', 'Overripe': '#FF6347'},
            points="all"
        )
        fig_hum_ripeness.update_layout(showlegend=False, yaxis_title="Humidity (%)")
        st.plotly_chart(fig_hum_ripeness, width="stretch")
    
    st.divider()
    
    # Row 5: Heatmap - Average conditions by Fruit & Ripeness
    st.subheader("🔥 Average Environmental Conditions Heatmap")
    
    col_heat_t, col_heat_h = st.columns(2)
    
    with col_heat_t:
        # Temperature heatmap
        temp_pivot = df_iot.pivot_table(
            values='Temperature_C',
            index='Ripeness',
            columns='Fruit_Type',
            aggfunc='mean'
        )
        
        fig_heat_temp = px.imshow(
            temp_pivot,
            labels=dict(x="Fruit Type", y="Ripeness", color="Temp (°C)"),
            title="Average Temperature by Category",
            color_continuous_scale='Reds',
            text_auto='.1f'
        )
        st.plotly_chart(fig_heat_temp, width="stretch")
    
    with col_heat_h:
        # Humidity heatmap
        hum_pivot = df_iot.pivot_table(
            values='Humidity_pct',
            index='Ripeness',
            columns='Fruit_Type',
            aggfunc='mean'
        )
        
        fig_heat_hum = px.imshow(
            hum_pivot,
            labels=dict(x="Fruit Type", y="Ripeness", color="Humidity (%)"),
            title="Average Humidity by Category",
            color_continuous_scale='Blues',
            text_auto='.1f'
        )
        st.plotly_chart(fig_heat_hum, width="stretch")
    
    st.divider()
    
    # Row 6: Environmental Quality Gauge
    st.subheader("🎯 Environmental Quality Assessment")
    
    col_g1, col_g2, col_g3 = st.columns(3)
    
    avg_temp = df_iot['Temperature_C'].mean()
    avg_hum = df_iot['Humidity_pct'].mean()
    
    with col_g1:
        # Temperature gauge
        fig_gauge_temp = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_temp,
            title={'text': "Average Temperature"},
            gauge={
                'axis': {'range': [0, 40]},
                'bar': {'color': "#FF6347"},
                'steps': [
                    {'range': [0, 10], 'color': "#ADD8E6"},
                    {'range': [10, 20], 'color': "#90EE90"},
                    {'range': [20, 30], 'color': "#FFD700"},
                    {'range': [30, 40], 'color': "#FF6347"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 30
                }
            }
        ))
        fig_gauge_temp.update_layout(height=300)
        st.plotly_chart(fig_gauge_temp, width="stretch")
    
    with col_g2:
        # Humidity gauge
        fig_gauge_hum = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_hum,
            title={'text': "Average Humidity"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4682B4"},
                'steps': [
                    {'range': [0, 30], 'color': "#FFE4B5"},
                    {'range': [30, 50], 'color': "#ADD8E6"},
                    {'range': [50, 70], 'color': "#90EE90"},
                    {'range': [70, 100], 'color': "#4682B4"}
                ],
                'threshold': {
                    'line': {'color': "blue", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig_gauge_hum.update_layout(height=300)
        st.plotly_chart(fig_gauge_hum, width="stretch")
    
    with col_g3:
        st.write("### Quality Guidelines")
        st.write("**Ideal Temperature:**")
        st.write("🟢 20-25°C: Optimal")
        st.write("🟡 15-20°C or 25-30°C: Good")
        st.write("🔴 <15°C or >30°C: Suboptimal")
        
        st.write("")
        st.write("**Ideal Humidity:**")
        st.write("🟢 50-70%: Optimal")
        st.write("🟡 40-50% or 70-80%: Good")
        st.write("🔴 <40% or >80%: Suboptimal")
        
        # Current status
        st.write("")
        if 20 <= avg_temp <= 25 and 50 <= avg_hum <= 70:
            st.success("✅ Current conditions are optimal!")
        elif 15 <= avg_temp <= 30 and 40 <= avg_hum <= 80:
            st.warning("⚠️ Current conditions are acceptable.")
        else:
            st.error("❌ Current conditions need adjustment.")
    
    st.divider()

else:
    st.info("💡 No temperature/humidity data available yet. Use the IoT sensor feature to collect environmental data!")

# ========== END OF TEMPERATURE & HUMIDITY VISUALIZATIONS ==========


# Automated Insights
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

st.divider()

# ========== CHARTS START HERE (YOUR ORIGINAL CODE) ==========

# Confidence vs Ripeness Scatter
st.header("🎯 Confidence vs Ripeness")

fig_scatter = px.scatter(
    df_filtered,
    x='Fruit_Confidence',
    y='Ripeness_Confidence',
    color='Ripeness',
    size='Fruit_Confidence',
    hover_data=['Fruit_Type', 'Source'],
    title="Fruit vs Ripeness Confidence Relationship",
    color_discrete_map={'Unripe': '#90EE90', 'Ripe': '#FFD700', 'Overripe': '#FF6347'}
)
st.plotly_chart(fig_scatter, width="stretch")

st.divider()

# Source Distribution
st.header("📷 Source Distribution")

source_counts = df_filtered['Source'].value_counts()

fig_source = px.pie(
    values=source_counts.values,
    names=source_counts.index,
    hole=0.5,
    title="Data Source Breakdown"
)
fig_source.update_traces(textinfo='percent+label')
st.plotly_chart(fig_source, width="stretch")

st.divider()

# Fruit and Ripeness Distribution
st.header("🍎 Distribution Analysis")

col1, col2 = st.columns(2)

with col1:
    # Fruit Type Distribution
    fruit_counts = df_filtered['Fruit_Type'].value_counts()
    fig_fruit = px.pie(
        values=fruit_counts.values,
        names=fruit_counts.index,
        title="Fruit Type Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_fruit.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_fruit, width="stretch")

with col2:
    # Ripeness Distribution
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
    st.plotly_chart(fig_ripeness, width="stretch")

st.divider()

# Combined Analysis
st.header("🔀 Combined Analysis")

col1, col2 = st.columns(2)

with col1:
    # Stacked Bar Chart
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
    st.plotly_chart(fig_stacked, width="stretch")

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
    st.plotly_chart(fig_grouped, width="stretch")

st.divider()

# Confidence Scores
st.header("📊 Confidence Score Analysis")

col1, col2 = st.columns(2)

with col1:
    fig_box_fruit = px.box(
        df_filtered,
        x='Fruit_Type',
        y='Fruit_Confidence',
        title="Fruit Type Confidence Distribution",
        color='Fruit_Type',
        points="all"
    )
    fig_box_fruit.update_layout(showlegend=False)
    st.plotly_chart(fig_box_fruit, width="stretch")

with col2:
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
    st.plotly_chart(fig_box_ripeness, width="stretch")

st.divider()

# Detailed Data Table
st.header("📋 Detailed Data")

table = df_filtered.sort_values('ID', ascending=False).copy()

def highlight_low_conf(row):
    color = 'background-color: #ffb3b3'
    default = [''] * len(row)
    
    if 'Fruit_Confidence' in row and pd.notna(row['Fruit_Confidence']) and row['Fruit_Confidence'] < 50:
        return [color] * len(row)
    
    return default

st.write(table.style.apply(highlight_low_conf, axis=1))

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
