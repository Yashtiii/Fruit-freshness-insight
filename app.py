# app.py - FIXED VERSION
import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.express as px

# ---------- CONFIG ----------
st.set_page_config(page_title="Fruit Freshness Insights", layout="wide", initial_sidebar_state="expanded")

DATA_CSV = "fruit_analysis_results.csv"
IMAGES_DIR = "."

# ---------- UTILS ----------
def load_data(csv_path=DATA_CSV):
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception:
            pass
    else:
        df["timestamp"] = pd.to_datetime(df.index.map(lambda i: datetime.now()))
    return df

def latest_image_path(images_dir=IMAGES_DIR):
    images = [os.path.join(images_dir, f) for f in os.listdir(images_dir)
              if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    if not images:
        return None
    images.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return images[0]

def shelf_life_label(days_est):
    try:
        d = float(days_est)
        if d <= 2:
            return "Very short"
        if d <= 4:
            return "Short"
        if d <= 7:
            return "Moderate"
        return "Long"
    except Exception:
        return str(days_est)

def nice_metric(value, unit=""):
    return f"{value} {unit}".strip()

# ✅ NEW: Safe float conversion helper
def safe_float(value, default="—"):
    """Safely convert to float, return default if fails"""
    if value is None or value == "" or value == "—":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

# ---------- PAGE LAYOUT ----------
st.markdown(
    """
    <style>
    .stApp { background-color: #0f1724; color: #e6eef8; }
    .card { padding: 18px; border-radius: 12px; background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); box-shadow: 0 6px 18px rgba(2,6,23,0.6); }
    .kpi { font-size: 22px; color: #9fb7d9; }
    .kpi-value { font-size: 34px; font-weight: 700; color: #e8f4ff; }
    .status-good { color: #16a34a; font-weight:700; }
    .status-warn { color: #f59e0b; font-weight:700; }
    .status-bad { color: #ef4444; font-weight:700; }
    </style>
    """, unsafe_allow_html=True
)

st.title("🍏 Fruit Freshness — Insight Dashboard")
st.write("Visual and actionable view of detected fruit, environment metrics and shelf-life trends.")

# ---------- Load Data ----------
df = load_data()
last_row = df.iloc[-1] if not df.empty else None

# Left column: KPIs and controls
left, middle, right = st.columns([2.2, 3, 2.2])

with left:
    st.markdown("### Latest Snapshot")
    if last_row is None:
        st.info("No entries found in the CSV. Run detection or add rows to `fruit_analysis_results.csv`.")
    else:
        fruit_type = last_row.get("fruit_type", last_row.get("Fruit Type", "Unknown"))
        ripeness = last_row.get("ripeness", last_row.get("Ripeness", "Unknown"))
        temp = last_row.get("temperature", last_row.get("Temperature (C)", last_row.get("Temperature", "—")))
        humidity = last_row.get("humidity", last_row.get("Humidity (%)", last_row.get("Humidity", "—")))
        est_shelf = last_row.get("estimated_shelf_life_days", last_row.get("Estimated Shelf Life (days)", last_row.get("shelf_life_days", "—")))
        timestamp = last_row.get("timestamp", "")

        # KPI cards
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"<div class='kpi'>Fruit</div><div class='kpi-value'>{fruit_type}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"<div class='kpi'>Ripeness</div><div class='kpi-value'>{ripeness}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        # ✅ FIXED: Safe conversion for temperature and humidity
        c1, c2 = st.columns(2)
        with c1:
            temp_val = safe_float(temp)
            if temp_val != "—":
                st.metric("Temperature (°C)", nice_metric(round(temp_val, 1)))
            else:
                st.metric("Temperature (°C)", "—")
        
        with c2:
            hum_val = safe_float(humidity)
            if hum_val != "—":
                st.metric("Humidity (%)", nice_metric(round(hum_val, 1)))
            else:
                st.metric("Humidity (%)", "—")

        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        label = shelf_life_label(est_shelf)
        st.markdown(f"<div class='kpi'>Estimated Shelf Life</div><div class='kpi-value'>{est_shelf} days — {label}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if timestamp is not None:
            try:
                ttext = pd.to_datetime(timestamp)
                st.caption(f"Last measured: {ttext}")
            except Exception:
                st.caption(f"Last measured: {timestamp}")

        st.write("")
        if st.button("🔄 Refresh data"):
            st.rerun()  # Changed from st.experimental_rerun()

with middle:
    st.markdown("### Timeline & Trends")
    if df.empty:
        st.info("No historical data to show.")
    else:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            except Exception:
                pass

        # Temperature trend
        if "temperature" in df.columns:
            fig_t = px.line(df, x="timestamp", y="temperature", title="Temperature over time", markers=True)
            fig_t.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_t, use_container_width=True)

        # Humidity trend
        if "humidity" in df.columns:
            fig_h = px.line(df, x="timestamp", y="humidity", title="Humidity over time", markers=True)
            fig_h.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_h, use_container_width=True)

        # Ripeness distribution
        if "ripeness" in df.columns:
            rip_counts = df["ripeness"].value_counts().reset_index()
            rip_counts.columns = ["ripeness", "count"]
            fig_r = px.pie(rip_counts, names="ripeness", values="count", title="Ripeness distribution", hole=0.4)
            st.plotly_chart(fig_r, use_container_width=True)

with right:
    st.markdown("### Last image & actions")
    image_path = latest_image_path(IMAGES_DIR)
    if image_path:
        st.image(image_path, caption=os.path.basename(image_path), use_column_width=True)
    else:
        st.info("No result images found in project folder.")

    st.write("")
    st.markdown("### Quick Insights")
    if last_row is None:
        st.write("- No recent data.")
    else:
        insights = []
        # ✅ FIXED: Safe conversion in insights
        temp_val = safe_float(temp, None)
        hum_val = safe_float(humidity, None)
        shelf_val = safe_float(est_shelf, None)
        
        if temp_val is not None:
            if temp_val >= 28:
                insights.append("Temperature is high — consider cooling to extend shelf life.")
            elif temp_val <= 8:
                insights.append("Temperature is low — watch for chill damage on sensitive fruits.")
        
        if hum_val is not None and hum_val >= 80:
            insights.append("High humidity — risk of mold; ensure good ventilation.")
        
        if shelf_val is not None and shelf_val <= 2:
            insights.append("Shelf life very short — recommend immediate consumption or fast sale.")
        
        if ripeness and str(ripeness).lower() in ("unripe", "raw"):
            insights.append("Fruit is unripe — store in a cool, humid place for slower ripening, or room-temp to accelerate.")

        if insights:
            for i in insights:
                st.write("•", i)
        else:
            st.write("• No immediate alerts. Conditions look reasonable.")

    st.write("")
    st.markdown("### Actions")
    if st.button("📥 Export filtered CSV (ripeness = Unripe)"):
        if not df.empty and "ripeness" in df.columns:
            fname = "unripe_filtered.csv"
            df[df["ripeness"].str.lower() == "unripe"].to_csv(fname, index=False)
            st.success(f"Exported `{fname}`")
        else:
            st.warning("No matching data to export.")

# ---------- Footer: small analytics ----------
st.markdown("---")
st.markdown("#### Historical stats")
if df.empty:
    st.write("No data")
else:
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total observations", len(df))
    with cols[1]:
        if "fruit_type" in df.columns:
            st.metric("Distinct fruits", df["fruit_type"].nunique())
        else:
            st.metric("Distinct fruits", "—")
    with cols[2]:
        if "ripeness" in df.columns:
            st.metric("Most common ripeness", df["ripeness"].mode().iat[0] if len(df["ripeness"].mode()) > 0 else "—")
        else:
            st.metric("Most common ripeness", "—")
    with cols[3]:
        if "estimated_shelf_life_days" in df.columns:
            # ✅ FIXED: Safe conversion for avg shelf life
            try:
                avg_shelf = df["estimated_shelf_life_days"].apply(lambda x: safe_float(x, None))
                avg_shelf = avg_shelf[avg_shelf.notna()].mean()
                st.metric("Avg shelf life (days)", round(avg_shelf, 1) if pd.notna(avg_shelf) else "—")
            except:
                st.metric("Avg shelf life (days)", "—")
        else:
            st.metric("Avg shelf life (days)", "—")

st.caption("Tip: Add more sensor readings (temperature & humidity) to improve insights. Modify thresholds in app.py to match your product needs.")
