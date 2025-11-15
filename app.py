# app.py - CLEANED VERSION (2 tabs only)
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
from datetime import datetime
import os
import time
import re

try:
    import serial
    SERIAL_AVAILABLE = True
except Exception:
    SERIAL_AVAILABLE = False

# -------------------- PAGE CONFIG -------------------- #
st.set_page_config(
    page_title="🍎 Fruit Ripeness & Type Detector",
    page_icon="🍎",
    layout="wide"
)

st.title("🍎 Fruit Ripeness & Type Detector")

CSV_FILE = "fruit_analysis_results.csv"

# -------------------- SESSION STATE -------------------- #
if "results_history" not in st.session_state:
    if os.path.exists(CSV_FILE):
        st.session_state.results_history = pd.read_csv(CSV_FILE).to_dict("records")
    else:
        st.session_state.results_history = []

if "result_counter" not in st.session_state:
    st.session_state.result_counter = len(st.session_state.results_history)

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

# -------------------- MODEL LOADING -------------------- #
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "fruit_ripeness_with_person_rejection.keras")
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found: {model_path}")
        return None
    return tf.keras.models.load_model(model_path)

model = load_model()
if model is None:
    st.stop()

RIPENESS_CLASSES = ["Unripe", "Ripe", "Overripe", "Not Fruit"]
FRUIT_TYPES = ["Apple", "Orange"]

# -------------------- IMAGE UTILITIES -------------------- #
def preprocess_image(image_pil):
    try:
        img = np.array(image_pil)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = cv2.resize(img, (224, 224))
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img.astype("float32"))
        return np.expand_dims(img, axis=0)
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return None

def analyze_image(image_pil):
    try:
        proc = preprocess_image(image_pil)
        if proc is None:
            return None
        preds = model.predict(proc, verbose=0)
        ripeness_probs = preds[0][0]
        fruit_probs = preds[1][0]

        ridx = int(np.argmax(ripeness_probs))
        fidx = int(np.argmax(fruit_probs))

        ripeness = RIPENESS_CLASSES[ridx]
        ripeness_conf = float(ripeness_probs[ridx])

        if ridx == 3:  # Not Fruit
            return {
                "is_fruit": False,
                "ripeness": ripeness,
                "ripeness_conf": ripeness_conf,
                "fruit": "N/A",
                "fruit_conf": 0.0,
                "ripeness_probs": ripeness_probs,
                "fruit_probs": fruit_probs
            }
        else:
            return {
                "is_fruit": True,
                "ripeness": ripeness,
                "ripeness_conf": ripeness_conf,
                "fruit": FRUIT_TYPES[fidx],
                "fruit_conf": float(fruit_probs[fidx]),
                "ripeness_probs": ripeness_probs,
                "fruit_probs": fruit_probs
            }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# -------------------- SAVE/CSV -------------------- #
def save_result_to_csv(result, source, temp=None, hum=None):
    """Save result with temperature and humidity"""
    st.session_state.result_counter += 1
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # ✅ FIXED: Properly save temp and humidity
    entry = {
        "ID": st.session_state.result_counter,
        "Timestamp": ts,
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Source": source,
        "Is_Fruit": result.get("is_fruit", True),
        "Fruit_Type": result.get("fruit", "N/A"),
        "Fruit_Confidence": round(result.get("fruit_conf", 0) * 100, 2),
        "Ripeness": result.get("ripeness", "N/A"),
        "Ripeness_Confidence": round(result.get("ripeness_conf", 0) * 100, 2),
        "Temperature_C": temp if temp is not None else None,  # ✅ Store temp
        "Humidity_pct": hum if hum is not None else None      # ✅ Store humidity
    }
    
    st.session_state.results_history.append(entry)
    pd.DataFrame(st.session_state.results_history).to_csv(CSV_FILE, index=False)

# -------------------- SENSOR FUNCTIONS -------------------- #
def read_clean_sensor(esp, max_lines=40):
    for _ in range(max_lines):
        raw = esp.readline().decode("ascii", errors="ignore").strip()
        if not raw:
            continue
        low = raw.lower()
        if low.startswith("ets") or low.startswith("rst:") or "boot" in low:
            continue
        if "humidity" in low and "temperature" in low:
            return raw
    return None

def parse_temp_hum(line):
    try:
        clean = line.replace("°", " ").replace("%", " ").replace("C", " ").replace("c", " ")
        clean = clean.replace(",", " ")
        hum_m = re.search(r"humidity[:=\s]+([+-]?\d*\.?\d+)", clean, re.IGNORECASE)
        temp_m = re.search(r"temperature[:=\s]+([+-]?\d*\.?\d+)", clean, re.IGNORECASE)

        hum = float(hum_m.group(1)) if hum_m else None
        temp = float(temp_m.group(1)) if temp_m else None

        if hum is not None and (hum < 0 or hum > 100):
            hum = None
        if temp is not None and (temp < -20 or temp > 60):
            temp = None

        return temp, hum
    except Exception:
        return None, None

# -------------------- SHELF-LIFE CALCULATION -------------------- #
def combined_shelf_life(ripeness, temp, hum):
    base = {
        "Unripe": (5, 7),
        "Ripe": (2, 3),
        "Overripe": (0, 0)
    }
    if ripeness not in base:
        return "N/A"

    low, high = base[ripeness]
    if ripeness == "Overripe":
        return "0 days"

    if temp is None or hum is None:
        return f"{low}-{high} days"

    if temp > 30:
        factor = 0.6
    elif 20 <= temp <= 30 and 50 <= hum <= 80:
        factor = 1.0
    elif temp < 20 and 55 <= hum <= 65:
        factor = 1.2
    else:
        factor = 0.8

    adj_low = max(0, round(low * factor))
    adj_high = max(0, round(high * factor))
    if adj_low > adj_high:
        adj_low, adj_high = adj_high, adj_low
    return f"{adj_low}-{adj_high} days"

# -------------------- UI / TABS -------------------- #
tab1, tab2 = st.tabs(["📸 Analyze Fruit", "📡 Shelf Life (IoT)"])

# -------------------- TAB 1: Analyze -------------------- #
with tab1:
    st.header("📸 Analyze Fruit (Camera + Upload)")
    left_col, right_col = st.columns([1, 1])

    with left_col:
        camera_img = st.camera_input("Capture from camera")
    with right_col:
        upload_img = st.file_uploader("Or upload image", type=["jpg", "jpeg", "png"])

    image = None
    if upload_img:
        try:
            image = Image.open(upload_img)
            st.image(image, use_container_width="stretch")
        except Exception as e:
            st.error(f"Cannot open uploaded image: {e}")
    elif camera_img:
        try:
            image = Image.open(camera_img)
            st.image(image, use_container_width="stretch")
        except Exception as e:
            st.error(f"Cannot open camera image: {e}")

    if image:
        if st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing image..."):
                result = analyze_image(image)

            if result is None:
                st.error("Analysis failed.")
            else:
                st.session_state.last_analysis = result
                
                # ✅ Save WITHOUT temp/humidity first
                save_result_to_csv(result, "Camera/Upload", temp=None, hum=None)

                if not result["is_fruit"]:
                    st.warning("⚠️ Not a fruit detected. Please provide an apple or orange image.")
                    st.metric("Detection", result["ripeness"], f"{result['ripeness_conf']:.1%}")
                else:
                    st.success("✅ Fruit detected and saved.")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("🍎 Ripeness", result["ripeness"], f"{result['ripeness_conf']:.1%}")
                    with col_b:
                        st.metric("🍊 Fruit Type", result["fruit"], f"{result['fruit_conf']:.1%}")

                    with st.expander("📊 Detailed Scores"):
                        st.write("**Ripeness Breakdown**")
                        for i, cls in enumerate(RIPENESS_CLASSES):
                            st.progress(float(result["ripeness_probs"][i]), 
                                       text=f"{cls}: {result['ripeness_probs'][i]:.2%}")

                        st.write("**Fruit Type Breakdown**")
                        for i, cls in enumerate(FRUIT_TYPES):
                            st.progress(float(result["fruit_probs"][i]), 
                                       text=f"{cls}: {result['fruit_probs'][i]:.2%}")

                    st.info(f"💾 Result #{st.session_state.result_counter} saved. Go to IoT tab to add temperature/humidity.")

# -------------------- TAB 2: IoT -------------------- #
with tab2:
    st.header("📡 Shelf Life Estimation (IoT)")

    if st.session_state.last_analysis is None:
        st.warning("⚠️ Analyze a fruit image first in the Analyze tab.")
    else:
        last = st.session_state.last_analysis
        st.write(f"### Fruit: **{last.get('fruit', 'N/A')}**")
        st.write(f"### Ripeness: **{last.get('ripeness', 'N/A')}**")

        if st.button("📡 Read Sensor (ESP32 DHT11)"):
            if not SERIAL_AVAILABLE:
                st.error("❌ PySerial not installed. Run: pip install pyserial")
            else:
                try:
                    esp = serial.Serial("COM5", 115200, timeout=2)
                    time.sleep(1.5)
                    raw = read_clean_sensor(esp, max_lines=40)
                    esp.close()

                    if raw is None:
                        st.error("❌ No valid sensor data received.")
                    else:
                        temp, hum = parse_temp_hum(raw)
                        st.metric("🌡️ Temperature (°C)", f"{temp:.1f}" if temp else "N/A")
                        st.metric("💧 Humidity (%)", f"{hum:.1f}" if hum else "N/A")

                        shelf = combined_shelf_life(last["ripeness"], temp, hum)
                        st.success(f"📦 Estimated Shelf Life: {shelf}")

                        # ✅ UPDATE THE LAST ROW WITH TEMP/HUMIDITY
                        if len(st.session_state.results_history) > 0:
                            # Update the most recent entry
                            st.session_state.results_history[-1]["Temperature_C"] = temp
                            st.session_state.results_history[-1]["Humidity_pct"] = hum
                            st.session_state.results_history[-1]["Source"] = "Combined(IoT)"
                            
                            # Save updated CSV
                            pd.DataFrame(st.session_state.results_history).to_csv(CSV_FILE, index=False)
                            
                            st.success("✅ Temperature and humidity saved to CSV!")
                        
                except Exception as e:
                    st.error(f"❌ Serial error: {e}")

st.markdown("---")
st.markdown("Made with ❤️ using TensorFlow, MobileNetV2 & Streamlit (ESP32 IoT)")
st.info("📊 Go to the Dashboard page (sidebar) to view detailed analytics!")
